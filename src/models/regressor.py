from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Depthwise Separable Convolution ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


# --- Main Model ---
class BlazePoseLite(nn.Module):
    def __init__(self, heatmap_size=64, num_keypoints=33):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints

        # --- Encoder ---
        self.enc1 = DepthwiseSeparableConv(1, 16, stride=2)  # 128 -> 64
        self.enc2 = DepthwiseSeparableConv(16, 32, stride=2)  # 64 -> 32
        self.enc3 = DepthwiseSeparableConv(32, 64, stride=2)  # 32 -> 16
        self.enc4 = DepthwiseSeparableConv(64, 128, stride=2)  # 16 -> 8
        self.enc5 = DepthwiseSeparableConv(128, 192, stride=2)  # 8 -> 4

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ResidualBlock(192),
            ResidualBlock(192),
            ResidualBlock(192),
        )

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(192 + 128, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128 + 64, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64 + 32, 32, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(32 + 16, 32, kernel_size=2, stride=2)

        # --- Heatmap Head ---
        self.heatmap_out = nn.Conv2d(32, num_keypoints, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)  # [B, 16, 64, 64]
        e2 = self.enc2(e1)  # [B, 32, 32, 32]
        e3 = self.enc3(e2)  # [B, 64, 16, 16]
        e4 = self.enc4(e3)  # [B, 128, 8, 8]
        e5 = self.enc5(e4)  # [B, 192, 4, 4]

        b = self.bottleneck(e5)

        b_up = F.interpolate(b, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.up4(torch.cat([b_up, e4], dim=1))  # 4 -> 8
        d3 = self.up3(torch.cat([d4, e3], dim=1))  # 8 -> 16
        d2 = self.up2(torch.cat([d3, e2], dim=1))  # 16 -> 32
        d1 = self.up1(torch.cat([d2, e1], dim=1))  # 32 -> 64

        # heatmaps = self.heatmap_out(d1)  # [B, K, 64, 64]
        # heatmaps = 10 * heatmaps  # масштабирование логитов

        logits = self.heatmap_out(d1)
        heatmaps = logits  # усиливаем контраст логитов

        # Softmax по пространству
        # B, K, H, W = heatmaps.shape
        # heatmaps = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)

        # heatmaps = self.heatmap_out(d1)
        heatmaps = F.interpolate(
            heatmaps,
            size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear",
            align_corners=False,
        )

        heatmaps = 15 * heatmaps  # усиление контраста (проверено)

        # Softmax по пространству
        # B, K, H, W = heatmaps.shape
        # heatmaps = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)
        return heatmaps


class RegressorWrapper(torch.nn.Module):
    def __init__(self, tflite_model_path: Path):
        import tensorflow as tf

        super().__init__()
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to numpy\
        scale, zero_point = self.input_details[0]["quantization"]

        input_data = x.detach().cpu().numpy()
        input_data = input_data / scale + zero_point
        input_data = np.clip(input_data, 0, 255).astype(self.input_details[0]["dtype"])

        print("Input shape:", input_data.shape)
        print("Input dtype:", input_data.dtype)
        print("Input min/max:", input_data.min(), input_data.max())
        print(
            "Input has NaN or Inf:",
            np.isnan(input_data).any(),
            np.isinf(input_data).any(),
        )
        print("Expected input shape:", self.input_details[0]["shape"])
        print("Input quantization:", self.input_details[0]["quantization"])
        print("Output quantization:", self.output_details[0]["quantization"])
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        print("Output shape:", output_data.shape)
        print("Output min/max:", output_data.min(), output_data.max())
        print(
            "Output has NaN or Inf:",
            np.isnan(output_data).any(),
            np.isinf(output_data).any(),
        )
        return torch.from_numpy(output_data)
