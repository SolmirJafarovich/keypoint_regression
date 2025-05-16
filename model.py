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


def soft_argmax_2d(heatmaps):
    B, C, H, W = heatmaps.shape
    flat = heatmaps.view(B, C, -1)
    probs = F.softmax(flat, dim=2)

    xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    xv, yv = xv.reshape(-1), yv.reshape(-1)

    x = torch.sum(probs * xv, dim=2)
    y = torch.sum(probs * yv, dim=2)
    return torch.stack([x, y], dim=2)
