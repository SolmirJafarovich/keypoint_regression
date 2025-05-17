from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from src.config import config


class CombinedClassifier(nn.Module):
    def __init__(self, keypoint_dim=66):
        super().__init__()
        base = mobilenet_v2(weights=None)
        base.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.features = base.features
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.keypoint_proj = nn.Linear(keypoint_dim, 64)
        self.classifier = nn.Sequential(
            nn.Linear(base.last_channel + 64, 128),
            nn.ReLU(),
            nn.Linear(128, config.classifier.num_classes),
        )

    def forward(self, img, kps):
        x = self.features(img)
        x = self.image_pool(x).view(x.size(0), -1)
        k = self.keypoint_proj(kps)
        combined = torch.cat([x, k], dim=1)
        return self.classifier(combined)


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, tflite_model_path: Path):
        import tensorflow as tf

        super().__init__()
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Check input count
        assert len(self.input_details) == 2, "Expected 2 inputs: image and keypoints"

        # Cache input indices for clarity
        self.input_index_img = self.input_details[0]["index"]
        self.input_index_kps = self.input_details[1]["index"]

    def forward(self, img: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
        # Convert input to numpy
        img_np = img.detach().cpu().numpy().astype(self.input_details[0]["dtype"])
        kps_np = kps.detach().cpu().numpy().astype(self.input_details[1]["dtype"])
        kps_np = kps_np.reshape(1, -1)


        # Set input tensors
        self.interpreter.set_tensor(self.input_index_img, img_np)
        self.interpreter.set_tensor(self.input_index_kps, kps_np)
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return torch.from_numpy(output_data)
