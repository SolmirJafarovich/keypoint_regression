import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class CombinedClassifier(nn.Module):
    def __init__(self, keypoint_dim=66, num_classes=4):
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
            nn.Linear(128, num_classes),
        )

    def forward(self, img, kps):
        x = self.features(img)
        x = self.image_pool(x).view(x.size(0), -1)
        k = self.keypoint_proj(kps)
        combined = torch.cat([x, k], dim=1)
        return self.classifier(combined)


class ClassifierWrapper:
    pass
