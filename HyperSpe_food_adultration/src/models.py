from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ImageAdulterationClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(num_classes: int = 4, device: str = 'cpu') -> nn.Module:
    model = ImageAdulterationClassifier(num_classes=num_classes, pretrained=True)
    return model.to(device)
