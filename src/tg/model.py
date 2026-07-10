"""A small CNN for 32x32 single-channel images (Yearbook-style).

Kept intentionally tiny so a full sequential-training + all-methods sweep runs
in minutes on an Apple-Silicon MacBook Pro (MPS) or even CPU.
"""
from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> width*2 x 1 x 1
        )
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h)


def build_model(cfg: dict) -> SmallCNN:
    return SmallCNN(
        in_channels=cfg.get("in_channels", 1),
        num_classes=cfg.get("num_classes", 2),
        width=cfg.get("width", 32),
    )
