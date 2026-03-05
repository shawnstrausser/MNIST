"""
cnn.py — Convolutional Neural Network for MNIST.

Unlike the fully-connected model (simple_fc.py), a CNN looks at small
patches of the image at a time using sliding filters (convolutions).
This lets it learn spatial patterns like edges, curves, and loops —
exactly the kind of features that distinguish handwritten digits.

Architecture:
  Conv(1→32, 3x3) → ReLU → Conv(32→64, 3x3) → ReLU → MaxPool(2x2)
  → Flatten → Linear(9216→128) → ReLU → Linear(128→10)

Typically reaches ~99% test accuracy — a nice bump over the ~97% FC model.
"""

import torch.nn as nn


class CNNNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 → 28x28, 32 filters
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 28x28 → 28x28, 64 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 → 14x14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                # 64 * 14 * 14 = 12544
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
