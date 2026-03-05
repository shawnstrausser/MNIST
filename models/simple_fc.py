"""
simple_fc.py — A simple fully-connected (dense) neural network.

This is the simplest architecture for MNIST: flatten each 28x28 image into
a 784-number list, then pass it through three layers that progressively
narrow down to 10 output scores — one per digit (0-9).

Architecture:  784 → 128 → 64 → 10

The highest score wins: if output[3] is the largest, the model predicts "3".
Achieves ~97% test accuracy — solid for such a simple design.
"""

import torch.nn as nn


class SimpleFCNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),           # 28x28 image → flat list of 784 numbers
            nn.Linear(28 * 28, 128),  # 784 → 128
            nn.ReLU(),              # zero out negatives (adds non-linearity)
            nn.Linear(128, 64),     # 128 → 64
            nn.ReLU(),
            nn.Linear(64, 10),      # 64 → 10 digit scores
        )

    def forward(self, x):
        return self.net(x)
