import torch.nn as nn


class SimpleFCNet(nn.Module):
    """Simple fully-connected network: 784 -> 128 -> 64 -> 10"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)
