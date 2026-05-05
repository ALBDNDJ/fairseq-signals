import torch, torch.nn as nn

class SimpleCNN1D(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):  # x: [B, 12, T]
        feat = self.net(x).squeeze(-1)
        return self.fc(feat)