import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, hidden_channels=32, out_channels=64):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(2, hidden_channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, E_real, E_imag):
        E = torch.stack([E_real, E_imag], dim=1)

        x = self.conv1(E)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x
