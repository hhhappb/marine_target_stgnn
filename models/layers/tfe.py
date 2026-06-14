import torch
import torch.nn as nn


class TFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFELayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 4:
            batch, P, C, N = x.shape
            x = x.permute(0, 3, 2, 1).reshape(batch * N, C, P)
            x = self.conv1d(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x.reshape(batch, N, C, P).permute(0, 3, 2, 1)
            return x
        else:
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x.permute(0, 2, 1)
            return x
