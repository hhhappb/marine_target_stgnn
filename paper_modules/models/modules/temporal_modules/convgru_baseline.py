from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUBaselineTemporal(nn.Module):
    """时间思路 1：原 ST-GNN 风格的卷积门控时间压缩。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            _TemporalGate(in_channels, hidden_channels),
            _TemporalGate(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=2)


class _TemporalGate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.update = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.update(x)) * torch.tanh(self.output(x))
