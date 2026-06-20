from __future__ import annotations

import torch
import torch.nn as nn


class RangeMigrationTemporal(nn.Module):
    """时间思路 3：跨距离单元时间混合，允许目标存在轻微 range migration。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=2)
