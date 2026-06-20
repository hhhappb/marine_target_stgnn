from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleTCNTemporal(nn.Module):
    """时间思路 2：多尺度时间卷积，同时看短期、中期和较长脉冲变化。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        branch_channels = max(1, hidden_channels // 3)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
                nn.Conv2d(in_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
                nn.Conv2d(in_channels, branch_channels, kernel_size=(5, 1), padding=(2, 0)),
            ]
        )
        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1)).mean(dim=2)
