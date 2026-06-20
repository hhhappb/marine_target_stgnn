from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolTemporal(nn.Module):
    """时间消融思路：不做显式时间建模，只对脉冲维做平均池化。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x.mean(dim=2))
