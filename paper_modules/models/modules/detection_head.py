from __future__ import annotations

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """逐距离单元检测头：输入 [B, C, N]，输出二分类 logits [B, 2, N]。"""

    def __init__(self, in_channels: int, hidden_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
