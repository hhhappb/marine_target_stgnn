from __future__ import annotations

import torch
import torch.nn as nn


class STGNNTemporalGate(nn.Module):
    """原 ST-GNN 的 TFE：沿慢时间维做门控卷积和 stride=2 压缩。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.update = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"STGNNTemporalGate 期望输入 [B, C, P, N]，实际为 {tuple(x.shape)}。")
        if x.size(1) != self.in_channels:
            raise ValueError(f"STGNNTemporalGate 通道数不匹配：期望 {self.in_channels}，实际 {x.size(1)}。")
        return torch.sigmoid(self.update(x)) * torch.tanh(self.output(x))
