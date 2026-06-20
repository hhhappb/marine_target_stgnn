from __future__ import annotations

import torch
import torch.nn as nn


class RealImagFeatures(nn.Module):
    """输入思路 1：原 ST-GNN 基线，只使用复数回波的实部和虚部。"""

    out_channels = 2

    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(echoes):
            raise TypeError("RealImagFeatures 需要复数输入，形状为 [B, P, N]。")
        return torch.stack([echoes.real, echoes.imag], dim=1)
