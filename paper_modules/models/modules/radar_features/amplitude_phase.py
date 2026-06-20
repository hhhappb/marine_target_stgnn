from __future__ import annotations

import torch
import torch.nn as nn


class AmplitudePhaseFeatures(nn.Module):
    """输入思路 2：在实部/虚部基础上显式加入幅度和相位表示。"""

    out_channels = 5

    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(echoes):
            raise TypeError("AmplitudePhaseFeatures 需要复数输入，形状为 [B, P, N]。")
        phase = torch.angle(echoes)
        return torch.stack(
            [
                echoes.real,
                echoes.imag,
                torch.abs(echoes),
                torch.sin(phase),
                torch.cos(phase),
            ],
            dim=1,
        )
