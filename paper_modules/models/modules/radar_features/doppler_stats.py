from __future__ import annotations

import torch
import torch.nn as nn

from .phase_diffs import PhaseAmplitudeDiffFeatures


class DopplerStatsFeatures(nn.Module):
    """输入思路 4：加入短慢时间 Doppler 统计，恢复轻量频域先验。"""

    out_channels = 11

    def __init__(self):
        super().__init__()
        self.base = PhaseAmplitudeDiffFeatures()

    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        base = self.base(echoes)
        doppler_power = torch.abs(torch.fft.fft(echoes, dim=1)) ** 2
        peak = doppler_power.max(dim=1, keepdim=True).values
        mean = doppler_power.mean(dim=1, keepdim=True)
        ratio = peak / (mean + 1e-6)
        stats = torch.stack(
            [
                peak.expand(-1, echoes.size(1), -1),
                mean.expand(-1, echoes.size(1), -1),
                ratio.expand(-1, echoes.size(1), -1),
            ],
            dim=1,
        )
        return torch.cat([base, stats], dim=1)
