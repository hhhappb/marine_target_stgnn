from __future__ import annotations

import torch
import torch.nn as nn

from .amplitude_phase import AmplitudePhaseFeatures


class PhaseAmplitudeDiffFeatures(nn.Module):
    """输入思路 3：加入相邻脉冲幅度差和相位差，显式建模幅相稳定性。"""

    out_channels = 8

    def __init__(self):
        super().__init__()
        self.base = AmplitudePhaseFeatures()

    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        base = self.base(echoes)
        amp = torch.abs(echoes)
        phase_diff = torch.zeros_like(amp)
        amp_diff = torch.zeros_like(amp)
        if echoes.size(1) > 1:
            amp_diff[:, 1:, :] = amp[:, 1:, :] - amp[:, :-1, :]
            phase_diff[:, 1:, :] = torch.angle(echoes[:, 1:, :] * torch.conj(echoes[:, :-1, :]))
        diffs = torch.stack([amp_diff, torch.sin(phase_diff), torch.cos(phase_diff)], dim=1)
        return torch.cat([base, diffs], dim=1)
