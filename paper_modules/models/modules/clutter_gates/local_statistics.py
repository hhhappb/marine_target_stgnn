from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalStatisticsClutterGate(nn.Module):
    """杂波思路 1：利用局部幅度统计抑制疑似 sea spike 的消息传播。"""

    def __init__(self, strength: float = 0.25, window: int = 5):
        super().__init__()
        self.strength = float(strength)
        self.window = int(window)

    def forward(self, features: torch.Tensor, echoes: torch.Tensor | None = None) -> torch.Tensor:
        if echoes is None:
            return features
        amp = torch.abs(echoes)
        pad = self.window // 2
        flat = amp.reshape(-1, 1, amp.size(-1))
        local_mean = F.avg_pool1d(flat, self.window, stride=1, padding=pad).reshape_as(amp)
        local_power = F.avg_pool1d(flat * flat, self.window, stride=1, padding=pad).reshape_as(amp)
        local_std = (local_power - local_mean * local_mean).clamp_min(0.0).sqrt()
        risk = torch.sigmoid((amp - local_mean).abs() / (local_std + 1e-6))
        gate = (1.0 - self.strength * risk).clamp_min(0.0)
        if features.dim() == 4:
            return features * gate.unsqueeze(1)
        return features * gate.mean(dim=1).unsqueeze(1)
