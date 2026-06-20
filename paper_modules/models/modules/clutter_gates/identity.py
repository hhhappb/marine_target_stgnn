from __future__ import annotations

import torch
import torch.nn as nn


class IdentityClutterGate(nn.Module):
    """杂波思路基线：不使用海杂波统计门控。"""

    def forward(self, features: torch.Tensor, echoes: torch.Tensor | None = None) -> torch.Tensor:
        return features
