from __future__ import annotations

import torch

from .common import RangeAttentionBase


class DistanceDecayGraph(RangeAttentionBase):
    """空间思路 3：距离衰减动态图，远距离边默认受到物理距离惩罚。"""

    def __init__(self, in_channels: int, out_channels: int, decay: float = 0.25):
        super().__init__(in_channels, out_channels)
        self.decay = float(decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, shape = self.project(x)
        scores = self.attention_scores(q, k)
        scores = scores - self.decay * self.distance_matrix(shape[2], x.device).unsqueeze(0)
        return self.finish(torch.softmax(scores, dim=-1), v, shape)
