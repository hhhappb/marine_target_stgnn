from __future__ import annotations

import torch

from .common import RangeAttentionBase


class LocalRangeGraph(RangeAttentionBase):
    """空间思路 1：固定局部距离图，对齐原 ST-GNN 的相邻距离单元先验。"""

    def __init__(self, in_channels: int, out_channels: int, radius: int = 1):
        super().__init__(in_channels, out_channels)
        self.radius = max(1, int(radius))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, shape = self.project(x)
        scores = self.attention_scores(q, k)
        mask = self.local_mask(shape[2], x.device, self.radius)
        scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
        return self.finish(torch.softmax(scores, dim=-1), v, shape)
