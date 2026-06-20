from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import RangeAttentionBase


class PriorDynamicRangeGraph(RangeAttentionBase):
    """空间思路 4：雷达先验动态图，融合局部邻接、距离衰减和特征相似性。"""

    def __init__(self, in_channels: int, out_channels: int, radius: int = 1, decay: float = 0.25, similarity_weight: float = 0.1):
        super().__init__(in_channels, out_channels)
        self.radius = max(1, int(radius))
        self.decay = float(decay)
        self.similarity_weight = float(similarity_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, pulses, ranges = x.shape
        h = x.permute(0, 2, 3, 1).reshape(bsz * pulses, ranges, channels)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = self.attention_scores(q, k)
        scores = scores - self.decay * self.distance_matrix(ranges, x.device).unsqueeze(0)
        local_bias = self.local_mask(ranges, x.device, self.radius).float() * 0.5
        similarity = torch.matmul(F.normalize(h, dim=-1), F.normalize(h, dim=-1).transpose(-1, -2))
        scores = scores + local_bias.unsqueeze(0) + self.similarity_weight * similarity
        return self.finish(torch.softmax(scores, dim=-1), v, (bsz, pulses, ranges, q.size(-1)))
