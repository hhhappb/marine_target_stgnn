from __future__ import annotations

import torch

from .common import RangeAttentionBase


class DynamicAttentionGraph(RangeAttentionBase):
    """空间思路 2：纯动态注意力图，让所有距离单元自适应建立联系。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, shape = self.project(x)
        scores = self.attention_scores(q, k)
        return self.finish(torch.softmax(scores, dim=-1), v, shape)
