from __future__ import annotations

import torch
import torch.nn as nn

from .cross_entropy import CrossEntropyDetectionLoss


class PfaAwareDetectionLoss(nn.Module):
    """训练目标思路 1：惩罚杂波样本中目标概率最高的尾部，贴近低虚警检测目标。"""

    def __init__(self, class_weights: torch.Tensor | None = None, tail_fraction: float = 0.01, tail_weight: float = 0.1):
        super().__init__()
        self.ce = CrossEntropyDetectionLoss(class_weights)
        self.tail_fraction = float(tail_fraction)
        self.tail_weight = float(tail_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, labels)
        clutter_mask = labels == 0
        if not torch.any(clutter_mask):
            return loss
        target_probs = torch.softmax(logits, dim=1)[:, 1, :]
        clutter_scores = target_probs[clutter_mask]
        k = max(1, int(self.tail_fraction * clutter_scores.numel()))
        tail_scores = torch.topk(clutter_scores, k).values
        return loss + self.tail_weight * tail_scores.mean()
