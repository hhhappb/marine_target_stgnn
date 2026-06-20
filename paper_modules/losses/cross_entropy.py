from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntropyDetectionLoss(nn.Module):
    """训练目标基线：标准逐距离单元交叉熵。"""

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, labels)
