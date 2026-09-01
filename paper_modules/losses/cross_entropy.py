from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntropyDetectionLoss(nn.Module):
    """训练目标基线：标准逐距离单元交叉熵。"""

    def __init__(self, class_weights: torch.Tensor | None = None, ignore_index: int = -100):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=int(ignore_index))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3 or labels.dim() != 2:
            raise ValueError(
                "CrossEntropyDetectionLoss 期望 logits=[B,C,N]、labels=[B,N]，"
                f"实际为 logits={tuple(logits.shape)}、labels={tuple(labels.shape)}。"
            )
        if logits.size(0) != labels.size(0) or logits.size(2) != labels.size(1):
            raise ValueError("CrossEntropyDetectionLoss 的 batch/range cell 维不一致。")
        # 展平逐距离单元分类，避免 CUDA nll_loss2d 的非确定性实现；损失定义不变。
        flat_logits = logits.permute(0, 2, 1).reshape(-1, logits.size(1))
        flat_labels = labels.reshape(-1)
        return self.loss(flat_logits, flat_labels)
