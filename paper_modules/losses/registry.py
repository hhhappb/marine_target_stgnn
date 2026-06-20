from __future__ import annotations

import torch
import torch.nn as nn

from .cross_entropy import CrossEntropyDetectionLoss
from .pfa_aware import PfaAwareDetectionLoss


def build_loss(config: dict[str, object], class_weights: torch.Tensor | None = None) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = str(loss_cfg.get("type", "cross_entropy"))
    if loss_type == "cross_entropy":
        return CrossEntropyDetectionLoss(class_weights)
    if loss_type == "pfa_aware":
        return PfaAwareDetectionLoss(
            class_weights,
            tail_fraction=float(loss_cfg.get("tail_fraction", 0.01)),
            tail_weight=float(loss_cfg.get("tail_weight", 0.1)),
        )
    raise ValueError(f"Unknown loss type: {loss_type}")
