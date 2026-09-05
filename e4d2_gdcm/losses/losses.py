"""Loss functions for radar target detection.

Standard protocol: unweighted CrossEntropyLoss averaged over all
range bins within a batch (per ST-GNN paper Section III-C).
For imbalanced datasets, weighted CE or Focal Loss can be used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights, averaged over all range bins.

    Args:
        class_weight: [w_clutter, w_target], e.g. [1.0, 10.0]
    """

    def __init__(self, class_weight=(1.0, 10.0)):
        super().__init__()
        self.register_buffer('weight', torch.tensor(class_weight))

    def forward(self, logits, labels):
        """logits: [B, 2, N], labels: [B, N]"""
        B, _, N = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, 2)
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(logits_flat, labels_flat, weight=self.weight)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        alpha: weight for target class (clutter weight = 1 - alpha)
        gamma: focusing parameter (default 2.0)
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        """logits: [B, 2, N], labels: [B, N]"""
        ce = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, 2),
            labels.reshape(-1), reduction='none',
        )
        pt = torch.exp(-ce)
        alpha_t = self.alpha * labels.float().reshape(-1) + \
                  (1 - self.alpha) * (1 - labels.float().reshape(-1))
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


def get_loss(name='ce', **kwargs):
    """Factory for loss functions.

    Args:
        name: 'ce' (unweighted), 'wce' (weighted CE), 'focal'
        **kwargs: passed to loss constructor

    Returns:
        nn.Module loss function
    """
    if name == 'ce':
        return nn.CrossEntropyLoss()
    elif name == 'wce':
        weight = kwargs.get('weight', (1.0, 10.0))
        return WeightedCrossEntropyLoss(weight)
    elif name == 'focal':
        return FocalLoss(**{k: v for k, v in kwargs.items() if k in ('alpha', 'gamma')})
    else:
        raise ValueError(f"Unknown loss: {name}")
