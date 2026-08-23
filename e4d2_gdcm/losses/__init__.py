"""E4-D2 loss functions."""

from e4d2_gdcm.losses.losses import (
    WeightedCrossEntropyLoss,
    FocalLoss,
    get_loss,
)

__all__ = ['WeightedCrossEntropyLoss', 'FocalLoss', 'get_loss']
