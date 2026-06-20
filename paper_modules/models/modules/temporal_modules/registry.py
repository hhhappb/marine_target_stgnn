from __future__ import annotations

import torch.nn as nn

from .convgru_baseline import ConvGRUBaselineTemporal
from .multiscale_tcn import MultiScaleTCNTemporal
from .range_migration import RangeMigrationTemporal
from .temporal_pool import MeanPoolTemporal


TEMPORAL_MODULES: dict[str, type[nn.Module]] = {
    "convgru_baseline": ConvGRUBaselineTemporal,
    "multiscale_tcn": MultiScaleTCNTemporal,
    "range_migration": RangeMigrationTemporal,
    "mean_pool": MeanPoolTemporal,
}


def build_temporal_module(temporal_type: str, in_channels: int, hidden_channels: int, out_channels: int) -> nn.Module:
    try:
        cls = TEMPORAL_MODULES[temporal_type]
    except KeyError as exc:
        known = ", ".join(sorted(TEMPORAL_MODULES))
        raise ValueError(f"Unknown temporal module type: {temporal_type}. Known: {known}") from exc
    return cls(in_channels, hidden_channels, out_channels)
