from __future__ import annotations

import torch.nn as nn

from .diff_bicam_tfe import DiffBiCAMTFE
from .stgnn_tfe import STGNNTemporalGate


def build_temporal_module(config: dict[str, object], in_channels: int, out_channels: int) -> nn.Module:
    temporal_type = str(config.get("type", "stgnn_tfe"))
    if temporal_type == "stgnn_tfe":
        return STGNNTemporalGate(in_channels, out_channels)
    if temporal_type == "diff_bicam_tfe":
        return DiffBiCAMTFE(
            in_channels,
            out_channels,
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.1)),
            use_diff=bool(config.get("use_diff", True)),
            use_coattention=bool(config.get("use_coattention", True)),
            use_prev=bool(config.get("use_prev", True)),
            use_next=bool(config.get("use_next", True)),
            shuffle_diff=bool(config.get("shuffle_diff", False)),
        )
    raise ValueError(f"Unknown temporal module type: {temporal_type}")


__all__ = ["build_temporal_module", "DiffBiCAMTFE", "STGNNTemporalGate"]
