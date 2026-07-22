from __future__ import annotations

import torch.nn as nn

from .diff_tfe import DiffTFE
from .fixed_uniform_tfe import FixedUniformTemporalMixerTFE
from .lag_aware_anti_alias_tfe import LagAwareAntiAliasTFE
from .pulse_attention_tfe import PulseAttentionOnlyTFE
from .stgnn_tfe import STGNNTemporalGate


def build_temporal_module(config: dict[str, object], in_channels: int, out_channels: int) -> nn.Module:
    temporal_type = str(config.get("type", "stgnn_tfe"))
    if temporal_type == "stgnn_tfe":
        return STGNNTemporalGate(in_channels, out_channels)
    if temporal_type == "diff_tfe":
        return DiffTFE(
            in_channels,
            out_channels,
            diff_scale=float(config.get("diff_scale", 0.1)),
            use_diff=bool(config.get("use_diff", True)),
        )
    if temporal_type == "pulse_attention_only_tfe":
        return PulseAttentionOnlyTFE(
            in_channels,
            out_channels,
            attention_dim=int(config.get("attention_dim", 64)),
            num_heads=int(config.get("num_heads", 4)),
            residual_scale=float(config.get("residual_scale", 0.1)),
            use_attention=bool(config.get("use_attention", True)),
        )
    if temporal_type == "fixed_uniform_tfe":
        return FixedUniformTemporalMixerTFE(
            in_channels,
            out_channels,
            mixer_dim=int(config.get("mixer_dim", 64)),
            residual_scale=float(config.get("residual_scale", 0.1)),
            use_mixer=bool(config.get("use_mixer", True)),
        )
    if temporal_type == "lag_aware_anti_alias_tfe":
        return LagAwareAntiAliasTFE(
            in_channels,
            out_channels,
            kernel_size=int(config.get("kernel_size", 3)),
            filter_groups=int(config.get("filter_groups", 1)),
            gamma_init=float(config.get("gamma_init", 0.05)),
            gamma_max=float(config.get("gamma_max", 1.0)),
            use_filter=bool(config.get("use_filter", True)),
            padding_mode=str(config.get("padding_mode", "replicate")),
        )
    raise ValueError(f"Unknown temporal module type: {temporal_type}")


__all__ = [
    "build_temporal_module",
    "DiffTFE",
    "FixedUniformTemporalMixerTFE",
    "LagAwareAntiAliasTFE",
    "PulseAttentionOnlyTFE",
    "STGNNTemporalGate",
]
