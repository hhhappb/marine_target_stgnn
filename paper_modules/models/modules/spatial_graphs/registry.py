from __future__ import annotations

import torch.nn as nn

from .distance_decay import DistanceDecayGraph
from .dynamic_attention import DynamicAttentionGraph
from .local_range import LocalRangeGraph
from .original_stfe import OriginalSTFEGraph
from .radar_prior_dynamic_sfe import RadarPriorDynamicSFE


def build_spatial_graph(config: dict[str, object], in_channels: int, out_channels: int) -> nn.Module:
    graph_type = str(config.get("type", "local_3"))
    if graph_type == "local_3":
        return LocalRangeGraph(in_channels, out_channels, radius=1)
    if graph_type == "original_stfe":
        return OriginalSTFEGraph(in_channels, out_channels)
    if graph_type == "local_k":
        return LocalRangeGraph(in_channels, out_channels, radius=int(config.get("k", 1)))
    if graph_type == "pure_dynamic":
        return DynamicAttentionGraph(in_channels, out_channels)
    if graph_type == "distance_dynamic":
        return DistanceDecayGraph(in_channels, out_channels, decay=float(config.get("distance_decay", 0.25)))
    if graph_type == "radar_prior_dynamic_sfe":
        return RadarPriorDynamicSFE(
            in_channels,
            out_channels,
            static_gamma=float(config.get("static_gamma", 0.5)),
            static_delta=int(config.get("static_delta", 5)),
            static_weight=float(config.get("static_weight", 0.7)),
            dynamic_topk=int(config.get("dynamic_topk", 2)),
            dynamic_temperature=float(config.get("dynamic_temperature", 0.2)),
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown spatial graph type: {graph_type}")
