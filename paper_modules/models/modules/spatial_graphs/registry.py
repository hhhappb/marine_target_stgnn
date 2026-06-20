from __future__ import annotations

import torch.nn as nn

from .distance_decay import DistanceDecayGraph
from .dynamic_attention import DynamicAttentionGraph
from .local_range import LocalRangeGraph
from .prior_dynamic import PriorDynamicRangeGraph


def build_spatial_graph(config: dict[str, object], in_channels: int, out_channels: int) -> nn.Module:
    graph_type = str(config.get("type", "local_3"))
    if graph_type == "local_3":
        return LocalRangeGraph(in_channels, out_channels, radius=1)
    if graph_type == "local_k":
        return LocalRangeGraph(in_channels, out_channels, radius=int(config.get("k", 1)))
    if graph_type == "pure_dynamic":
        return DynamicAttentionGraph(in_channels, out_channels)
    if graph_type == "distance_dynamic":
        return DistanceDecayGraph(in_channels, out_channels, decay=float(config.get("distance_decay", 0.25)))
    if graph_type == "local_plus_dynamic_prior":
        return PriorDynamicRangeGraph(
            in_channels,
            out_channels,
            radius=int(config.get("k", 1)),
            decay=float(config.get("distance_decay", 0.25)),
            similarity_weight=float(config.get("similarity_weight", 0.1)),
        )
    raise ValueError(f"Unknown spatial graph type: {graph_type}")
