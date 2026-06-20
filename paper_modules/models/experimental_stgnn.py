from __future__ import annotations

import torch
import torch.nn as nn

from .modules import ClutterAwareGate, DetectionHead, RadarFeatureEncoder, SpatialGraphModule, TemporalModule


class ExperimentalSTGNN(nn.Module):
    """可配置的新论文 ST-GNN 实验骨架。

    稳定接口：
        输入：复数雷达回波 [B, P, N]
        输出：逐距离单元二分类 logits [B, 2, N]

    这里不写具体创新细节，只负责按配置组装：
        雷达特征 -> 空间图 -> 杂波门控 -> 时间建模 -> 检测头
    """

    def __init__(self, config: dict[str, object]):
        super().__init__()
        model_cfg = config.get("model", {})
        feature_cfg = config.get("radar_features", {})
        spatial_cfg = config.get("spatial_graph", {})
        temporal_cfg = config.get("temporal", {})
        gate_cfg = config.get("clutter_gate", {})
        head_cfg = config.get("detection_head", {})

        self.pulses = int(model_cfg.get("pulses", 4))
        self.range_cells = int(model_cfg.get("range_cells", 14))

        feature_channels = int(feature_cfg.get("out_channels", 64))
        spatial_channels = int(spatial_cfg.get("out_channels", 128))
        temporal_channels = int(temporal_cfg.get("out_channels", 1024))

        self.radar_features = RadarFeatureEncoder(
            feature_type=str(feature_cfg.get("type", "real_imag")),
            hidden_channels=int(feature_cfg.get("hidden_channels", 32)),
            out_channels=feature_channels,
        )
        self.spatial_graph = SpatialGraphModule(
            in_channels=feature_channels,
            out_channels=spatial_channels,
            graph_type=str(spatial_cfg.get("type", "local_3")),
            k=int(spatial_cfg.get("k", 1)),
            use_distance_decay=bool(spatial_cfg.get("use_distance_decay", False)),
            distance_decay=float(spatial_cfg.get("distance_decay", 0.25)),
            use_feature_similarity=bool(spatial_cfg.get("use_feature_similarity", False)),
            similarity_weight=float(spatial_cfg.get("similarity_weight", 0.1)),
        )
        self.clutter_gate = ClutterAwareGate(
            enabled=bool(gate_cfg.get("enabled", False)),
            gate_type=str(gate_cfg.get("type", "local_statistics")),
            strength=float(gate_cfg.get("strength", 0.25)),
            window=int(gate_cfg.get("window", 5)),
        )
        self.temporal = TemporalModule(
            in_channels=spatial_channels,
            hidden_channels=int(temporal_cfg.get("hidden_channels", 256)),
            out_channels=temporal_channels,
            temporal_type=str(temporal_cfg.get("type", "convgru_baseline")),
        )
        self.detection_head = DetectionHead(
            in_channels=temporal_channels,
            hidden_channels=int(head_cfg.get("hidden_channels", 512)),
        )

    def forward(self, echoes: torch.Tensor, return_features: bool = False):
        x = self.radar_features(echoes)
        spatial = self.spatial_graph(x)
        spatial = self.clutter_gate(spatial, echoes)
        temporal = self.temporal(spatial)
        logits = self.detection_head(temporal)
        if return_features:
            return logits, {"radar_features": x, "spatial": spatial, "temporal": temporal}
        return logits
