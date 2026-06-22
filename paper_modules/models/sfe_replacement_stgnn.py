from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DetectionHead, RadarFeatureEncoder, SpatialGraphModule


class SFEReplacementSTGNN(nn.Module):
    """严格对齐 ST-GNN 主干的模块替换实验骨架。

    稳定接口：
        输入：复数雷达回波 [B, P, N]
        输出：逐距离单元二分类 logits [B, 2, N]

    内部顺序对齐原始 ST-GNN：
        FT/输入特征 -> SFE1 -> TFE1 -> SFE2 -> TFE2 -> 检测头
    """

    def __init__(self, config: dict[str, object]):
        super().__init__()
        model_cfg = config.get("model", {})
        feature_cfg = config.get("radar_features", {})
        spatial_cfg = config.get("spatial_graph", {})
        temporal_cfg = config.get("temporal", {})
        gate_cfg = config.get("clutter_gate", {})
        head_cfg = config.get("detection_head", {})

        if bool(gate_cfg.get("enabled", False)):
            raise ValueError("论文主干替换实验当前不支持同时开启 clutter_gate。")

        temporal_type = str(temporal_cfg.get("type", "stgnn_tfe"))
        if temporal_type != "stgnn_tfe":
            raise ValueError(f"sfe_replacement_stgnn 只支持 temporal.type=stgnn_tfe，实际为 {temporal_type}。")

        self.pulses = int(model_cfg.get("pulses", 4))
        self.range_cells = int(model_cfg.get("range_cells", 14))

        feature_channels = int(feature_cfg.get("out_channels", 64))
        spatial1_channels = int(spatial_cfg.get("stage1_out_channels", 128))
        temporal1_channels = int(temporal_cfg.get("stage1_out_channels", 256))
        spatial2_channels = int(spatial_cfg.get("stage2_out_channels", 512))
        temporal2_channels = int(temporal_cfg.get("stage2_out_channels", temporal_cfg.get("out_channels", 1024)))

        self.radar_features = RadarFeatureEncoder(
            feature_type=str(feature_cfg.get("type", "real_imag")),
            hidden_channels=int(feature_cfg.get("hidden_channels", 32)),
            out_channels=feature_channels,
        )
        self.spatial_graph1 = self._build_spatial_graph(spatial_cfg, feature_channels, spatial1_channels)
        self.tfe1 = _STGNNTemporalGate(spatial1_channels, temporal1_channels)
        self.spatial_graph2 = self._build_spatial_graph(spatial_cfg, temporal1_channels, spatial2_channels)
        self.tfe2 = _STGNNTemporalGate(spatial2_channels, temporal2_channels)
        self.detection_head = DetectionHead(
            in_channels=temporal2_channels,
            hidden_channels=int(head_cfg.get("hidden_channels", 512)),
        )

    @staticmethod
    def _build_spatial_graph(config: dict[str, object], in_channels: int, out_channels: int) -> SpatialGraphModule:
        return SpatialGraphModule(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_type=str(config.get("type", "local_3")),
            k=int(config.get("k", 1)),
            use_distance_decay=bool(config.get("use_distance_decay", False)),
            distance_decay=float(config.get("distance_decay", 0.25)),
            dynamic_topk=int(config.get("dynamic_topk", 2)),
            static_gamma=float(config.get("static_gamma", 0.5)),
            static_delta=int(config.get("static_delta", 5)),
            static_weight=float(config.get("static_weight", 0.7)),
            dynamic_temperature=float(config.get("dynamic_temperature", 0.2)),
            dropout=float(config.get("dropout", 0.1)),
        )

    def forward(self, echoes: torch.Tensor, return_features: bool = False):
        if not torch.is_complex(echoes):
            raise TypeError("SFEReplacementSTGNN 需要复数输入 [B, P, N]。")
        if echoes.dim() != 3:
            raise ValueError(f"SFEReplacementSTGNN 期望输入 [B, P, N]，实际为 {tuple(echoes.shape)}。")
        if echoes.size(1) != self.pulses or echoes.size(2) != self.range_cells:
            raise ValueError(
                f"输入形状 [B, {echoes.size(1)}, {echoes.size(2)}] 与配置 "
                f"pulses/range_cells [{self.pulses}, {self.range_cells}] 不一致。"
            )

        x = self.radar_features(echoes)
        spatial1 = F.relu(self.spatial_graph1(x))
        temporal1 = self.tfe1(spatial1)
        spatial2 = F.relu(self.spatial_graph2(temporal1))
        temporal2 = self.tfe2(spatial2)
        # P=4 时 TFE2 后时间维为 1；P=16/32 观测时间消融时用均值压缩保持检测接口不变。
        temporal_features = temporal2.mean(dim=2)
        logits = self.detection_head(temporal_features)

        if return_features:
            return logits, {
                "radar_features": x,
                "spatial1": spatial1,
                "temporal1": temporal1,
                "spatial2": spatial2,
                "temporal2": temporal2,
            }
        return logits


class _STGNNTemporalGate(nn.Module):
    """原 ST-GNN 的 TFE 门控压缩：沿 pulse 维做 stride=2 时间压缩。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.update = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.update(x)) * torch.tanh(self.output(x))
