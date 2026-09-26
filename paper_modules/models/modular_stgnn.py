from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DetectionHead, RadarFeatureEncoder, SpatialGraphModule, TemporalModule


class ModularSTGNN(nn.Module):
    """ST-GNN 可插拔实验主干。

    四个核心槽位可分别配置：SFE1、TFE1、SFE2、TFE2。未替换槽位使用
    论文对齐模块，输入输出接口保持 [B,C,P,N]。
    """

    def __init__(self, config: dict[str, object]):
        super().__init__()
        model_cfg = config.get("model", {})
        feature_cfg = config.get("radar_features", {})
        spatial_cfg = config.get("spatial_graph", {})
        spatial1_override = config.get("spatial1", {})
        spatial2_override = config.get("spatial2", {})
        temporal_cfg = config.get("temporal", {})
        temporal1_override = config.get("temporal1", {})
        temporal2_override = config.get("temporal2", {})
        gate_cfg = config.get("clutter_gate", {})
        head_cfg = config.get("detection_head", {})

        mappings = {
            "model": model_cfg,
            "radar_features": feature_cfg,
            "spatial_graph": spatial_cfg,
            "spatial1": spatial1_override,
            "spatial2": spatial2_override,
            "temporal": temporal_cfg,
            "temporal1": temporal1_override,
            "temporal2": temporal2_override,
            "clutter_gate": gate_cfg,
            "detection_head": head_cfg,
        }
        invalid = [name for name, value in mappings.items() if not isinstance(value, dict)]
        if invalid:
            raise ValueError(f"以下配置必须是 mapping：{', '.join(invalid)}。")
        if bool(gate_cfg.get("enabled", False)):
            raise ValueError("模块替换实验当前不支持同时开启 clutter_gate。")

        spatial1_cfg = {**spatial_cfg, **spatial1_override}
        spatial2_cfg = {**spatial_cfg, **spatial2_override}
        temporal1_cfg = {**temporal_cfg, **temporal1_override}
        temporal2_cfg = {**temporal_cfg, **temporal2_override}

        self.pulses = int(model_cfg.get("pulses", 4))
        self.range_cells = int(model_cfg.get("range_cells", 14))

        feature_channels = int(feature_cfg.get("out_channels", 64))
        spatial1_channels = int(spatial1_cfg.get("stage1_out_channels", 128))
        temporal1_channels = int(temporal1_cfg.get("stage1_out_channels", 256))
        spatial2_channels = int(spatial2_cfg.get("stage2_out_channels", 512))
        temporal2_channels = int(
            temporal2_cfg.get(
                "stage2_out_channels",
                temporal2_cfg.get("out_channels", 1024),
            )
        )

        self.radar_features = RadarFeatureEncoder(
            feature_type=str(feature_cfg.get("type", "real_imag")),
            hidden_channels=int(feature_cfg.get("hidden_channels", 32)),
            out_channels=feature_channels,
        )
        self.spatial_graph1 = self._build_spatial_graph(
            spatial1_cfg, feature_channels, spatial1_channels
        )
        self.tfe1 = TemporalModule(
            temporal1_cfg, spatial1_channels, temporal1_channels
        )
        self.spatial_graph2 = self._build_spatial_graph(
            spatial2_cfg, temporal1_channels, spatial2_channels
        )
        self.tfe2 = TemporalModule(
            temporal2_cfg, spatial2_channels, temporal2_channels
        )
        self.detection_head = DetectionHead(
            in_channels=temporal2_channels,
            hidden_channels=int(head_cfg.get("hidden_channels", 512)),
        )

    @staticmethod
    def _build_spatial_graph(
        config: dict[str, object],
        in_channels: int,
        out_channels: int,
    ) -> SpatialGraphModule:
        if "dynamic_topk" in config:
            raise ValueError("dynamic_topk 已移除；动态图仅使用局部距离窗口。")
        return SpatialGraphModule(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_type=str(config.get("type", "local_3")),
            k=int(config.get("k", 1)),
            use_distance_decay=bool(config.get("use_distance_decay", False)),
            distance_decay=float(config.get("distance_decay", 0.25)),
            static_gamma=float(config.get("static_gamma", 0.5)),
            static_delta=int(config.get("static_delta", 5)),
            static_weight=float(config.get("static_weight", 0.7)),
            dynamic_temperature=float(config.get("dynamic_temperature", 0.2)),
            dropout=float(config.get("dropout", 0.1)),
        )

    @staticmethod
    def _stage2_raw_echoes(
        raw_echoes: torch.Tensor,
        expected_pulses: int,
    ) -> torch.Tensor:
        # TFE1 的 stride=2 输出对应原慢时间轴上的偶数位置；二级证据因此
        # 使用抽取后的复数回波，相邻样本表达原始脉冲的 lag-2 演化。
        stage2_echoes = raw_echoes[:, ::2, :]
        if stage2_echoes.size(1) != expected_pulses:
            raise ValueError(
                "TFE2 雷达证据与隐藏特征的时间长度不一致："
                f"evidence={stage2_echoes.size(1)}, features={expected_pulses}。"
            )
        return stage2_echoes

    def forward(self, echoes: torch.Tensor, return_features: bool = False):
        if not torch.is_complex(echoes):
            raise TypeError("ModularSTGNN 需要复数输入 [B, P, N]。")
        if echoes.dim() != 3:
            raise ValueError(
                f"ModularSTGNN 期望输入 [B, P, N]，实际为 {tuple(echoes.shape)}。"
            )
        if echoes.size(1) != self.pulses or echoes.size(2) != self.range_cells:
            raise ValueError(
                f"输入形状 [B, {echoes.size(1)}, {echoes.size(2)}] 与配置 "
                f"pulses/range_cells [{self.pulses}, {self.range_cells}] 不一致。"
            )

        encoded = self.radar_features(echoes)
        spatial1 = F.relu(self.spatial_graph1(encoded))
        temporal1 = self.tfe1(spatial1, raw_echoes=echoes)
        spatial2 = F.relu(self.spatial_graph2(temporal1))
        stage2_echoes = None
        if bool(getattr(self.tfe2.impl, "requires_raw_echoes", False)):
            stage2_echoes = self._stage2_raw_echoes(echoes, spatial2.size(2))
        temporal2 = self.tfe2(spatial2, raw_echoes=stage2_echoes)
        temporal_features = temporal2.mean(dim=2)
        logits = self.detection_head(temporal_features)

        if return_features:
            return logits, {
                "radar_features": encoded,
                "spatial1": spatial1,
                "temporal1": temporal1,
                "spatial2": spatial2,
                "temporal2": temporal2,
            }
        return logits

    def get_temporal_diagnostics(self) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for stage, module in (("tfe1", self.tfe1), ("tfe2", self.tfe2)):
            for name, value in module.get_diagnostics().items():
                diagnostics[f"{stage}_{name}"] = float(value)
        return diagnostics

    def get_temporal_training_diagnostics(self) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for stage, module in (("tfe1", self.tfe1), ("tfe2", self.tfe2)):
            for name, value in module.get_training_diagnostics().items():
                diagnostics[f"{stage}_{name}"] = float(value)
        return diagnostics
