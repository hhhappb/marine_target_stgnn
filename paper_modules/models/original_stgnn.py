from __future__ import annotations

import torch
import torch.nn as nn

from models.st_gnn import STGNNDetector


class OriginalSTGNN(nn.Module):
    """原始 ST-GNN 的薄适配器。

    冻结区 `models/st_gnn.py` 的默认 forward 返回样本级概率；这里固定使用
    `return_features=True` 取逐距离单元 logits，适配统一接口 [B, 2, N]。
    """

    def __init__(self, config: dict[str, object]):
        super().__init__()
        model_cfg = config.get("model", {})
        self.pulses = int(model_cfg.get("pulses", 4))
        self.range_cells = int(model_cfg.get("range_cells", 256))
        self.backbone = STGNNDetector(P=self.pulses, N=self.range_cells)

    def forward(self, echoes: torch.Tensor, return_features: bool = False):
        if not torch.is_complex(echoes):
            raise TypeError("OriginalSTGNN 需要复数输入 [B, P, N]。")
        if echoes.dim() != 3:
            raise ValueError(f"OriginalSTGNN 期望输入 [B, P, N]，实际为 {tuple(echoes.shape)}。")
        if echoes.size(1) != self.pulses or echoes.size(2) != self.range_cells:
            raise ValueError(
                f"输入形状 [B, {echoes.size(1)}, {echoes.size(2)}] 与配置 "
                f"pulses/range_cells [{self.pulses}, {self.range_cells}] 不一致。"
            )

        sample_probs, logits, feature_maps, temporal_features = self.backbone(echoes, return_features=True)
        if return_features:
            return logits, {
                "sample_probs": sample_probs,
                "feature_maps": feature_maps,
                "temporal_features": temporal_features,
            }
        return logits
