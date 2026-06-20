from __future__ import annotations

import torch
import torch.nn as nn

from .radar_features import build_raw_feature_extractor


class RadarFeatureEncoder(nn.Module):
    """雷达输入特征编码器。

    作用：
        1. 根据配置选择一个具体的雷达先验输入思路；
        2. 将该思路生成的原始特征通道映射到统一 hidden feature。

    输入：复数回波 [B, P, N]
    输出：特征图 [B, out_channels, P, N]
    """

    def __init__(self, feature_type: str = "real_imag", hidden_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.feature_type = feature_type
        self.raw_features = build_raw_feature_extractor(feature_type)
        in_channels = int(self.raw_features.out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
        )

    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        raw = self.raw_features(echoes)
        return self.net(raw)
