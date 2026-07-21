from __future__ import annotations

import torch.nn as nn

from .temporal_modules import build_temporal_module


class TemporalModule(nn.Module):
    """时间模块包装器：按配置选择原 TFE 或 TFE 替换模块。"""

    def __init__(self, config: dict[str, object], in_channels: int, out_channels: int):
        super().__init__()
        self.impl = build_temporal_module(config, in_channels, out_channels)

    def forward(self, x):
        return self.impl(x)

    def get_diagnostics(self) -> dict[str, float]:
        """返回最近一个 batch 的时间模块诊断量。"""
        return dict(getattr(self.impl, "last_diagnostics", {}))
