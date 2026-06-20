import torch.nn as nn

from .temporal_modules import build_temporal_module


class TemporalModule(nn.Module):
    """时间建模模块包装器：按配置选择一个具体时间处理思路。"""

    def __init__(self, in_channels: int, hidden_channels: int = 256, out_channels: int = 1024, temporal_type: str = "convgru_baseline"):
        super().__init__()
        self.impl = build_temporal_module(temporal_type, in_channels, hidden_channels, out_channels)

    def forward(self, x):
        return self.impl(x)
