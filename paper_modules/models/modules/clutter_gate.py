import torch.nn as nn

from .clutter_gates import build_clutter_gate


class ClutterAwareGate(nn.Module):
    """海杂波门控包装器：按配置选择一个具体杂波抑制思路。"""

    def __init__(self, enabled: bool = False, gate_type: str = "local_statistics", strength: float = 0.25, window: int = 5):
        super().__init__()
        self.impl = build_clutter_gate({"enabled": enabled, "type": gate_type, "strength": strength, "window": window})

    def forward(self, features, echoes=None):
        return self.impl(features, echoes)
