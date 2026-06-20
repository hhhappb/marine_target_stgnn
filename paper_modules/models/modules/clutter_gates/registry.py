from __future__ import annotations

import torch.nn as nn

from .identity import IdentityClutterGate
from .local_statistics import LocalStatisticsClutterGate


def build_clutter_gate(config: dict[str, object]) -> nn.Module:
    if not bool(config.get("enabled", False)):
        return IdentityClutterGate()
    gate_type = str(config.get("type", "local_statistics"))
    if gate_type == "local_statistics":
        return LocalStatisticsClutterGate(
            strength=float(config.get("strength", 0.25)),
            window=int(config.get("window", 5)),
        )
    raise ValueError(f"Unknown clutter gate type: {gate_type}")
