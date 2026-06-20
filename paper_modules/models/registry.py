from __future__ import annotations

import torch.nn as nn

from .experimental_stgnn import ExperimentalSTGNN


def build_model(config: dict[str, object]) -> nn.Module:
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "experimental_stgnn"))
    if name == "experimental_stgnn":
        return ExperimentalSTGNN(config)
    raise ValueError(f"Unknown paper model: {name}")
