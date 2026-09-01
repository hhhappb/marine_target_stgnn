from __future__ import annotations

import torch.nn as nn

from .modular_stgnn import ModularSTGNN
from .original_stgnn import OriginalSTGNN


def build_model(config: dict[str, object]) -> nn.Module:
    model_cfg = config.get("model", {})
    if "name" not in model_cfg:
        raise ValueError("paper_modules 模型配置必须显式声明 model.name。")
    name = str(model_cfg["name"])
    if name == "original_stgnn":
        return OriginalSTGNN(config)
    if name == "modular_stgnn":
        return ModularSTGNN(config)
    raise ValueError(f"Unknown paper model: {name}")
