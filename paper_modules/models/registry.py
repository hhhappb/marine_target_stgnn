from __future__ import annotations

import torch.nn as nn

from .modular_stgnn import ModularSTGNN
from .original_stgnn import OriginalSTGNN
from .ipix_pax_l1_a1_tfe1 import IpixPaxL1A1TFE1


def build_model(config: dict[str, object]) -> nn.Module:
    model_cfg = config.get("model", {})
    if "name" not in model_cfg:
        raise ValueError("paper_modules 模型配置必须显式声明 model.name。")
    name = str(model_cfg["name"])
    if name == "original_stgnn":
        return OriginalSTGNN(config)
    if name == "modular_stgnn":
        return ModularSTGNN(config)
    if name == "ipix_pax_l1_a1_tfe1":
        return IpixPaxL1A1TFE1(config)
    raise ValueError(f"Unknown paper model: {name}")
