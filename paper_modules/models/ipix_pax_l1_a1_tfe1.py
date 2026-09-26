from __future__ import annotations

from collections.abc import Mapping

from paper_modules.models.modules.temporal_modules.pulse_attention_tfe import (
    PulseAttentionOnlyTFE,
)
from paper_modules.models.original_stgnn import OriginalSTGNN


IPIX_PAX_L1_A1_TFE1_CONFIG = {
    "type": "pulse_attention_only_tfe",
    "attention_dim": 64,
    "num_heads": 4,
    "residual_scale": 0.1,
    "use_attention": True,
}


class IpixPaxL1A1TFE1(OriginalSTGNN):
    """PAX-L1 Original ST-GNN with an uncapped attention residual at TFE1 only."""

    def __init__(self, config: dict[str, object]):
        super().__init__(config)
        if self.pulses != 16 or self.range_cells != 14:
            raise ValueError(
                "IpixPaxL1A1TFE1 is fixed to P=16, N=14; "
                f"received P={self.pulses}, N={self.range_cells}."
            )

        model_cfg = config.get("model", {})
        tfe1_cfg = model_cfg.get("tfe1") if isinstance(model_cfg, Mapping) else None
        if not isinstance(tfe1_cfg, Mapping):
            raise ValueError("ipix_pax_l1_a1_tfe1 requires an explicit model.tfe1 mapping.")
        if "residual_ratio_cap" in tfe1_cfg or "residual_bound" in tfe1_cfg:
            raise ValueError("A1-TFE1 has no residual cap or bound parameter.")

        for key, expected in IPIX_PAX_L1_A1_TFE1_CONFIG.items():
            if key not in tfe1_cfg or tfe1_cfg[key] != expected:
                raise ValueError(
                    f"A1-TFE1 requires model.tfe1.{key}={expected!r}; "
                    f"received {tfe1_cfg.get(key)!r}."
                )

        self.backbone.tfe1 = PulseAttentionOnlyTFE(
            in_channels=128,
            out_channels=256,
            attention_dim=int(tfe1_cfg["attention_dim"]),
            num_heads=int(tfe1_cfg["num_heads"]),
            residual_scale=float(tfe1_cfg["residual_scale"]),
            use_attention=bool(tfe1_cfg["use_attention"]),
        )
