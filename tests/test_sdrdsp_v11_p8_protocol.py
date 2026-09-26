from __future__ import annotations

import pytest
import torch

from paper_modules.datasets import build_dataset
from paper_modules.datasets.scr_npz import (
    SDRDSP_PROTOCOL_RANGE_CELLS,
    SDRDSP_V11_P8_PROTOCOL,
    SDRDSP_V11_PROTOCOL,
    SDRDSP_V11_PROTOCOL_SPECS,
)
from paper_modules.models import build_model
from paper_modules.models.modules.temporal_modules import (
    PulseAttentionOnlyTFE,
    build_temporal_module,
)


def test_p8_window_rule_counts_match_frozen_contract() -> None:
    # 生成规则 range(0, length-P, P) 对 6940/6520 脉冲的算术结果
    train_starts = len(range(0, 6940 - 8, 8))
    test_starts = len(range(0, 6520 - 8, 8))
    assert train_starts == 867
    assert test_starts == 814
    spec = SDRDSP_V11_PROTOCOL_SPECS[SDRDSP_V11_P8_PROTOCOL]
    assert spec["pulses"] == 8
    assert spec["train_windows_per_scr"] == train_starts
    assert spec["test_windows_per_scr"] == test_starts
    assert spec["train_shape"] == [867 * 14, 2, 8, 256]
    assert SDRDSP_PROTOCOL_RANGE_CELLS[SDRDSP_V11_P8_PROTOCOL] == 256


def test_p4_contract_unchanged() -> None:
    spec = SDRDSP_V11_PROTOCOL_SPECS[SDRDSP_V11_PROTOCOL]
    assert spec["pulses"] == 4
    assert spec["train_windows_per_scr"] == 1734
    assert spec["test_windows_per_scr"] == 1629
    assert spec["train_shape"] == [24276, 2, 4, 256]
    assert SDRDSP_PROTOCOL_RANGE_CELLS[SDRDSP_V11_PROTOCOL] == 256


@pytest.mark.parametrize("protocol", [SDRDSP_V11_PROTOCOL, SDRDSP_V11_P8_PROTOCOL])
def test_v11_protocols_require_p99_normalization(protocol: str) -> None:
    config = {
        "model": {"name": "modular_stgnn", "pulses": 8, "range_cells": 256},
        "dataset": {
            "type": "scr_npz",
            "data_dir": "data/sdrdsp_v11_p8_n256_seed42",
            "protocol": protocol,
            "normalization": "none",
        },
    }
    with pytest.raises(ValueError, match="precomputed_train_p99_iq"):
        build_dataset(config, "train")


@pytest.mark.parametrize(
    "slots",
    [
        ("stgnn_tfe", "stgnn_tfe"),
        ("pulse_attention_only_tfe", "stgnn_tfe"),
        ("pulse_attention_only_tfe", "pulse_attention_only_tfe"),
    ],
)
def test_p8_backbone_wiring_time_dims(slots: tuple[str, str]) -> None:
    config = {
        "model": {"name": "modular_stgnn", "pulses": 8, "range_cells": 14},
        "radar_features": {"type": "real_imag", "hidden_channels": 4, "out_channels": 8},
        "spatial_graph": {
            "type": "original_stfe",
            "stage1_out_channels": 8,
            "stage2_out_channels": 16,
        },
        "temporal": {
            "type": "stgnn_tfe",
            "stage1_out_channels": 12,
            "stage2_out_channels": 20,
            "out_channels": 20,
        },
        "temporal1": {
            "type": slots[0],
            "attention_dim": 64,
            "num_heads": 4,
            "residual_scale": 0.1,
            "use_attention": True,
        },
        "temporal2": {
            "type": slots[1],
            "attention_dim": 64,
            "num_heads": 4,
            "residual_scale": 0.1,
            "use_attention": True,
        },
        "detection_head": {"hidden_channels": 8},
    }
    torch.manual_seed(42)
    model = build_model(config)
    echoes = torch.complex(torch.randn(2, 8, 14), torch.randn(2, 8, 14))
    logits, feats = model(echoes, return_features=True)
    assert logits.shape == (2, 2, 14)
    # 时间维 8 -> 4 -> 2
    assert feats["temporal1"].shape[2] == 4
    assert feats["temporal2"].shape[2] == 2
    assert feats["temporal1"].shape[1] == 12
    assert feats["temporal2"].shape[1] == 20


def test_pulse_attention_p8_defaults_and_shape() -> None:
    module = build_temporal_module(
        {
            "type": "pulse_attention_only_tfe",
            "attention_dim": 64,
            "num_heads": 4,
            "residual_scale": 0.1,
            "use_attention": True,
        },
        8,
        16,
    )
    assert isinstance(module, PulseAttentionOnlyTFE)
    assert module.counterfactual_mode == "learned"
    assert module.diagnostic_logit_multiplier == 1.0
    x = torch.randn(2, 8, 8, 16)
    assert module(x).shape == (2, 16, 4, 16)
