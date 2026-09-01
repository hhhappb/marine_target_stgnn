from __future__ import annotations

import torch

from paper_modules.models import build_model
from paper_modules.models.modules.spatial_graphs.local_range import LocalRangeGraph
from paper_modules.models.modules.spatial_graphs.original_stfe import OriginalSTFEGraph
from paper_modules.models.modules.temporal_modules import ComplexEvidenceResidualTFE


def _original_module_config(pulses: int, model_name: str) -> dict[str, object]:
    config: dict[str, object] = {
        "model": {
            "name": model_name,
            "pulses": pulses,
            "range_cells": 14,
        }
    }
    if model_name == "original_stgnn":
        return config
    config.update(
        {
            "radar_features": {
                "type": "real_imag",
                "hidden_channels": 32,
                "out_channels": 64,
            },
            "spatial_graph": {
                "type": "original_stfe",
                "stage1_out_channels": 128,
                "stage2_out_channels": 512,
                "dropout": 0.1,
            },
            "temporal": {
                "type": "stgnn_tfe",
                "stage1_out_channels": 256,
                "stage2_out_channels": 1024,
                "out_channels": 1024,
            },
            "detection_head": {"hidden_channels": 512},
        }
    )
    return config


def _small_modular_config(pulses: int = 16) -> dict[str, object]:
    return {
        "model": {"name": "modular_stgnn", "pulses": pulses, "range_cells": 14},
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
        "detection_head": {"hidden_channels": 8},
    }


def _echoes(batch: int, pulses: int) -> torch.Tensor:
    return torch.complex(
        torch.randn(batch, pulses, 14),
        torch.randn(batch, pulses, 14),
    )


def test_all_original_slots_match_original_stgnn_in_eval_and_train() -> None:
    torch.manual_seed(42)
    original = build_model(_original_module_config(4, "original_stgnn"))
    torch.manual_seed(42)
    modular = build_model(_original_module_config(4, "modular_stgnn"))

    original_parameters = list(original.parameters())
    modular_parameters = list(modular.parameters())
    assert len(original_parameters) == len(modular_parameters)
    for original_parameter, modular_parameter in zip(
        original_parameters, modular_parameters, strict=True
    ):
        assert torch.equal(original_parameter, modular_parameter)

    echoes = _echoes(2, 4)
    original.eval()
    modular.eval()
    with torch.no_grad():
        assert torch.equal(original(echoes), modular(echoes))

    original.train()
    modular.train()
    torch.manual_seed(123)
    original_logits = original(echoes)
    torch.manual_seed(123)
    modular_logits = modular(echoes)
    assert torch.equal(original_logits, modular_logits)


def test_complex_evidence_replaces_both_tfe_stages_and_receives_gradients() -> None:
    config = _small_modular_config(16)
    config["temporal1"] = {"type": "complex_evidence_residual_tfe"}
    config["temporal2"] = {"type": "complex_evidence_residual_tfe"}
    model = build_model(config)
    assert isinstance(model.tfe1.impl, ComplexEvidenceResidualTFE)
    assert isinstance(model.tfe2.impl, ComplexEvidenceResidualTFE)

    captured: dict[str, torch.Tensor] = {}

    def capture_stage2_echoes(module, args, kwargs):
        captured["raw_echoes"] = kwargs["raw_echoes"].detach().clone()

    handle = model.tfe2.impl.register_forward_pre_hook(
        capture_stage2_echoes,
        with_kwargs=True,
    )
    echoes = _echoes(2, 16)
    logits, features = model(echoes, return_features=True)
    handle.remove()

    assert logits.shape == (2, 2, 14)
    assert features["temporal1"].shape == (2, 12, 8, 14)
    assert features["temporal2"].shape == (2, 20, 4, 14)
    assert torch.equal(captured["raw_echoes"], echoes[:, ::2, :])

    logits.sum().backward()
    for temporal_module in (model.tfe1.impl, model.tfe2.impl):
        gradients = [
            temporal_module.evidence_update.weight.grad,
            temporal_module.evidence_output.weight.grad,
        ]
        assert all(gradient is not None for gradient in gradients)
        assert sum(int(torch.count_nonzero(gradient)) for gradient in gradients) > 0


def test_spatial_stages_can_be_configured_independently() -> None:
    config = _small_modular_config(8)
    config["spatial1"] = {"type": "local_3"}
    config["spatial2"] = {"type": "original_stfe"}
    model = build_model(config)

    assert isinstance(model.spatial_graph1.impl, LocalRangeGraph)
    assert isinstance(model.spatial_graph2.impl, OriginalSTFEGraph)
    assert model(_echoes(2, 8)).shape == (2, 2, 14)
