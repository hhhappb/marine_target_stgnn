from __future__ import annotations

import math

import pytest
import torch

from paper_modules.models import build_model
from paper_modules.models.modules.temporal_modules import (
    ComplexEvidenceResidualTFE,
    ComplexEvidenceTFE,
    STGNNTemporalGate,
    build_temporal_module,
)


def _config(pulses: int, temporal_type: str) -> dict[str, object]:
    return {
        "model": {
            "name": "modular_stgnn",
            "pulses": pulses,
            "range_cells": 14,
        },
        "radar_features": {
            "type": "real_imag",
            "hidden_channels": 4,
            "out_channels": 8,
        },
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
            "type": temporal_type,
            "stage1_out_channels": 12,
            "eps": 1e-6,
            "evidence_mode": "normal",
        },
        "temporal2": {
            "type": temporal_type,
            "stage2_out_channels": 20,
            "out_channels": 20,
            "eps": 1e-6,
            "evidence_mode": "normal",
        },
        "detection_head": {"hidden_channels": 8},
    }


def _echoes(batch: int, pulses: int, ranges: int) -> torch.Tensor:
    magnitude = 0.5 + torch.rand(batch, pulses, ranges)
    phase = 2.0 * math.pi * torch.rand(batch, pulses, ranges)
    return torch.polar(magnitude, phase)


def test_residual_uses_the_same_radar_evidence_definition() -> None:
    residual = ComplexEvidenceResidualTFE(8, 12)
    modulation = ComplexEvidenceTFE(8, 12)
    echoes = _echoes(3, 8, 14)
    assert torch.equal(
        residual._build_evidence(echoes),
        modulation._build_evidence(echoes),
    )


def test_zero_residual_matches_original_gate_exactly() -> None:
    torch.manual_seed(42)
    original = STGNNTemporalGate(8, 12)
    torch.manual_seed(42)
    candidate = ComplexEvidenceResidualTFE(8, 12)
    x = torch.randn(3, 8, 8, 14)
    echoes = _echoes(3, 8, 14)
    assert torch.equal(original(x), candidate(x, raw_echoes=echoes))
    assert torch.count_nonzero(candidate.evidence_update.weight) == 0
    assert torch.count_nonzero(candidate.evidence_output.weight) == 0


def test_residual_branch_can_create_independent_temporal_signal() -> None:
    module = ComplexEvidenceResidualTFE(8, 12)
    with torch.no_grad():
        module.update.weight.zero_()
        module.update.bias.zero_()
        module.output.weight.zero_()
        module.output.bias.zero_()
        module.evidence_output.weight.fill_(0.2)
    x = torch.zeros(2, 8, 4, 14)
    echoes = _echoes(2, 4, 14)
    output = module(x, raw_echoes=echoes)
    assert torch.count_nonzero(output) > 0


def test_residual_branch_has_gradients_and_reversible_interventions() -> None:
    module = ComplexEvidenceResidualTFE(8, 12, evidence_mode="normal")
    with torch.no_grad():
        module.evidence_update.weight.normal_(mean=0.0, std=0.2)
        module.evidence_output.weight.normal_(mean=0.0, std=0.2)
    x = torch.randn(4, 8, 4, 14, requires_grad=True)
    echoes = _echoes(4, 4, 14)

    normal = module(x, raw_echoes=echoes)
    normal.sum().backward()
    assert module.evidence_update.weight.grad is not None
    assert module.evidence_output.weight.grad is not None
    assert torch.count_nonzero(module.evidence_update.weight.grad) > 0
    assert torch.count_nonzero(module.evidence_output.weight.grad) > 0

    module.evidence_mode = "off"
    off = module(x.detach())
    module.evidence_mode = "shuffle"
    shuffled = module(x.detach(), raw_echoes=echoes)
    module.evidence_mode = "normal"
    restored = module(x.detach(), raw_echoes=echoes)
    assert not torch.equal(normal.detach(), off)
    assert not torch.equal(normal.detach(), shuffled)
    assert torch.equal(normal.detach(), restored)


@pytest.mark.parametrize("pulses", [4, 8, 16, 32])
def test_residual_both_tfe_model_interface_and_common_initialization(
    pulses: int,
) -> None:
    torch.manual_seed(42)
    original = build_model(_config(pulses, "stgnn_tfe"))
    torch.manual_seed(42)
    candidate = build_model(
        _config(pulses, "complex_evidence_residual_tfe")
    )
    for name, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])

    echoes = _echoes(2, pulses, 14)
    original.eval()
    candidate.eval()
    candidate_logits, features = candidate(echoes, return_features=True)
    assert torch.equal(original(echoes), candidate_logits)
    assert features["temporal1"].shape[2] == pulses // 2
    assert features["temporal2"].shape[2] == pulses // 4
    assert isinstance(candidate.tfe1.impl, ComplexEvidenceResidualTFE)
    assert isinstance(candidate.tfe2.impl, ComplexEvidenceResidualTFE)
def test_residual_registry_and_fail_loud_inputs() -> None:
    module = build_temporal_module(
        {"type": "complex_evidence_residual_tfe"},
        8,
        12,
    )
    assert isinstance(module, ComplexEvidenceResidualTFE)
    x = torch.randn(2, 8, 4, 14)
    with pytest.raises(ValueError, match="raw_echoes"):
        module(x)
    with pytest.raises(TypeError, match="复数"):
        module(x, raw_echoes=torch.randn(2, 4, 14))
    with pytest.raises(ValueError, match="shape"):
        module(x, raw_echoes=_echoes(2, 8, 14))
    with pytest.raises(ValueError, match="evidence_mode"):
        ComplexEvidenceResidualTFE(8, 12, evidence_mode="unknown")
