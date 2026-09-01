from __future__ import annotations

import math

import pytest
import torch

from paper_modules.models import build_model
from paper_modules.models.modules.temporal_modules import (
    ComplexEvidenceTFE,
    STGNNTemporalGate,
    build_temporal_module,
)


def _small_config(pulses: int, temporal1_type: str) -> dict[str, object]:
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
            "type": temporal1_type,
            "stage1_out_channels": 12,
            "beta_max": 0.1,
            "eps": 1e-6,
            "use_modulation": True,
        },
        "temporal2": {
            "type": "stgnn_tfe",
            "stage2_out_channels": 20,
            "out_channels": 20,
        },
        "detection_head": {"hidden_channels": 8},
    }


def _complex_echoes(batch: int, pulses: int, ranges: int) -> torch.Tensor:
    magnitude = 0.5 + torch.rand(batch, pulses, ranges)
    phase = 2.0 * math.pi * torch.rand(batch, pulses, ranges)
    return torch.polar(magnitude, phase)


def test_complex_evidence_is_global_phase_and_scale_invariant() -> None:
    module = ComplexEvidenceTFE(8, 12)
    echoes = _complex_echoes(2, 8, 14)
    reference = module._build_evidence(echoes)
    transformed = 3.5 * torch.exp(torch.tensor(1.2j)) * echoes
    actual = module._build_evidence(transformed)
    assert torch.allclose(actual, reference, atol=2e-6, rtol=2e-6)


def test_complex_evidence_keeps_range_cells_independent() -> None:
    module = ComplexEvidenceTFE(8, 12)
    echoes = _complex_echoes(2, 8, 14)
    changed = echoes.clone()
    changed[:, :, 5] *= torch.exp(torch.tensor(0.7j))
    reference = module._build_evidence(echoes)
    actual = module._build_evidence(changed)
    keep = torch.ones(14, dtype=torch.bool)
    keep[5] = False
    assert torch.equal(actual[:, :, :, keep], reference[:, :, :, keep])


def test_zero_projection_matches_original_gate() -> None:
    torch.manual_seed(42)
    original = STGNNTemporalGate(8, 12)
    torch.manual_seed(42)
    candidate = ComplexEvidenceTFE(8, 12)
    x = torch.randn(2, 8, 8, 14)
    echoes = _complex_echoes(2, 8, 14)
    assert torch.equal(original(x), candidate(x, raw_echoes=echoes))
    assert torch.count_nonzero(candidate.evidence_proj.weight) == 0


def test_complex_evidence_modulation_is_bounded_and_has_gradients() -> None:
    module = ComplexEvidenceTFE(8, 12, beta_max=0.1)
    with torch.no_grad():
        module.evidence_proj.weight.normal_(mean=0.0, std=0.2)
    x = torch.randn(2, 8, 8, 14, requires_grad=True)
    echoes = _complex_echoes(2, 8, 14)
    evidence = module._build_evidence(echoes)
    modulation = module.beta_max * torch.tanh(module.evidence_proj(evidence))
    assert modulation.abs().max() <= module.beta_max
    module(x, raw_echoes=echoes).sum().backward()
    assert module.evidence_proj.weight.grad is not None
    assert torch.count_nonzero(module.evidence_proj.weight.grad) > 0


@pytest.mark.parametrize("pulses", [4, 8, 16, 32])
def test_tfe1_only_model_interface_and_common_initialization(pulses: int) -> None:
    torch.manual_seed(42)
    original = build_model(_small_config(pulses, "stgnn_tfe"))
    torch.manual_seed(42)
    candidate = build_model(_small_config(pulses, "complex_evidence_tfe"))
    for name, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])

    echoes = _complex_echoes(2, pulses, 14)
    original.eval()
    candidate.eval()
    candidate_logits, features = candidate(echoes, return_features=True)
    assert torch.equal(original(echoes), candidate_logits)
    assert features["temporal1"].shape[2] == pulses // 2
    assert features["temporal2"].shape[2] == pulses // 4
    assert isinstance(candidate.tfe1.impl, ComplexEvidenceTFE)
    assert isinstance(candidate.tfe2.impl, STGNNTemporalGate)


def test_complex_evidence_registry_and_fail_loud_inputs() -> None:
    assert isinstance(
        build_temporal_module({"type": "complex_evidence_tfe"}, 8, 12),
        ComplexEvidenceTFE,
    )
    module = ComplexEvidenceTFE(8, 12)
    x = torch.randn(2, 8, 4, 14)
    with pytest.raises(ValueError, match="raw_echoes"):
        module(x)
    with pytest.raises(TypeError, match="复数"):
        module(x, raw_echoes=torch.randn(2, 4, 14))
    with pytest.raises(ValueError, match="shape"):
        module(x, raw_echoes=_complex_echoes(2, 8, 14))


def test_evidence_intervention_modes_are_causal_and_reversible() -> None:
    module = ComplexEvidenceTFE(8, 12, evidence_mode="normal")
    with torch.no_grad():
        module.evidence_proj.weight.normal_(mean=0.0, std=0.2)
        module.evidence_proj.bias.fill_(0.05)
    x = torch.randn(4, 8, 4, 14)
    echoes = _complex_echoes(4, 4, 14)

    normal = module(x, raw_echoes=echoes)
    expected_shuffle = torch.roll(torch.flip(echoes, dims=(0,)), shifts=7, dims=2)
    module.evidence_mode = "shuffle"
    assert torch.equal(module._intervene_on_evidence(echoes), expected_shuffle)
    module.evidence_mode = "off"
    off = module(x, raw_echoes=echoes)
    module.evidence_mode = "shuffle"
    shuffled = module(x, raw_echoes=echoes)
    module.evidence_mode = "normal"
    restored = module(x, raw_echoes=echoes)

    assert not torch.equal(normal, off)
    assert not torch.equal(normal, shuffled)
    assert torch.equal(normal, restored)


def test_evidence_mode_registry_and_validation() -> None:
    module = build_temporal_module(
        {"type": "complex_evidence_tfe", "evidence_mode": "shuffle"},
        8,
        12,
    )
    assert isinstance(module, ComplexEvidenceTFE)
    assert module.evidence_mode == "shuffle"
    with pytest.raises(ValueError, match="evidence_mode"):
        ComplexEvidenceTFE(8, 12, evidence_mode="unknown")