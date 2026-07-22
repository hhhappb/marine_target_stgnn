from __future__ import annotations

import copy

import pytest
import torch

from paper_modules.models import build_model
from paper_modules.models.modules.temporal_modules import (
    DiffTFE,
    FixedUniformTemporalMixerTFE,
    PulseAttentionOnlyTFE,
    ScaleNormalizedDifferenceDecompositionTFE,
    STGNNTemporalGate,
    build_temporal_module,
)


def _small_stgnn_config(
    pulses: int,
    temporal_type: str,
    use_modulation: bool = True,
) -> dict[str, object]:
    temporal_override = {
        "type": temporal_type,
        "beta_max": 0.1,
        "eps": 1e-6,
        "use_modulation": use_modulation,
    }
    return {
        "model": {
            "name": "sfe_replacement_stgnn",
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
            **temporal_override,
            "stage1_out_channels": 12,
        },
        "temporal2": {
            **temporal_override,
            "stage2_out_channels": 20,
            "out_channels": 20,
        },
        "detection_head": {"hidden_channels": 8},
    }


def test_pulse_attention_is_p_by_p_and_keeps_range_cells_separate() -> None:
    module = PulseAttentionOnlyTFE(8, 12, attention_dim=8, num_heads=4)
    output = module(torch.randn(2, 8, 4, 256))
    assert output.shape == (2, 12, 2, 256)
    assert module.last_attention_shape == (2 * 256, 4, 4, 4)


def test_pulse_attention_counterfactual_modes_keep_cell_diagnostics() -> None:
    module = PulseAttentionOnlyTFE(8, 12, attention_dim=8, num_heads=4)
    x = torch.randn(2, 8, 4, 16)
    for mode in ("learned", "uniform", "identity", "residual_off"):
        module.set_counterfactual_mode(mode)
        assert module(x).shape == (2, 12, 2, 16)
        assert module.last_cell_diagnostics["attention_entropy"].shape == (2, 16)
    with pytest.raises(ValueError, match="未知 pulse attention"):
        module.set_counterfactual_mode("not_a_mode")


def test_pulse_attention_diagnostic_logit_multiplier_changes_weights_without_retraining() -> None:
    torch.manual_seed(42)
    module = PulseAttentionOnlyTFE(8, 12, attention_dim=8, num_heads=4)
    x = torch.randn(2, 8, 4, 16)
    module(x)
    base_weights = module.last_attention_weights.detach().clone()
    module.set_diagnostic_logit_multiplier(10.0)
    module(x)
    scaled_weights = module.last_attention_weights.detach().clone()
    assert not torch.equal(base_weights, scaled_weights)
    with pytest.raises(ValueError, match="有限正数"):
        module.set_diagnostic_logit_multiplier(0.0)


def test_fixed_uniform_mixer_has_uniform_cell_diagnostics_and_shape() -> None:
    module = FixedUniformTemporalMixerTFE(8, 12, mixer_dim=8, residual_scale=0.1)
    output = module(torch.randn(2, 8, 4, 16))
    assert output.shape == (2, 12, 2, 16)
    assert module.last_mixer_shape == (2 * 16, 1, 4, 4)
    assert torch.allclose(module.last_cell_diagnostics["attention_entropy"], torch.ones(2, 16))
    assert torch.allclose(
        module.last_cell_diagnostics["attention_max_weight"],
        torch.full((2, 16), 0.25),
    )


def test_corrected_diff_zero_enhancement_matches_original_gate() -> None:
    torch.manual_seed(42)
    original = STGNNTemporalGate(8, 12)
    corrected = DiffTFE(8, 12, use_diff=False)
    corrected.update.load_state_dict(copy.deepcopy(original.update.state_dict()))
    corrected.output.load_state_dict(copy.deepcopy(original.output.state_dict()))
    x = torch.randn(2, 8, 4, 256)
    assert torch.equal(original(x), corrected(x))


@pytest.mark.parametrize("pulses", [2, 4, 8, 16, 32])
def test_scale_normalized_difference_shape_and_finite_values(pulses: int) -> None:
    module = ScaleNormalizedDifferenceDecompositionTFE(8, 12)
    output = module(torch.randn(2, 8, pulses, 14))
    assert output.shape == (2, 12, (pulses + 1) // 2, 14)
    assert torch.isfinite(output).all()


def test_scale_normalized_difference_zero_projection_matches_original_gate() -> None:
    torch.manual_seed(42)
    original = STGNNTemporalGate(8, 12)
    torch.manual_seed(42)
    candidate = ScaleNormalizedDifferenceDecompositionTFE(8, 12)
    x = torch.randn(2, 8, 8, 14)
    assert torch.equal(original(x), candidate(x))
    assert torch.count_nonzero(candidate.evidence_proj.weight) == 0
    assert torch.count_nonzero(candidate.evidence_proj.bias) == 0


def test_scale_normalized_difference_identity_control_keeps_parameters_and_common_initialization() -> None:
    torch.manual_seed(42)
    active = ScaleNormalizedDifferenceDecompositionTFE(8, 12, use_modulation=True)
    torch.manual_seed(42)
    identity = ScaleNormalizedDifferenceDecompositionTFE(8, 12, use_modulation=False)
    for name, value in active.state_dict().items():
        assert torch.equal(value, identity.state_dict()[name])

    with torch.no_grad():
        active.evidence_proj.weight.fill_(0.25)
    x = torch.randn(2, 8, 8, 14)
    identity_output = identity(x)
    original = STGNNTemporalGate(8, 12)
    original.update.load_state_dict(copy.deepcopy(identity.update.state_dict()))
    original.output.load_state_dict(copy.deepcopy(identity.output.state_dict()))
    assert torch.equal(identity_output, original(x))
    assert not torch.equal(active(x), identity_output)


def test_scale_normalized_difference_bounds_gradients_and_optional_diagnostics() -> None:
    module = ScaleNormalizedDifferenceDecompositionTFE(8, 12, beta_max=0.1)
    with torch.no_grad():
        module.evidence_proj.weight.normal_(mean=0.0, std=0.2)
    x = torch.randn(2, 8, 8, 14, requires_grad=True)
    trend, _, normalized_curvature = module._build_evidence(x)
    modulation = module.beta_max * torch.tanh(
        module.evidence_proj(torch.cat([trend, normalized_curvature], dim=1))
    )
    assert modulation.abs().max() <= module.beta_max
    module(x).sum().backward()
    assert module.evidence_proj.weight.grad is not None
    assert torch.count_nonzero(module.evidence_proj.weight.grad) > 0
    assert module.last_diagnostics == {}
    assert module.last_cell_diagnostics == {}

    diagnostic_module = ScaleNormalizedDifferenceDecompositionTFE(
        8,
        12,
        collect_diagnostics=True,
    )
    diagnostic_module(torch.randn(2, 8, 8, 14))
    assert "difference_trend_rms" in diagnostic_module.last_diagnostics
    assert diagnostic_module.last_cell_diagnostics["trend_rms"].shape == (2, 14)


@pytest.mark.parametrize(
    ("pulses", "tfe1_length", "tfe2_length"),
    [(4, 2, 1), (8, 4, 2), (16, 8, 4), (32, 16, 8)],
)
def test_both_stage_scale_normalized_difference_model_interface(
    pulses: int,
    tfe1_length: int,
    tfe2_length: int,
) -> None:
    model = build_model(
        _small_stgnn_config(pulses, "scale_normalized_difference_decomposition_tfe")
    )
    echoes = torch.complex(torch.randn(2, pulses, 14), torch.randn(2, pulses, 14))
    logits, features = model(echoes, return_features=True)
    assert logits.shape == (2, 2, 14)
    assert features["temporal1"].shape[2] == tfe1_length
    assert features["temporal2"].shape[2] == tfe2_length
    assert isinstance(model.tfe1.impl, ScaleNormalizedDifferenceDecompositionTFE)
    assert isinstance(model.tfe2.impl, ScaleNormalizedDifferenceDecompositionTFE)
    assert model.tfe1.impl.evidence_proj.weight is not model.tfe2.impl.evidence_proj.weight
    assert torch.isfinite(logits).all()


def test_both_stage_candidate_preserves_original_common_initialization_and_has_gradients() -> None:
    torch.manual_seed(42)
    original = build_model(_small_stgnn_config(8, "stgnn_tfe"))
    torch.manual_seed(42)
    candidate = build_model(
        _small_stgnn_config(8, "scale_normalized_difference_decomposition_tfe")
    )
    original_state = original.state_dict()
    candidate_state = candidate.state_dict()
    for name, value in original_state.items():
        assert torch.equal(value, candidate_state[name])

    echoes = torch.complex(torch.randn(2, 8, 14), torch.randn(2, 8, 14))
    original.eval()
    candidate.eval()
    original_output = original(echoes)
    candidate_output = candidate(echoes)
    assert torch.equal(original_output, candidate_output)
    candidate_output.sum().backward()
    for module in (candidate.tfe1.impl, candidate.tfe2.impl):
        gradient = module.evidence_proj.weight.grad
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0


def test_public_model_types_and_fail_loud_checks() -> None:
    assert isinstance(build_temporal_module({"type": "diff_tfe"}, 8, 12), DiffTFE)
    assert isinstance(
        build_temporal_module({"type": "pulse_attention_only_tfe", "attention_dim": 8}, 8, 12),
        PulseAttentionOnlyTFE,
    )
    assert isinstance(build_temporal_module({"type": "fixed_uniform_tfe"}, 8, 12), FixedUniformTemporalMixerTFE)
    assert isinstance(
        build_temporal_module({"type": "scale_normalized_difference_decomposition_tfe"}, 8, 12),
        ScaleNormalizedDifferenceDecompositionTFE,
    )
    with pytest.raises(ValueError, match="Unknown temporal module type"):
        build_temporal_module({"type": "not_a_temporal_module"}, 8, 12)
    with pytest.raises(ValueError, match="Unknown temporal module type"):
        build_temporal_module({"type": "diff_bicam_tfe"}, 8, 12)
    with pytest.raises(ValueError, match="P>=2"):
        DiffTFE(8, 12)(torch.randn(1, 8, 1, 256))
    with pytest.raises(ValueError, match="P>=2"):
        PulseAttentionOnlyTFE(8, 12, attention_dim=8)(torch.randn(1, 8, 1, 256))
    with pytest.raises(ValueError, match="P>=2"):
        ScaleNormalizedDifferenceDecompositionTFE(8, 12)(torch.randn(1, 8, 1, 256))
