from __future__ import annotations

import copy

import pytest
import torch

from paper_modules.models.modules.temporal_modules import (
    CorrectedDiffOnlyTFE,
    FixedUniformTemporalMixerTFE,
    PulseAttentionOnlyTFE,
    STGNNTemporalGate,
    build_temporal_module,
)


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
    corrected = CorrectedDiffOnlyTFE(8, 12, use_diff=False)
    corrected.update.load_state_dict(copy.deepcopy(original.update.state_dict()))
    corrected.output.load_state_dict(copy.deepcopy(original.output.state_dict()))
    x = torch.randn(2, 8, 4, 256)
    assert torch.equal(original(x), corrected(x))


def test_public_model_types_and_fail_loud_checks() -> None:
    assert isinstance(build_temporal_module({"type": "corrected_diff_only_tfe"}, 8, 12), CorrectedDiffOnlyTFE)
    assert isinstance(
        build_temporal_module({"type": "pulse_attention_only_tfe", "attention_dim": 8}, 8, 12),
        PulseAttentionOnlyTFE,
    )
    assert isinstance(build_temporal_module({"type": "fixed_uniform_tfe"}, 8, 12), FixedUniformTemporalMixerTFE)
    with pytest.raises(ValueError, match="Unknown temporal module type"):
        build_temporal_module({"type": "not_a_temporal_module"}, 8, 12)
    with pytest.raises(ValueError, match="P>=2"):
        CorrectedDiffOnlyTFE(8, 12)(torch.randn(1, 8, 1, 256))
    with pytest.raises(ValueError, match="P>=2"):
        PulseAttentionOnlyTFE(8, 12, attention_dim=8)(torch.randn(1, 8, 1, 256))
