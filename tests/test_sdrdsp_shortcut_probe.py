from __future__ import annotations

import numpy as np
import pytest

from paper_modules.experiments.leakage_probe import (
    audit_target_record,
    build_factor_effects,
    canonical_array_sha256,
    classify_pd_drop,
    fig9_error_metrics,
    paper_order_stat_threshold,
    strict_rank_threshold,
    summarize_gain_distribution,
    transform_target_cells,
    validate_pfa_values,
)


def test_component_hash_is_canonical_across_array_layouts() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4) + 1j

    c_hash = canonical_array_sha256(np.ascontiguousarray(values), "<c8")
    f_hash = canonical_array_sha256(np.asfortranarray(values), "<c8")

    assert c_hash == f_hash


def test_scr_audit_recovers_pure_target_power_and_window_scr() -> None:
    windows, pulses, cells = 3, 4, 21
    raw_complex = np.zeros((windows, pulses, cells), dtype=np.complex128)
    for window in range(windows):
        raw_complex[window, :, :20] = np.sqrt(window + 1.0)
        raw_complex[window, :, 20] = 3.0 + 2.0j
    gains = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    base_power = 40.0
    injected = raw_complex.copy()
    injected[:, :, 20] += np.sqrt(base_power * gains)[:, None]

    def pack(values: np.ndarray) -> np.ndarray:
        return np.stack([values.real, values.imag], axis=1).astype(np.float32)

    record = {
        "target_position_local_zero_based": 20,
        "reference_cells_local_zero_based": list(range(20)),
        "speed_mps": 0.4,
        "target_amplitude": np.sqrt(base_power),
        "reference_power_sum": base_power,
    }
    summary, vectors = audit_target_record(pack(injected), pack(raw_complex), record, 0, gains)

    np.testing.assert_allclose(vectors["recovered_gain"], gains, rtol=1e-6, atol=1e-6)
    expected_reference = 20.0 * np.arange(1.0, 4.0)
    expected_scr = 10.0 * np.log10(base_power * gains / expected_reference)
    np.testing.assert_allclose(vectors["actual_scr_db"], expected_scr, rtol=1e-6, atol=1e-6)
    assert summary["max_abs_reference_cell_residual"] == 0.0
    assert summary["max_abs_scr_identity_residual_db"] < 1e-5
    assert summary["gain_replay_allclose"] is True


def test_gain_distribution_reports_deep_fade_fractions() -> None:
    gains = np.array([0.005, 0.05, 0.5, 4.0], dtype=np.float64)

    summary = summarize_gain_distribution(gains)

    assert summary["count"] == 4
    assert summary["fraction_lt_0_1"] == pytest.approx(0.5)
    assert summary["fraction_lt_0_01"] == pytest.approx(0.25)
    assert summary["fraction_gt_3"] == pytest.approx(0.25)
    assert summary["unique_count"] == 4
    assert summary["nonpositive_count"] == 0


def test_factor_effects_use_paired_windows_and_are_deterministic() -> None:
    vectors = {
        "I": np.array([1, 1, 0, 1], dtype=np.float64),
        "P": np.array([1, 0, 0, 1], dtype=np.float64),
        "A": np.array([1, 1, 0, 0], dtype=np.float64),
        "R": np.array([0, 0, 0, 0], dtype=np.float64),
    }

    first = build_factor_effects(vectors, bootstrap_resamples=200, seed=9)
    second = build_factor_effects(vectors, bootstrap_resamples=200, seed=9)

    assert first == second
    assert first["Y00_ideal"] == pytest.approx(0.75)
    assert first["Y10_phase_bundle"] == pytest.approx(0.5)
    assert first["Y01_rcs"] == pytest.approx(0.5)
    assert first["Y11_combined"] == pytest.approx(0.0)
    assert first["effects"]["D_phase"]["estimate"] == pytest.approx(0.25)
    assert first["effects"]["D_rcs"]["estimate"] == pytest.approx(0.25)
    assert first["effects"]["D_combined"]["estimate"] == pytest.approx(0.75)
    assert first["effects"]["D_interaction"]["estimate"] == pytest.approx(0.25)


def test_classify_pd_drop_uses_predeclared_boundaries() -> None:
    assert classify_pd_drop(0.019) == "negligible_or_improved"
    assert classify_pd_drop(0.02) == "mild"
    assert classify_pd_drop(0.05) == "clear"
    assert classify_pd_drop(0.10) == "clear"
    assert classify_pd_drop(0.101) == "strong"


def test_strict_rank_threshold_never_exceeds_budget_with_boundary_ties() -> None:
    scores = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    result = strict_rank_threshold(scores, target_pfa=0.25)

    assert result["false_alarm_budget"] == 2
    assert result["selected_clutter_bins"] == 0
    assert result["calibration_actual_pf"] <= 0.25
    assert result["boundary_tie_count"] == 3
    assert result["boundary_tie_excluded"] is True


def test_strict_rank_threshold_selects_exact_unique_rank_budget() -> None:
    scores = np.arange(10, dtype=np.float64)

    result = strict_rank_threshold(scores, target_pfa=0.3)

    assert result["false_alarm_budget"] == 3
    assert result["selected_clutter_bins"] == 3
    assert result["calibration_actual_pf"] == pytest.approx(0.3)
    assert result["threshold"] == 2.0
    assert result["boundary_tie_excluded"] is False


def test_paper_order_stat_includes_boundary_ties_and_marks_degeneracy() -> None:
    scores = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    result = paper_order_stat_threshold(scores, target_pfa=0.25)

    assert result["rank_one_based"] == 2
    assert result["threshold"] == 0.0
    assert result["selected_clutter_bins"] == 3
    assert result["boundary_tie_count"] == 3
    assert result["boundary_tie_excluded"] is False
    assert result["numerically_degenerate"] is False


def test_fig9_error_metrics_use_non_overlapping_low_and_transition_regions() -> None:
    scr = list(range(-24, 15, 2))
    reference = [
        {"scr_db": float(value), "pd": 0.5, "digitization_uncertainty": 0.01}
        for value in scr
    ]
    per_scr = [
        {"scr_db": value, "PD": 0.6 if value <= -20 else 0.7 if -18 <= value <= -8 else 0.5}
        for value in scr
    ]

    metrics = fig9_error_metrics(per_scr, reference)

    assert metrics["RMSE_low_scr"] == pytest.approx(0.1)
    assert metrics["RMSE_transition"] == pytest.approx(0.2)
    assert metrics["RMSE_all"] > 0.0


def test_validate_pfa_values_rejects_duplicates_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="不允许重复"):
        validate_pfa_values([0.001, 0.001])
    with pytest.raises(ValueError, match="位于"):
        validate_pfa_values([0.0])


def _sample_target_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((2, 2, 4, 5), dtype=np.float32)
    for window in range(2):
        for pulse in range(4):
            for cell in range(5):
                x[window, 0, pulse, cell] = 10 * window + pulse + cell / 10
                x[window, 1, pulse, cell] = -5 * window - pulse - cell / 20
    labels = np.zeros((2, 5), dtype=np.int64)
    labels[0, 1] = 1
    labels[1, 3] = 1
    return x, labels


def test_target_pulse_shuffle_changes_only_target_cells_and_preserves_samples() -> None:
    x, labels = _sample_target_data()

    transformed = transform_target_cells(
        x,
        labels,
        mode="target_pulse_shuffle",
        rng=np.random.default_rng(7),
    )

    non_target_mask = np.broadcast_to((labels == 0)[:, None, None, :], x.shape)
    np.testing.assert_array_equal(transformed[non_target_mask], x[non_target_mask])
    for window, target in ((0, 1), (1, 3)):
        original = x[window, 0, :, target] + 1j * x[window, 1, :, target]
        changed = transformed[window, 0, :, target] + 1j * transformed[window, 1, :, target]
        np.testing.assert_allclose(np.sort_complex(changed), np.sort_complex(original))
        assert not np.array_equal(changed, original)


def test_target_phase_random_changes_only_target_phase_and_preserves_magnitude() -> None:
    x, labels = _sample_target_data()

    transformed = transform_target_cells(
        x,
        labels,
        mode="target_phase_random",
        rng=np.random.default_rng(11),
    )

    non_target_mask = np.broadcast_to((labels == 0)[:, None, None, :], x.shape)
    np.testing.assert_array_equal(transformed[non_target_mask], x[non_target_mask])
    for window, target in ((0, 1), (1, 3)):
        original = x[window, 0, :, target] + 1j * x[window, 1, :, target]
        changed = transformed[window, 0, :, target] + 1j * transformed[window, 1, :, target]
        np.testing.assert_allclose(np.abs(changed), np.abs(original), rtol=1e-6, atol=1e-6)
        assert not np.allclose(changed, original)
