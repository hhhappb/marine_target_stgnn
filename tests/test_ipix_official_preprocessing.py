from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from paper_modules.datasets.ipix_window import load_ipix_arrays
from scripts.preprocess_ipix import (
    OFFICIAL_IPIXLOAD_SHA256,
    official_ipixload_auto,
    parse_args,
)


def _official_scalar_reference(i_raw: np.ndarray, q_raw: np.ndarray) -> np.ndarray:
    columns: list[np.ndarray] = []
    for range_index in range(i_raw.shape[1]):
        i_values = i_raw[:, range_index].astype(np.float64)
        q_values = q_raw[:, range_index].astype(np.float64)
        i_values = (i_values - np.mean(i_values)) / np.std(i_values, ddof=0)
        q_values = (q_values - np.mean(q_values)) / np.std(q_values, ddof=0)
        sin_inbal = np.mean(i_values * q_values)
        i_values = (i_values - q_values * sin_inbal) / np.sqrt(1.0 - sin_inbal**2)
        columns.append(i_values + 1j * q_values)
    return np.stack(columns, axis=1)


def test_official_ipixload_auto_matches_per_rangebin_double_reference() -> None:
    i_raw = np.array(
        [[2, 9], [5, 4], [11, 7], [3, 15], [8, 6], [13, 12]],
        dtype=np.uint8,
    )
    q_raw = np.array(
        [[12, 3], [7, 14], [4, 8], [15, 5], [6, 11], [10, 2]],
        dtype=np.uint8,
    )

    actual, stats = official_ipixload_auto(i_raw, q_raw)
    expected = _official_scalar_reference(i_raw, q_raw)

    assert actual.dtype == np.complex128
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(stats["std_i"], np.std(i_raw.astype(np.float64), axis=0, ddof=0))
    assert OFFICIAL_IPIXLOAD_SHA256 == "40f5499eeb0d1b1d7e158658bdcfddb31a18ef01eea4023e7da8ca0ba21bd517"


def test_official_ipixload_auto_does_not_use_matlab_default_n_minus_one() -> None:
    i_raw = np.array([[1], [2], [5], [9]], dtype=np.uint8)
    q_raw = np.array([[8], [3], [6], [1]], dtype=np.uint8)

    _, stats = official_ipixload_auto(i_raw, q_raw)

    population_std = np.std(i_raw.astype(np.float64), axis=0, ddof=0)
    sample_std = np.std(i_raw.astype(np.float64), axis=0, ddof=1)
    np.testing.assert_array_equal(stats["std_i"], population_std)
    assert not np.array_equal(stats["std_i"], sample_std)


def test_preprocess_defaults_to_official_ipixload_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["preprocess_ipix.py"])

    args = parse_args()

    assert args.processing_mode == "official_ipixload_auto"
    assert args.stats_scope == "full_file"


def test_legacy_preprocessing_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["preprocess_ipix.py", "--processing-mode", "legacy_numpy_float32"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_train_only_statistics_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["preprocess_ipix.py", "--stats-scope", "train_only"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_ipix_loader_requires_declared_official_processing_mode(tmp_path: Path) -> None:
    official_path = tmp_path / "sample__hh__train.npz"
    np.savez(
        official_path,
        E=np.ones((2, 4, 14), dtype=np.complex128),
        y_range=np.zeros((2, 14), dtype=np.uint8),
        processing_mode=np.array("official_ipixload_auto"),
    )

    x, y = load_ipix_arrays(
        official_path,
        expected_processing_mode="official_ipixload_auto",
    )

    assert x.dtype == np.complex64
    assert y.dtype == np.int64

    with pytest.raises(ValueError, match="预处理模式不匹配"):
        load_ipix_arrays(official_path, expected_processing_mode="not_official")
