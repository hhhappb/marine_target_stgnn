from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from paper_modules.datasets import build_dataset
from paper_modules.datasets.ipix_window import (
    _non_circular_clutter_fill,
    _sample_non_circular_shift,
)
from paper_modules.datasets.registry import validate_ipix_source_split
from paper_modules.experiments.auto_experiment import comparable_mismatches, config_metadata


def test_non_circular_fill_moves_profile_and_labels_without_wrap(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size: torch.zeros(size, dtype=torch.int64),
    )
    real = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    imag = real + 10
    labels = torch.tensor([0, 0, 1, 0, 0, 0])
    roles = torch.tensor([0, 1, 2, 1, 0, 0], dtype=torch.int8)

    shifted_real, shifted_imag, shifted_labels = _non_circular_clutter_fill(
        real, imag, labels, roles, shift=2
    )

    assert shifted_real.tolist() == [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]
    assert shifted_imag.tolist() == [[10.0, 10.0, 10.0, 11.0, 12.0, 13.0]]
    assert shifted_labels.tolist() == [0, 0, 0, 0, 1, 0]


def test_non_circular_shift_is_bidirectional_legal_and_nonzero() -> None:
    roles = torch.tensor([0, 1, 2, 1, 0, 0], dtype=torch.int8)
    shifts = {
        _sample_non_circular_shift(roles, max_shift=5, include_identity=False)
        for _ in range(100)
    }
    assert shifts <= {-1, 1, 2}
    assert 0 not in shifts
    assert shifts


def test_non_circular_fill_rejects_target_neighborhood_clipping() -> None:
    real = torch.zeros((1, 6))
    labels = torch.zeros(6, dtype=torch.int64)
    roles = torch.tensor([0, 1, 2, 1, 0, 0], dtype=torch.int8)
    with pytest.raises(ValueError, match="截断 target-related"):
        _non_circular_clutter_fill(real, real, labels, roles, shift=-2)


def _write_npz(path: Path, roles: np.ndarray) -> None:
    echoes = np.arange(24, dtype=np.float32).reshape(2, 2, 6).astype(np.complex64)
    labels = np.broadcast_to((roles == 2).astype(np.int64), (2, 6)).copy()
    np.savez(
        path,
        E=echoes,
        y_range=labels,
        range_roles=roles,
        processing_mode=np.array("official_ipixload_auto"),
    )


def test_build_dataset_uses_disjoint_sources_and_train_only_augmentation(tmp_path: Path) -> None:
    roles = np.array([0, 1, 2, 1, 0, 0], dtype=np.int8)
    _write_npz(tmp_path / "source_a__hh__train.npz", roles)
    _write_npz(tmp_path / "source_b__hh__test.npz", roles)
    config = {
        "dataset": {
            "type": "ipix_window",
            "data_dir": str(tmp_path),
            "polarizations": ["hh"],
            "train_sources": ["source_a"],
            "test_sources": ["source_b"],
            "require_disjoint_train_test_sources": True,
            "augment": {
                "range_roll": {
                    "enabled": True,
                    "mode": "clutter_fill",
                    "max_shift": 2,
                    "include_identity": False,
                }
            },
        },
        "train": {"seed": 42},
    }

    train_dataset = build_dataset(config, "train")
    test_dataset = build_dataset(config, "test")

    assert [path.stem for path in train_dataset.files] == ["source_a__hh__train"]
    assert [path.stem for path in test_dataset.files] == ["source_b__hh__test"]
    assert train_dataset._range_roll["mode"] == "clutter_fill"
    assert train_dataset._range_roll["enabled"] is True
    assert test_dataset._range_roll["enabled"] is False


def test_disjoint_source_validation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="source 重叠"):
        validate_ipix_source_split(
            {
                "train_sources": ["same"],
                "test_sources": ["same"],
                "require_disjoint_train_test_sources": True,
            }
        )


def test_config_metadata_builds_cross_file_source_unit() -> None:
    metadata = config_metadata(
        {
            "paths": {"data_dir": "data"},
            "model": {"name": "original_stgnn"},
            "dataset": {
                "type": "ipix_window",
                "data_dir": "data",
                "polarizations": ["hh"],
                "train_sources": ["source_a"],
                "test_sources": ["source_b"],
                "require_disjoint_train_test_sources": True,
            },
            "eval": {"protocol": "per_file_pol", "threshold_source": "train_clutter"},
            "train": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
        }
    )

    assert metadata["source_unit"] == "source_a->source_b"
    assert metadata["sources"] == ["source_a", "source_b"]
    assert metadata["train_sources"] == ["source_a"]
    assert metadata["test_sources"] == ["source_b"]
    assert metadata["require_disjoint_train_test_sources"] is True


def test_comparable_mismatches_detects_cross_file_source_changes() -> None:
    baseline = {
        "data_dir": "data",
        "expected_processing_mode": "official_ipixload_auto",
        "train_sources": ["source_a"],
        "test_sources": ["source_b"],
    }
    candidate = {**baseline, "test_sources": ["source_c"]}

    assert "test_sources" in comparable_mismatches(candidate, baseline)