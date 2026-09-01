from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from paper_modules.datasets.ipix_window import (
    IPIX_LABEL_IGNORE_INDEX,
    IpixWindowDataset,
    load_ipix_arrays,
)
from paper_modules.datasets.registry import build_dataset
from paper_modules.experiments.train import evaluate_files
from paper_modules.losses.cross_entropy import CrossEntropyDetectionLoss


class ConstantDetector(nn.Module):
    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        batch, _, range_cells = echoes.shape
        return torch.zeros(batch, 2, range_cells, dtype=torch.float32, device=echoes.device)


def write_ipix_npz(path: Path, windows: int = 2) -> None:
    real = np.arange(windows * 2 * 4, dtype=np.float32).reshape(windows, 2, 4)
    echoes = real.astype(np.complex64)
    related = np.broadcast_to(np.array([0, 1, 1, 1], dtype=np.int64), (windows, 4)).copy()
    np.savez(
        path,
        E=echoes,
        y_range=related,
        range_roles=np.array([0, 1, 2, 1], dtype=np.int8),
        primary_range_bin=np.array(3, dtype=np.int16),
        secondary_range_bins=np.array([2, 3, 4], dtype=np.int16),
    )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("stored", [0, 1, 1, 1]),
        ("related", [0, 1, 1, 1]),
        ("primary_strict", [0, 0, 1, 0]),
        (
            "primary_with_related_ignore",
            [0, IPIX_LABEL_IGNORE_INDEX, 1, IPIX_LABEL_IGNORE_INDEX],
        ),
    ],
)
def test_ipix_label_policies_are_derived_from_range_roles(
    tmp_path: Path,
    policy: str,
    expected: list[int],
) -> None:
    path = tmp_path / "sample__hh__train.npz"
    write_ipix_npz(path)

    _, labels = load_ipix_arrays(path, label_policy=policy)

    np.testing.assert_array_equal(labels, np.broadcast_to(expected, labels.shape))


def test_unknown_label_policy_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "sample__hh__train.npz"
    write_ipix_npz(path)

    with pytest.raises(ValueError, match="dataset.label_policy"):
        load_ipix_arrays(path, label_policy="unknown")


def test_guard_cells_are_excluded_from_class_weights(tmp_path: Path) -> None:
    path = tmp_path / "sample__hh__train.npz"
    write_ipix_npz(path)

    dataset = IpixWindowDataset([path], label_policy="primary_with_related_ignore")

    torch.testing.assert_close(dataset.class_weights(), torch.ones(2))
    assert int(torch.count_nonzero(dataset.y == IPIX_LABEL_IGNORE_INDEX)) == 4


def test_train_and_evaluation_label_policies_can_be_frozen_separately(
    tmp_path: Path,
) -> None:
    write_ipix_npz(tmp_path / "sample__hh__train.npz")
    write_ipix_npz(tmp_path / "sample__hh__test.npz")
    config = {
        "dataset": {
            "type": "ipix_window",
            "data_dir": str(tmp_path),
            "sources": ["sample"],
            "polarizations": ["hh"],
            "train_label_policy": "related",
            "evaluation_label_policy": "primary_with_related_ignore",
        },
        "train": {"seed": 42},
    }

    train_dataset = build_dataset(config, "train")
    test_dataset = build_dataset(config, "test")

    torch.testing.assert_close(train_dataset.y[0], torch.tensor([0, 1, 1, 1]))
    torch.testing.assert_close(
        test_dataset.y[0],
        torch.tensor([0, IPIX_LABEL_IGNORE_INDEX, 1, IPIX_LABEL_IGNORE_INDEX]),
    )


def test_range_roll_moves_echoes_labels_and_guard_cells_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample__hh__train.npz"
    write_ipix_npz(path, windows=1)
    dataset = IpixWindowDataset(
        [path],
        label_policy="primary_with_related_ignore",
        range_roll={"enabled": True, "mode": "circular", "max_shift": 3},
    )
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([1]))

    real, _, labels = dataset[0]

    torch.testing.assert_close(real, torch.roll(dataset.real[0], shifts=1, dims=-1))
    torch.testing.assert_close(labels, torch.roll(dataset.y[0], shifts=1, dims=-1))


def test_cross_entropy_ignores_guard_cells() -> None:
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 4, requires_grad=True)
    labels = torch.tensor([[0, IPIX_LABEL_IGNORE_INDEX, 1, IPIX_LABEL_IGNORE_INDEX]])
    expected = nn.CrossEntropyLoss()(logits[:, :, [0, 2]], labels[:, [0, 2]])

    actual = CrossEntropyDetectionLoss()(logits, labels)

    torch.testing.assert_close(actual, expected)


def test_threshold_and_metrics_exclude_guard_cells(tmp_path: Path) -> None:
    train_path = tmp_path / "sample__hh__train.npz"
    test_path = tmp_path / "sample__hh__test.npz"
    write_ipix_npz(train_path)
    write_ipix_npz(test_path)

    results = evaluate_files(
        model=ConstantDetector(),
        files=[test_path],
        batch_size=2,
        device=torch.device("cpu"),
        pfa_values=[0.5],
        threshold_files=[train_path],
        threshold_source="train_clutter",
        label_policy="primary_with_related_ignore",
    )

    assert results["num_target_bins"] == 2
    assert results["num_clutter_bins"] == 2
    assert results["num_ignore_bins"] == 4
    assert results["num_clutter_bins_for_threshold"] == 2
    assert results["pfa"]["0.5"]["TP"] == 2
    assert results["pfa"]["0.5"]["FP"] == 2
    assert results["pfa"]["0.5"]["FN"] == 0
    assert results["pfa"]["0.5"]["TN"] == 0
