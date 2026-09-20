from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from paper_modules.datasets.registry import build_dataset
from paper_modules.datasets.ipix_window import _non_circular_clutter_fill


def _write_split(path: Path) -> None:
    cells = np.arange(7, dtype=np.float32)
    echoes = (cells[None, None, :] + 10 * np.arange(2)[None, :, None]).astype(np.complex64)
    roles = np.array([0, 0, 1, 2, 1, 0, 0], dtype=np.int8)
    labels = (roles == 2).astype(np.int64)[None, :]
    np.savez(path, E=echoes, y_range=labels, range_roles=roles)


@pytest.mark.parametrize("shift", [-1, 1])
def test_clutter_fill_shifts_target_without_wrapping_or_reusing_target_as_fill(shift: int) -> None:
    cells = torch.arange(7, dtype=torch.float32)
    real = torch.stack((cells, cells + 10))
    imag = real + 20
    roles = torch.tensor([0, 0, 1, 2, 1, 0, 0], dtype=torch.int8)
    labels = (roles == 2).long()

    shifted_real, shifted_imag, shifted_labels = _non_circular_clutter_fill(
        real, imag, labels, roles, shift
    )

    assert torch.nonzero(shifted_labels).flatten().tolist() == [3 + shift]
    torch.testing.assert_close(shifted_real[:, 3 + shift], real[:, 3])
    torch.testing.assert_close(shifted_imag[:, 3 + shift], imag[:, 3])
    vacated = 0 if shift > 0 else -1
    assert int(shifted_labels[vacated]) == 0
    assert shifted_real[0, vacated].item() in {0, 1, 5, 6}


def test_clutter_fill_is_train_only_in_dataset_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_split(tmp_path / "sample__hh__train.npz")
    _write_split(tmp_path / "sample__hh__test.npz")
    config = {
        "dataset": {
            "type": "ipix_window",
            "data_dir": str(tmp_path),
            "sources": ["sample"],
            "polarizations": ["hh"],
            "label_policy": "primary_strict",
            "augment": {"range_roll": {"enabled": True, "mode": "clutter_fill", "max_shift": 1}},
        }
    }
    train = build_dataset(config, "train")
    test = build_dataset(config, "test")
    monkeypatch.setattr(
        "paper_modules.datasets.ipix_window._sample_non_circular_shift",
        lambda roles, max_shift, include_identity: 1,
    )

    train_real, _, train_labels = train[0]
    test_real, _, test_labels = test[0]
    assert torch.nonzero(train_labels).flatten().tolist() == [4]
    assert torch.nonzero(test_labels).flatten().tolist() == [3]
    torch.testing.assert_close(train_real[:, 4], test_real[:, 3])
    torch.testing.assert_close(test_real, test.real[0])


def test_clutter_fill_requires_range_roles(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "sample__hh__train.npz",
        E=np.zeros((1, 2, 7), dtype=np.complex64),
        y_range=np.zeros((1, 7), dtype=np.int64),
    )
    config = {
        "dataset": {
            "type": "ipix_window",
            "data_dir": str(tmp_path),
            "polarizations": ["hh"],
            "augment": {"range_roll": {"enabled": True, "mode": "clutter_fill", "max_shift": 1}},
        }
    }
    with pytest.raises(ValueError, match="range_roles"):
        build_dataset(config, "train")
