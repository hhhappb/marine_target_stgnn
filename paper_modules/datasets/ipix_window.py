from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_split_files(data_dir: Path, split: str, pols: list[str], sources: list[str] | None = None) -> list[Path]:
    files: list[Path] = []
    source_set = set(sources) if sources is not None else None
    for pol in pols:
        files.extend(data_dir.glob(f"*__{pol}__{split}.npz"))
    if source_set is None:
        return sorted(files)
    return sorted(path for path in files if parse_source_and_pol(path)[0] in source_set)


def parse_source_and_pol(path: Path) -> tuple[str, str]:
    parts = path.stem.split("__")
    if len(parts) < 3:
        return path.stem, "unknown"
    return parts[0], parts[1]


def load_ipix_arrays(path: Path, max_windows: int | None = None, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        x = data["E"]
        y = data["y_range"]
        if max_windows is not None and len(x) > max_windows:
            if rng is None:
                rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(len(x), size=max_windows, replace=False))
            x = x[idx]
            y = y[idx]
        return x.astype(np.complex64, copy=False), y.astype(np.int64, copy=False)


class IpixWindowDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        max_windows: int | None = None,
        seed: int = 42,
        range_roll: dict[str, Any] | None = None,
    ):
        self.files = files
        self.x_parts: list[np.ndarray] = []
        self.y_parts: list[np.ndarray] = []
        self.rng = np.random.default_rng(seed)
        self._range_roll = _parse_range_roll(range_roll)
        remaining = max_windows

        for path in files:
            if remaining is not None and remaining <= 0:
                break
            limit = remaining
            x, y = load_ipix_arrays(path, max_windows=limit, rng=self.rng)
            self.x_parts.append(x)
            self.y_parts.append(y)
            if remaining is not None:
                remaining -= len(x)

        if not self.x_parts:
            raise ValueError("No IPIX windows were loaded. Check data_dir, split, polarizations, and max_windows.")

        x = np.concatenate(self.x_parts, axis=0)
        y = np.concatenate(self.y_parts, axis=0)
        counts = np.bincount(y.reshape(-1), minlength=2).astype(np.float64)
        if np.any(counts == 0):
            self._class_weights = torch.ones(2, dtype=torch.float32)
        else:
            self._class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)

        self.real = torch.from_numpy(np.ascontiguousarray(x.real, dtype=np.float32))
        self.imag = torch.from_numpy(np.ascontiguousarray(x.imag, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self._range_roll = _finalize_range_roll(self._range_roll, int(self.y.shape[1]))

        self.x_parts = []
        self.y_parts = []

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real = self.real[idx]
        imag = self.imag[idx]
        y = self.y[idx]
        if self._range_roll["enabled"]:
            max_shift = int(self._range_roll["max_shift"])
            shift = int(torch.randint(0, max_shift + 1, (1,)).item())
            if shift:
                real = torch.roll(real, shifts=shift, dims=-1)
                imag = torch.roll(imag, shifts=shift, dims=-1)
                y = torch.roll(y, shifts=shift, dims=-1)
        return real, imag, y

    def class_weights(self) -> torch.Tensor:
        return self._class_weights.clone()


def _parse_range_roll(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    enabled = bool(config.get("enabled", False))
    mode = str(config.get("mode", "circular"))
    if mode != "circular":
        raise ValueError(f"range_roll.mode 仅支持 circular，实际为 {mode}。")
    return {
        "enabled": enabled,
        "max_shift": config.get("max_shift"),
        "mode": mode,
    }


def _finalize_range_roll(config: dict[str, Any], range_cells: int) -> dict[str, Any]:
    if not config["enabled"]:
        return {**config, "max_shift": 0}
    if range_cells < 2:
        raise ValueError("range_roll 需要至少 2 个 range cell。")
    max_shift = config["max_shift"]
    if max_shift is None:
        max_shift = range_cells - 1
    max_shift = int(max_shift)
    if max_shift <= 0 or max_shift >= range_cells:
        raise ValueError(f"range_roll.max_shift 必须在 [1, {range_cells - 1}] 内，实际为 {max_shift}。")
    return {**config, "max_shift": max_shift}
