from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_split_files(data_dir: Path, split: str, pols: list[str]) -> list[Path]:
    files: list[Path] = []
    for pol in pols:
        files.extend(data_dir.glob(f"*__{pol}__{split}.npz"))
    return sorted(files)


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
    def __init__(self, files: list[Path], max_windows: int | None = None, seed: int = 42):
        self.files = files
        self.x_parts: list[np.ndarray] = []
        self.y_parts: list[np.ndarray] = []
        self.rng = np.random.default_rng(seed)
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

        self.x_parts = []
        self.y_parts = []

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.real[idx], self.imag[idx], self.y[idx]

    def class_weights(self) -> torch.Tensor:
        return self._class_weights.clone()
