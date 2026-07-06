from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_scr(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("test_scr_"):
        raise ValueError(f"无法从文件名解析 SCR: {path}")
    return int(stem.replace("test_scr_", ""))


def list_test_scr_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("test_scr_*.npz"), key=_parse_scr)


def compute_train_norm(data_dir: Path) -> tuple[float, float, float, float]:
    train_path = data_dir / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"缺少 SCR 训练文件: {train_path}")
    with np.load(train_path) as data:
        x_train = data["X"]
        _validate_x(x_train, train_path)
        nr = x_train[:, 0].reshape(-1)
        ni = x_train[:, 1].reshape(-1)
        return float(nr.mean()), float(nr.std()), float(ni.mean()), float(ni.std())


def load_scr_arrays(
    data_dir: Path,
    split: str,
    scr: int | None = None,
    max_windows: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = _resolve_files(data_dir, split, scr)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    scr_parts: list[np.ndarray] = []
    remaining = max_windows

    for path in files:
        if remaining is not None and remaining <= 0:
            break
        with np.load(path) as data:
            x = data["X"]
            y = data["y"]
            _validate_x(x, path)
            if y.ndim != 2 or y.shape != (x.shape[0], x.shape[-1]):
                raise ValueError(f"{path} 的 y shape 应为 [B, N]，实际为 {y.shape}。")
            if remaining is not None and len(x) > remaining:
                if rng is None:
                    rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(len(x), size=remaining, replace=False))
                x = x[idx]
                y = y[idx]
            x_parts.append(x.astype(np.float32, copy=False))
            y_parts.append(y.astype(np.int64, copy=False))
            scr_value = -9999 if split == "train" else _parse_scr(path)
            scr_parts.append(np.full((len(x),), scr_value, dtype=np.int64))
            if remaining is not None:
                remaining -= len(x)

    if not x_parts:
        raise ValueError(f"没有加载到 SCR {split} 数据：data_dir={data_dir}, scr={scr}")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0), np.concatenate(scr_parts, axis=0)


class ScrNpzDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        scr: int | None = None,
        max_windows: int | None = None,
        seed: int = 42,
        norm: tuple[float, float, float, float] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.scr = scr
        self.rng = np.random.default_rng(seed)
        self.norm = norm if norm is not None else compute_train_norm(self.data_dir)

        x, y, scr_values = load_scr_arrays(self.data_dir, split, scr=scr, max_windows=max_windows, rng=self.rng)
        nr_mean, nr_std, ni_mean, ni_std = self.norm
        real = np.clip((x[:, 0] - nr_mean) / (nr_std + 1e-8), -5, 5)
        imag = np.clip((x[:, 1] - ni_mean) / (ni_std + 1e-8), -5, 5)

        counts = np.bincount(y.reshape(-1), minlength=2).astype(np.float64)
        if np.any(counts == 0):
            self._class_weights = torch.ones(2, dtype=torch.float32)
        else:
            self._class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)

        self.real = torch.from_numpy(np.ascontiguousarray(real, dtype=np.float32))
        self.imag = torch.from_numpy(np.ascontiguousarray(imag, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.scr_values = torch.from_numpy(np.ascontiguousarray(scr_values, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.real[idx], self.imag[idx], self.y[idx]

    def class_weights(self) -> torch.Tensor:
        return self._class_weights.clone()


def _resolve_files(data_dir: Path, split: str, scr: int | None) -> list[Path]:
    if split == "train":
        path = data_dir / "train.npz"
        if not path.exists():
            raise FileNotFoundError(f"缺少 SCR 训练文件: {path}")
        return [path]
    if split != "test":
        raise ValueError(f"SCR 数据集只支持 split=train/test，收到: {split}")
    if scr is not None:
        path = data_dir / f"test_scr_{scr}.npz"
        if not path.exists():
            raise FileNotFoundError(f"缺少 SCR 测试文件: {path}")
        return [path]
    files = list_test_scr_files(data_dir)
    if not files:
        raise FileNotFoundError(f"缺少 SCR 测试文件: {data_dir / 'test_scr_*.npz'}")
    return files


def _validate_x(x: np.ndarray, path: Path) -> None:
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"{path} 的 X shape 应为 [B, 2, P, N]，实际为 {x.shape}。")
