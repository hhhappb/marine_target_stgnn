from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


IPIX_LABEL_IGNORE_INDEX = -100
IPIX_LABEL_POLICIES = {
    "stored",
    "related",
    "primary_strict",
    "primary_with_related_ignore",
}
IPIX_SECONDARY_ECHO_POLICIES = {
    "stored",
    "clutter_permutation",
}
IPIX_RANGE_ROLL_MODES = {
    "circular",
    "clutter_fill",
}


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


def load_ipix_arrays(
    path: Path,
    max_windows: int | None = None,
    rng: np.random.Generator | None = None,
    window_fraction_range: list[float] | tuple[float, float] | None = None,
    label_policy: str = "stored",
    secondary_echo_policy: str = "stored",
    expected_processing_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if expected_processing_mode is not None:
            if "processing_mode" not in data:
                raise ValueError(f"IPIX 文件缺少 processing_mode，无法验证预处理来源：{path}")
            actual_processing_mode = str(data["processing_mode"].item())
            if actual_processing_mode != expected_processing_mode:
                raise ValueError(
                    f"IPIX 预处理模式不匹配：期望 {expected_processing_mode}，"
                    f"实际 {actual_processing_mode}，文件 {path}"
                )
        x = data["E"]
        y = _labels_for_policy(data, np.asarray(data["y_range"]), label_policy)
        if window_fraction_range is not None:
            start, end = _window_fraction_bounds(len(x), window_fraction_range)
            x = x[start:end]
            y = y[start:end]
        if max_windows is not None and len(x) > max_windows:
            if rng is None:
                rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(len(x), size=max_windows, replace=False))
            x = x[idx]
            y = y[idx]
        x = _secondary_echoes_for_policy(
            data,
            np.asarray(x),
            label_policy=label_policy,
            secondary_echo_policy=secondary_echo_policy,
            rng=rng,
        )
        return x.astype(np.complex64, copy=False), y.astype(np.int64, copy=False)


class IpixWindowDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        max_windows: int | None = None,
        seed: int = 42,
        range_roll: dict[str, Any] | None = None,
        window_fraction_range: list[float] | tuple[float, float] | None = None,
        label_policy: str = "stored",
        secondary_echo_policy: str = "stored",
        expected_processing_mode: str | None = None,
    ):
        self.files = files
        self.x_parts: list[np.ndarray] = []
        self.y_parts: list[np.ndarray] = []
        self.range_role_parts: list[np.ndarray] = []
        self.rng = np.random.default_rng(seed)
        self._range_roll = _parse_range_roll(range_roll)
        remaining = max_windows

        for path in files:
            if remaining is not None and remaining <= 0:
                break
            limit = remaining
            x, y = load_ipix_arrays(
                path,
                max_windows=limit,
                rng=self.rng,
                window_fraction_range=window_fraction_range,
                label_policy=label_policy,
                secondary_echo_policy=secondary_echo_policy,
                expected_processing_mode=expected_processing_mode,
            )
            self.x_parts.append(x)
            self.y_parts.append(y)
            if self._range_roll["enabled"] and self._range_roll["mode"] == "clutter_fill":
                roles = _load_range_roles(path, int(x.shape[2]))
                self.range_role_parts.append(
                    np.broadcast_to(roles, (len(x), len(roles))).copy()
                )
            if remaining is not None:
                remaining -= len(x)

        if not self.x_parts:
            raise ValueError("No IPIX windows were loaded. Check data_dir, split, polarizations, and max_windows.")

        x = np.concatenate(self.x_parts, axis=0)
        y = np.concatenate(self.y_parts, axis=0)
        valid_labels = y[(y == 0) | (y == 1)]
        if valid_labels.size == 0:
            raise ValueError("IPIX 标签策略没有产生任何 target/clutter 单元。")
        counts = np.bincount(valid_labels, minlength=2).astype(np.float64)
        if np.any(counts == 0):
            self._class_weights = torch.ones(2, dtype=torch.float32)
        else:
            self._class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)

        self.real = torch.from_numpy(np.ascontiguousarray(x.real, dtype=np.float32))
        self.imag = torch.from_numpy(np.ascontiguousarray(x.imag, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self._range_roll = _finalize_range_roll(self._range_roll, int(self.y.shape[1]))
        self.range_roles = (
            torch.from_numpy(
                np.ascontiguousarray(np.concatenate(self.range_role_parts, axis=0), dtype=np.int8)
            )
            if self.range_role_parts
            else None
        )

        self.x_parts = []
        self.y_parts = []
        self.range_role_parts = []

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real = self.real[idx]
        imag = self.imag[idx]
        y = self.y[idx]
        if self._range_roll["enabled"]:
            max_shift = int(self._range_roll["max_shift"])
            if self._range_roll["mode"] == "circular":
                shift = int(torch.randint(0, max_shift + 1, (1,)).item())
                real = torch.roll(real, shifts=shift, dims=-1)
                imag = torch.roll(imag, shifts=shift, dims=-1)
                y = torch.roll(y, shifts=shift, dims=-1)
            else:
                if self.range_roles is None:
                    raise RuntimeError("clutter_fill 增强缺少 range_roles。")
                roles = self.range_roles[idx]
                shift = _sample_non_circular_shift(
                    roles,
                    max_shift,
                    include_identity=bool(self._range_roll["include_identity"]),
                )
                real, imag, y = _non_circular_clutter_fill(real, imag, y, roles, shift)
        return real, imag, y

    def class_weights(self) -> torch.Tensor:
        return self._class_weights.clone()


def _labels_for_policy(data: Any, stored_labels: np.ndarray, label_policy: str) -> np.ndarray:
    policy = str(label_policy)
    if policy not in IPIX_LABEL_POLICIES:
        raise ValueError(
            f"dataset.label_policy 仅支持 {sorted(IPIX_LABEL_POLICIES)}，实际为 {policy!r}。"
        )
    if stored_labels.ndim != 2:
        raise ValueError(f"IPIX y_range 必须是二维数组，实际 shape={stored_labels.shape}。")
    if policy == "stored":
        return stored_labels
    if "range_roles" not in data:
        raise ValueError(f"label_policy={policy} 需要 NPZ 中存在 range_roles。")

    roles = np.asarray(data["range_roles"], dtype=np.int8)
    if roles.shape != (stored_labels.shape[1],):
        raise ValueError(
            "IPIX range_roles 与 y_range 的距离单元维不一致："
            f"roles={roles.shape}, y_range={stored_labels.shape}。"
        )
    if not np.all(np.isin(roles, [0, 1, 2])):
        raise ValueError(f"IPIX range_roles 仅允许0/1/2，实际为 {np.unique(roles).tolist()}。")
    if int(np.count_nonzero(roles == 2)) != 1:
        raise ValueError("IPIX range_roles 必须且只能包含一个 primary(role=2)距离单元。")

    labels = np.zeros(roles.shape[0], dtype=np.int64)
    if policy == "related":
        labels[roles > 0] = 1
    elif policy == "primary_strict":
        labels[roles == 2] = 1
    else:
        labels[roles == 1] = IPIX_LABEL_IGNORE_INDEX
        labels[roles == 2] = 1
    return np.broadcast_to(labels, stored_labels.shape).copy()


def _secondary_echoes_for_policy(
    data: Any,
    echoes: np.ndarray,
    label_policy: str,
    secondary_echo_policy: str,
    rng: np.random.Generator | None,
) -> np.ndarray:
    policy = str(secondary_echo_policy)
    if policy not in IPIX_SECONDARY_ECHO_POLICIES:
        raise ValueError(
            "dataset.secondary_echo_policy 仅支持 "
            f"{sorted(IPIX_SECONDARY_ECHO_POLICIES)}，实际为 {policy!r}。"
        )
    if policy == "stored":
        return echoes
    if label_policy != "primary_with_related_ignore":
        raise ValueError(
            "secondary_echo_policy=clutter_permutation 要求 "
            "label_policy=primary_with_related_ignore。"
        )
    if "range_roles" not in data:
        raise ValueError("clutter_permutation 需要 NPZ 中存在 range_roles。")
    if echoes.ndim != 3:
        raise ValueError(f"IPIX E 必须是 [windows, P, N]，实际 shape={echoes.shape}。")

    roles = np.asarray(data["range_roles"], dtype=np.int8)
    if roles.shape != (echoes.shape[2],):
        raise ValueError(
            "IPIX range_roles 与 E 的距离单元维不一致："
            f"roles={roles.shape}, E={echoes.shape}。"
        )
    secondary_indices = np.flatnonzero(roles == 1)
    clutter_indices = np.flatnonzero(roles == 0)
    if secondary_indices.size == 0:
        raise ValueError("clutter_permutation 至少需要一个 secondary(role=1) 单元。")
    if clutter_indices.size == 0:
        raise ValueError("clutter_permutation 至少需要一个 clutter(role=0) 单元。")
    if echoes.shape[0] == 0:
        raise ValueError("clutter_permutation 不能处理空窗口集合。")
    if rng is None:
        rng = np.random.default_rng(0)

    source = echoes
    replaced = echoes.copy()
    window_indices = np.arange(echoes.shape[0])
    for secondary_index in secondary_indices:
        if echoes.shape[0] > 1:
            shift = int(rng.integers(1, echoes.shape[0]))
            donor_windows = np.roll(window_indices, shift)
        else:
            donor_windows = window_indices
        donor_cells = rng.choice(clutter_indices, size=echoes.shape[0], replace=True)
        replaced[:, :, secondary_index] = source[donor_windows, :, donor_cells]
    return replaced


def _parse_window_fraction_range(values: list[float] | tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"window_fraction_range 必须包含[start, end]两个值，实际为 {values}。")
    start, end = (float(values[0]), float(values[1]))
    if not 0.0 <= start < end <= 1.0:
        raise ValueError(f"window_fraction_range 必须满足0<=start<end<=1，实际为 {values}。")
    return start, end


def _window_fraction_bounds(
    windows: int,
    values: list[float] | tuple[float, float],
) -> tuple[int, int]:
    if windows <= 0:
        raise ValueError(f"window_fraction_range 需要正窗口数，实际为 {windows}。")
    start_fraction, end_fraction = _parse_window_fraction_range(values)
    start = int(np.floor(windows * start_fraction))
    end = windows if end_fraction == 1.0 else int(np.floor(windows * end_fraction))
    if end <= start:
        raise ValueError(
            f"window_fraction_range 的时间比例切片为空：windows={windows}, range={values}。"
        )
    return start, end


def _parse_range_roll(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    enabled = bool(config.get("enabled", False))
    mode = str(config.get("mode", "circular"))
    if mode not in IPIX_RANGE_ROLL_MODES:
        raise ValueError(
            f"range_roll.mode 仅支持 {sorted(IPIX_RANGE_ROLL_MODES)}，实际为 {mode}。"
        )
    return {
        "enabled": enabled,
        "max_shift": config.get("max_shift"),
        "mode": mode,
        "include_identity": bool(config.get("include_identity", True)),
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

def _load_range_roles(path: Path, range_cells: int) -> np.ndarray:
    with np.load(path) as data:
        if "range_roles" not in data:
            raise ValueError(f"clutter_fill 需要 NPZ 中存在 range_roles：{path}")
        roles = np.asarray(data["range_roles"], dtype=np.int8)
    if roles.shape != (range_cells,):
        raise ValueError(f"IPIX range_roles 必须为 ({range_cells},)，实际为 {roles.shape}：{path}")
    if not np.all(np.isin(roles, [0, 1, 2])):
        raise ValueError(f"IPIX range_roles 仅允许0/1/2，实际为 {np.unique(roles).tolist()}。")
    if int(np.count_nonzero(roles == 2)) != 1:
        raise ValueError("clutter_fill 要求 range_roles 恰好包含一个 primary(role=2) 单元。")
    if not np.any(roles == 0):
        raise ValueError("clutter_fill 至少需要一个真实 clutter(role=0) 单元作为填充来源。")
    return roles


def _sample_non_circular_shift(
    roles: torch.Tensor,
    max_shift: int,
    include_identity: bool,
) -> int:
    if roles.ndim != 1:
        raise ValueError(f"range_roles 必须是一维，实际 shape={tuple(roles.shape)}。")
    related = torch.nonzero(roles > 0, as_tuple=False).flatten()
    if related.numel() == 0:
        raise ValueError("clutter_fill 要求至少一个 target-related 距离单元。")
    range_cells = int(roles.numel())
    minimum = max(-max_shift, -int(related.min().item()))
    maximum = min(max_shift, range_cells - 1 - int(related.max().item()))
    candidates = torch.arange(minimum, maximum + 1, dtype=torch.int64)
    if not include_identity:
        candidates = candidates[candidates != 0]
    if candidates.numel() == 0:
        raise ValueError("clutter_fill 在当前 target-related 范围和 max_shift 下没有合法非零位移。")
    choice = int(torch.randint(0, int(candidates.numel()), (1,)).item())
    return int(candidates[choice].item())


def _non_circular_clutter_fill(
    real: torch.Tensor,
    imag: torch.Tensor,
    labels: torch.Tensor,
    roles: torch.Tensor,
    shift: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if real.shape != imag.shape or real.ndim != 2:
        raise ValueError(
            "clutter_fill 要求 real/imag 具有相同的 [P, N] shape，"
            f"实际 real={tuple(real.shape)}, imag={tuple(imag.shape)}。"
        )
    range_cells = int(real.shape[-1])
    if labels.shape != (range_cells,) or roles.shape != (range_cells,):
        raise ValueError(
            "clutter_fill 要求 labels/range_roles 均为 [N]，"
            f"实际 labels={tuple(labels.shape)}, roles={tuple(roles.shape)}, N={range_cells}。"
        )
    if shift == 0:
        return real, imag, labels
    if abs(shift) >= range_cells:
        raise ValueError(f"clutter_fill 位移必须满足 abs(shift)<{range_cells}，实际为 {shift}。")
    related = torch.nonzero(roles > 0, as_tuple=False).flatten()
    if related.numel() == 0:
        raise ValueError("clutter_fill 要求至少一个 target-related 距离单元。")
    shifted_related = related + shift
    if int(shifted_related.min().item()) < 0 or int(shifted_related.max().item()) >= range_cells:
        raise ValueError("clutter_fill 位移会截断 target-related 邻域，拒绝执行。")
    clutter_indices = torch.nonzero(roles == 0, as_tuple=False).flatten()
    if clutter_indices.numel() == 0:
        raise ValueError("clutter_fill 至少需要一个真实 clutter(role=0) 单元作为填充来源。")

    shifted_real = torch.empty_like(real)
    shifted_imag = torch.empty_like(imag)
    shifted_labels = torch.empty_like(labels)
    if shift > 0:
        shifted_real[..., shift:] = real[..., :-shift]
        shifted_imag[..., shift:] = imag[..., :-shift]
        shifted_labels[shift:] = labels[:-shift]
        vacated = slice(0, shift)
        vacated_count = shift
    else:
        shifted_real[..., :shift] = real[..., -shift:]
        shifted_imag[..., :shift] = imag[..., -shift:]
        shifted_labels[:shift] = labels[-shift:]
        vacated = slice(shift, None)
        vacated_count = -shift
    donor_choices = torch.randint(0, int(clutter_indices.numel()), (vacated_count,))
    donor_indices = clutter_indices[donor_choices]
    shifted_real[..., vacated] = real[..., donor_indices]
    shifted_imag[..., vacated] = imag[..., donor_indices]
    shifted_labels[vacated] = 0
    return shifted_real, shifted_imag, shifted_labels
