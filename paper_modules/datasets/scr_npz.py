from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SDRDSP_V2_PROTOCOL = "sdrdsp_fig9_local_crop_v2"
SDRDSP_PHASE_RCS_PROTOCOL = "sdrdsp_fig9_local_crop_phase_rcs_v1"
SDRDSP_LOCAL_CROP_PROTOCOLS = {SDRDSP_V2_PROTOCOL, SDRDSP_PHASE_RCS_PROTOCOL}
TRAIN_SCR_VALUES = list(range(-12, 15, 2))
TEST_SCR_VALUES = list(range(-24, 15, 2))


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


def validate_sdrdsp_v2_manifest(
    data_dir: Path,
    expected_pulses: int | None,
    expected_range_cells: int | None,
) -> dict[str, Any]:
    """校验 v2 数据确实来自当前 Fig. 9 局部裁剪协议。"""
    if expected_pulses is None or expected_range_cells is None:
        raise ValueError("严格 SDRDSP v2 校验需要 model.pulses 和 model.range_cells。")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"严格 SDRDSP v2 数据缺少 manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 SDRDSP v2 manifest: {manifest_path}") from exc

    protocol_id = _manifest_value(manifest, "protocol.id")
    if protocol_id not in SDRDSP_LOCAL_CROP_PROTOCOLS:
        raise ValueError(f"未知 SDRDSP local-crop protocol: {protocol_id!r}。")
    expected_values = {
        "dataset": (
            "SDRDSP Fig. 9 local-crop reproduction"
            if protocol_id == SDRDSP_V2_PROTOCOL
            else "SDRDSP Fig. 9 local-crop target-model sensitivity"
        ),
        "protocol.id": protocol_id,
        "protocol.scope": "local_crop",
        "protocol.paper_experiment": "Fig. 9",
        "protocol.train_background_name": "20210106155330_01_staring.mat",
        "protocol.test_background_name": "20210106155432_01_staring.mat",
        "protocol.train_scr_db": TRAIN_SCR_VALUES,
        "protocol.test_scr_db": TEST_SCR_VALUES,
        "protocol.pulses": expected_pulses,
        "protocol.range_cells": expected_range_cells,
        "protocol.reference_cells": 20,
        "protocol.scr_reference_power": "sum_of_full_profile_per_cell_mean_power",
        "protocol.target_injection_order": "full_profile_before_non_overlapping_segmentation",
        "protocol.train_targets_per_scr": 5,
        "protocol.test_target_cell_one_based": 2083,
        "protocol.test_speed_mps": 0.4,
        "protocol.pulse_window": "non_overlapping_step_equals_P",
        "protocol.label_rule": "only_true_injected_range_cells_are_positive",
        "protocol.normalization": "none",
        "crop.paper_target_cell_one_based": 2083,
        "crop.paper_target_index_zero_based": 2082,
    }
    for dotted_path, expected in expected_values.items():
        actual = _manifest_value(manifest, dotted_path)
        if actual != expected:
            raise ValueError(
                f"SDRDSP v2 manifest 不匹配: {dotted_path}={actual!r}, expected={expected!r}。"
            )

    if protocol_id == SDRDSP_PHASE_RCS_PROTOCOL:
        realistic_values = {
            "protocol.target_model": "phase_noise_swerling1_window",
            "protocol.phase_noise_model": "within_window_random_walk",
            "protocol.phase_noise_std_deg": 10.0,
            "protocol.random_initial_phase": "per_4_pulse_window",
            "protocol.rcs_model": "swerling1_window",
            "protocol.rcs_power_distribution": "exponential_mean_1",
            "protocol.rcs_mean_power_normalized": True,
        }
        for dotted_path, expected in realistic_values.items():
            actual = _manifest_value(manifest, dotted_path)
            if actual != expected:
                raise ValueError(
                    f"SDRDSP realistic manifest 不匹配: {dotted_path}={actual!r}, expected={expected!r}。"
                )
        audits = list(_manifest_value(manifest, "audit.train_injection_by_scr").values())
        audits += list(_manifest_value(manifest, "audit.test_injection_by_scr").values())
        for audit in audits:
            for record in audit["target_records"]:
                if abs(float(record["rcs_mean_power_gain"]) - 1.0) > 1e-6:
                    raise ValueError("真实化目标的窗级 RCS 平均功率增益未归一化到 1。")

    if expected_range_cells != 256:
        raise ValueError(
            f"{SDRDSP_V2_PROTOCOL} 明确是 N=256 local-crop 协议，实际 model.range_cells={expected_range_cells}。"
        )
    crop_start = int(_manifest_value(manifest, "crop.crop_start_zero_based"))
    crop_end = int(_manifest_value(manifest, "crop.crop_end_exclusive_zero_based"))
    local_target = int(_manifest_value(manifest, "crop.local_target_index_zero_based"))
    if crop_end - crop_start != expected_range_cells or crop_start + local_target != 2082:
        raise ValueError(f"SDRDSP v2 裁剪窗口或目标索引不一致: crop={crop_start}:{crop_end}, local={local_target}。")

    reference_cells = int(_manifest_value(manifest, "protocol.reference_cells"))
    min_target_gap = int(_manifest_value(manifest, "protocol.min_target_gap"))
    if min_target_gap < reference_cells + 1:
        raise ValueError(
            f"训练目标间隔不足: min_target_gap={min_target_gap}, 至少应为 {reference_cells + 1}。"
        )
    if int(_manifest_value(manifest, "audit.min_train_target_gap")) < min_target_gap:
        raise ValueError("manifest 显示实际训练目标间隔小于协议要求。")
    if float(_manifest_value(manifest, "audit.max_abs_injected_scr_error_db")) > 1e-6:
        raise ValueError("manifest 显示注入 SCR 未通过 1e-6 dB 精度验收。")

    train_positive = _manifest_value(manifest, "audit.train_positive_bins_per_window")
    if any(float(train_positive[key]) != 5.0 for key in ("min", "mean", "max")):
        raise ValueError(f"训练标签正单元数量不符合五目标协议: {train_positive}。")
    test_positive = _manifest_value(manifest, "audit.test_positive_bins_per_window")
    if set(test_positive) != {str(value) for value in TEST_SCR_VALUES}:
        raise ValueError("测试标签审计没有覆盖全部 20 个 SCR。")
    for scr, summary in test_positive.items():
        if any(float(summary[key]) != 1.0 for key in ("min", "mean", "max")):
            raise ValueError(f"SCR={scr} 的测试标签不是每窗口一个目标单元: {summary}。")

    expected_test_names = {f"test_scr_{scr}.npz" for scr in TEST_SCR_VALUES}
    actual_test_names = {path.name for path in list_test_scr_files(data_dir)}
    if actual_test_names != expected_test_names:
        raise ValueError(
            f"SDRDSP v2 测试文件集合不完整: missing={sorted(expected_test_names - actual_test_names)}, "
            f"extra={sorted(actual_test_names - expected_test_names)}。"
        )
    train_path = data_dir / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"缺少 SDRDSP v2 训练文件: {train_path}")
    with np.load(train_path) as train_data:
        if "scr" not in train_data:
            raise ValueError(f"严格 SDRDSP v2 训练文件缺少 scr 字段: {train_path}。")
        actual_train_scr = sorted(int(value) for value in np.unique(train_data["scr"]))
    if actual_train_scr != TRAIN_SCR_VALUES:
        raise ValueError(f"训练 SCR 集合不匹配: actual={actual_train_scr}, expected={TRAIN_SCR_VALUES}。")
    for path in list_test_scr_files(data_dir):
        with np.load(path) as test_data:
            if "scr" not in test_data:
                raise ValueError(f"严格 SDRDSP v2 测试文件缺少 scr 字段: {path}。")
            values = np.unique(test_data["scr"])
        expected_scr = _parse_scr(path)
        if values.tolist() != [expected_scr]:
            raise ValueError(f"{path} 的 scr 字段不匹配: {values.tolist()}。")
    return manifest


def _manifest_value(manifest: dict[str, Any], dotted_path: str) -> Any:
    value: Any = manifest
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"SDRDSP v2 manifest 缺少字段: {dotted_path}。")
        value = value[key]
    return value


def load_scr_arrays(
    data_dir: Path,
    split: str,
    scr: int | None = None,
    max_windows: int | None = None,
    rng: np.random.Generator | None = None,
    require_scr_metadata: bool = False,
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
            if "scr" in data:
                scr_values = np.asarray(data["scr"], dtype=np.int64).reshape(-1)
                if scr_values.shape != (len(x),):
                    raise ValueError(f"{path} 的 scr shape 应为 [{len(x)}]，实际为 {scr_values.shape}。")
            elif require_scr_metadata:
                raise ValueError(f"严格 SDRDSP v2 数据缺少 scr 字段: {path}。")
            else:
                value = -9999 if split == "train" else _parse_scr(path)
                scr_values = np.full((len(x),), value, dtype=np.int64)
            if split == "test" and np.any(scr_values != _parse_scr(path)):
                raise ValueError(f"{path} 的 scr 字段与文件名不一致。")

            if remaining is not None and len(x) > remaining:
                if rng is None:
                    rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(len(x), size=remaining, replace=False))
                x = x[idx]
                y = y[idx]
                scr_values = scr_values[idx]
            x_parts.append(x.astype(np.float32, copy=False))
            y_parts.append(y.astype(np.int64, copy=False))
            scr_parts.append(scr_values)
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
        normalization: str = "train_standardize_clip",
        protocol: str | None = None,
        expected_pulses: int | None = None,
        expected_range_cells: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.scr = scr
        self.rng = np.random.default_rng(seed)
        self.normalization = normalization
        self.protocol = protocol
        if protocol is not None:
            if protocol not in SDRDSP_LOCAL_CROP_PROTOCOLS:
                raise ValueError(f"未知 SCR protocol: {protocol}")
            if normalization != "none":
                raise ValueError(f"{SDRDSP_V2_PROTOCOL} 必须使用 normalization=none。")
            validate_sdrdsp_v2_manifest(self.data_dir, expected_pulses, expected_range_cells)

        x, y, scr_values = load_scr_arrays(
            self.data_dir,
            split,
            scr=scr,
            max_windows=max_windows,
            rng=self.rng,
            require_scr_metadata=protocol in SDRDSP_LOCAL_CROP_PROTOCOLS,
        )
        if expected_pulses is not None and x.shape[2] != expected_pulses:
            raise ValueError(f"SCR 数据 pulses={x.shape[2]} 与 model.pulses={expected_pulses} 不一致。")
        if expected_range_cells is not None and x.shape[3] != expected_range_cells:
            raise ValueError(
                f"SCR 数据 range_cells={x.shape[3]} 与 model.range_cells={expected_range_cells} 不一致。"
            )
        if protocol in SDRDSP_LOCAL_CROP_PROTOCOLS:
            expected_positive = 5 if split == "train" else 1
            positive_counts = y.sum(axis=1)
            if np.any(positive_counts != expected_positive):
                raise ValueError(
                    f"严格 SDRDSP v2 {split} 标签应每窗口包含 {expected_positive} 个目标单元，"
                    f"实际范围为 [{positive_counts.min()}, {positive_counts.max()}]。"
                )
        if normalization == "none":
            if norm is not None:
                raise ValueError("normalization=none 时不应传入 norm。")
            self.norm = None
            real = x[:, 0]
            imag = x[:, 1]
        elif normalization == "train_standardize_clip":
            self.norm = norm if norm is not None else compute_train_norm(self.data_dir)
            nr_mean, nr_std, ni_mean, ni_std = self.norm
            real = np.clip((x[:, 0] - nr_mean) / (nr_std + 1e-8), -5, 5)
            imag = np.clip((x[:, 1] - ni_mean) / (ni_std + 1e-8), -5, 5)
        else:
            raise ValueError(f"未知 SCR normalization: {normalization}")

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
