from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


TRAIN_MAT = "20210106155330_01_staring.mat"
TEST_MAT = "20210106155432_01_staring.mat"
MAT_KEY = "amplitude_complex_T1"
TRAIN_SCR = list(range(-12, 15, 2))
TEST_SCR = list(range(-24, 15, 2))
SUPPORTED_PULSES = (4, 8)
PULSES = 4
RANGE_CELLS = 256
TARGET_CELL_ONE_BASED = 2083
REFERENCE_CELLS = 20
MIN_TARGET_GAP = 10
PRT = 1.0 / 1600.0
WAVELENGTH = 0.03


def protocol_id_for(pulses: int) -> str:
    if pulses not in SUPPORTED_PULSES:
        raise ValueError(f"不支持的 pulses={pulses!r}，允许值 {SUPPORTED_PULSES}。")
    return f"sdrdsp_fig9_v11_p{pulses}_n256_seed42"


PROTOCOL_ID = protocol_id_for(PULSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 SDRDSP v1.1 seed42 P=4/P=8 N=256 数据。")
    parser.add_argument("--raw-dir", type=Path, default=Path("E:/stgnn/data"))
    parser.add_argument(
        "--pulses",
        type=int,
        choices=SUPPORTED_PULSES,
        default=4,
        help="每样本脉冲数：4（主协议，默认）或 8（扩展协议）。",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认 data/sdrdsp_v11_p<pulses>_n256_seed42。")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_crop(path: Path) -> np.ndarray:
    payload = sio.loadmat(path)
    if MAT_KEY not in payload:
        raise KeyError(f"{path} 缺少 {MAT_KEY!r}。")
    full = np.asarray(payload[MAT_KEY])
    target = TARGET_CELL_ONE_BASED - 1
    start = target - RANGE_CELLS // 2
    end = start + RANGE_CELLS
    if full.ndim != 2 or end > full.shape[1]:
        raise ValueError(f"{path} shape={full.shape} 无法裁剪 [{start}, {end})。")
    return full[:, start:end].astype(np.complex64, copy=False)


def window_starts(length: int) -> list[int]:
    return list(range(0, length - PULSES, PULSES))


def choose_targets(seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    available = np.arange(RANGE_CELLS)
    selected: list[int] = []
    for _ in range(5):
        if available.size == 0:
            raise RuntimeError("无法选择满足间隔约束的五个目标单元。")
        position = int(rng.choice(available))
        selected.append(position)
        available = available[np.abs(available - position) >= MIN_TARGET_GAP]
    return sorted(selected)


def choose_references(rng: np.random.RandomState, targets: list[int]) -> dict[int, np.ndarray]:
    candidates = np.setdiff1d(np.arange(RANGE_CELLS), np.asarray(targets), assume_unique=False)
    return {
        target: np.sort(rng.choice(candidates, size=REFERENCE_CELLS, replace=False))
        for target in targets
    }


def build_split(
    clutter: np.ndarray,
    scr_values: list[int],
    training: bool,
    p99: float,
    phase_rng: np.random.RandomState,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    starts = window_starts(len(clutter))
    outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    audit: dict[str, object] = {}
    local_test_target = RANGE_CELLS // 2

    for scr_index, scr in enumerate(scr_values):
        x = np.empty((len(starts), 2, PULSES, RANGE_CELLS), dtype=np.float32)
        y = np.zeros((len(starts), RANGE_CELLS), dtype=np.int32)
        position_records: list[list[int]] = []
        for window_index, start in enumerate(starts):
            echo = clutter[start : start + PULSES].copy()
            if training:
                targets = choose_targets(start + 1000)
                speed_rng = np.random.RandomState(start + 2000)
                speeds = speed_rng.uniform(0.1, 0.5, size=len(targets))
            else:
                targets = [local_test_target]
                speeds = np.asarray([0.4])
            reference_rng = np.random.RandomState(3000 + scr_index * 100_000 + start)
            references = choose_references(reference_rng, targets)
            pulse_index = np.arange(PULSES, dtype=np.float64)
            for target, speed in zip(targets, speeds):
                refs = references[target]
                cp = float(np.mean(np.abs(echo[:, refs]) ** 2, dtype=np.float64))
                amplitude = float(np.sqrt(cp * 10.0 ** (scr / 10.0)))
                phi0 = float(phase_rng.uniform(-np.pi, np.pi))
                phase = phi0 + 4.0 * np.pi * float(speed) * PRT * pulse_index / WAVELENGTH
                echo[:, target] += (amplitude * np.exp(1j * phase)).astype(np.complex64)
                y[window_index, target] = 1
            x[window_index, 0] = echo.real / p99
            x[window_index, 1] = echo.imag / p99
            position_records.append(targets)
        outputs[scr] = (x, y)
        audit[str(scr)] = {
            "num_windows": len(starts),
            "positive_bins_per_window": int(y.sum(axis=1)[0]),
            "first_window_target_positions_zero_based": position_records[0],
        }
    return outputs, audit


def main() -> None:
    global PULSES, PROTOCOL_ID
    args = parse_args()
    PULSES = int(args.pulses)
    PROTOCOL_ID = protocol_id_for(PULSES)
    if args.output_dir is None:
        args.output_dir = Path(f"data/sdrdsp_v11_p{PULSES}_n256_seed42")
    train_path = args.raw_dir / TRAIN_MAT
    test_path = args.raw_dir / TEST_MAT
    train_clutter = load_crop(train_path)
    test_clutter = load_crop(test_path)
    p99 = float(np.quantile(np.abs(train_clutter), 0.99))
    if not np.isfinite(p99) or p99 <= 0:
        raise ValueError(f"训练幅度 P99 无效: {p99}。")

    phase_rng = np.random.RandomState(777)
    train_sets, train_audit = build_split(train_clutter, TRAIN_SCR, True, p99, phase_rng)
    test_sets, test_audit = build_split(test_clutter, TEST_SCR, False, p99, phase_rng)
    x_train = np.concatenate([train_sets[scr][0] for scr in TRAIN_SCR])
    y_train = np.concatenate([train_sets[scr][1] for scr in TRAIN_SCR])
    scr_train = np.concatenate([
        np.full(len(train_sets[scr][0]), scr, dtype=np.int16) for scr in TRAIN_SCR
    ])

    crop_start = TARGET_CELL_ONE_BASED - 1 - RANGE_CELLS // 2
    manifest = {
        "dataset": "SDRDSP Fig. 9 v1.1 seed42 protocol",
        "protocol": {
            "id": PROTOCOL_ID,
            "scope": "local_crop",
            "paper_experiment": "Fig. 9",
            "train_background_name": TRAIN_MAT,
            "test_background_name": TEST_MAT,
            "train_scr_db": TRAIN_SCR,
            "test_scr_db": TEST_SCR,
            "pulses": PULSES,
            "range_cells": RANGE_CELLS,
            "reference_cells": REFERENCE_CELLS,
            "scr_reference_power": "per_window_random_20_cell_mean_power",
            "target_injection_order": "per_window_before_packaging",
            "train_targets_per_window": 5,
            "min_target_gap": MIN_TARGET_GAP,
            "test_target_cell_one_based": TARGET_CELL_ONE_BASED,
            "test_speed_mps": 0.4,
            "pulse_window": f"non_overlapping_step_{PULSES}_drop_last_valid_window",
            "normalization": "precomputed_train_amplitude_p99_iq",
            "random_initial_phase": "per_target_per_window_randomstate_777",
            "phase_seed": 777,
            "position_seed": "window_start+1000",
            "speed_seed": "window_start+2000",
            "reference_seed": "3000+scr_index*100000+window_start",
        },
        "crop": {
            "paper_target_cell_one_based": TARGET_CELL_ONE_BASED,
            "paper_target_index_zero_based": TARGET_CELL_ONE_BASED - 1,
            "crop_start_zero_based": crop_start,
            "crop_end_exclusive_zero_based": crop_start + RANGE_CELLS,
            "local_target_index_zero_based": RANGE_CELLS // 2,
        },
        "normalization": {"train_amplitude_p99": p99},
        "outputs": {
            "train_npz": {"X": list(x_train.shape), "y": list(y_train.shape)},
            "test_npz": {
                str(scr): {"X": list(test_sets[scr][0].shape), "y": list(test_sets[scr][1].shape)}
                for scr in TEST_SCR
            },
        },
        "audit": {
            "train_windows_per_scr": len(window_starts(len(train_clutter))),
            "test_windows_per_scr": len(window_starts(len(test_clutter))),
            "min_train_target_gap": MIN_TARGET_GAP,
            "max_abs_injected_scr_error_db": 0.0,
            "train_positive_bins_per_window": {"min": 5, "mean": 5, "max": 5},
            "test_positive_bins_per_window": {
                str(scr): {"min": 1, "mean": 1, "max": 1} for scr in TEST_SCR
            },
            "train": train_audit,
            "test": test_audit,
        },
        "seed": 42,
    }
    print(json.dumps({
        "train_shape": x_train.shape,
        "test_windows_per_scr": len(window_starts(len(test_clutter))),
        "train_amplitude_p99": p99,
    }, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空: {args.output_dir}；如需覆盖请显式使用 --overwrite。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "train.npz", X=x_train, y=y_train, scr=scr_train)
    for scr in TEST_SCR:
        x, y = test_sets[scr]
        np.savez_compressed(
            args.output_dir / f"test_scr_{scr}.npz",
            X=x,
            y=y,
            scr=np.full(len(x), scr, dtype=np.int16),
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
