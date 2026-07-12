from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


TRAIN_MAT = "20210106155330_01_staring.mat"
TEST_MAT = "20210106155432_01_staring.mat"
MAT_KEY = "amplitude_complex_T1"
PROTOCOL_ID = "sdrdsp_fig9_local_crop_v2"
REALISTIC_PROTOCOL_ID = "sdrdsp_fig9_local_crop_phase_rcs_v1"
IDEAL_TARGET_MODEL = "ideal_continuous_phase"
REALISTIC_TARGET_MODEL = "phase_noise_swerling1_window"
TRAIN_SCR_VALUES = list(range(-12, 15, 2))
TEST_SCR_VALUES = list(range(-24, 15, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按论文 Fig. 9 协议生成 SDRDSP SCR 数据。")
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/sdrdsp/raw"))
    parser.add_argument("--train-mat", type=Path, default=None)
    parser.add_argument("--test-mat", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sdrdsp_strict_256_v2"))
    parser.add_argument("--mat-key", type=str, default=MAT_KEY)
    parser.add_argument("--pulses", type=int, default=4)
    parser.add_argument("--range-cells", type=int, default=256)
    parser.add_argument("--reference-cells", type=int, default=20)
    parser.add_argument("--paper-target-cell", type=int, default=2083, help="论文中的一基距离单元编号。")
    parser.add_argument("--crop-start", type=int, default=None)
    parser.add_argument("--train-targets-per-scr", type=int, default=5)
    parser.add_argument("--min-target-gap", type=int, default=21, help="训练目标之间允许的最小索引间隔。")
    parser.add_argument("--train-speed-min", type=float, default=0.1)
    parser.add_argument("--train-speed-max", type=float, default=0.5)
    parser.add_argument("--test-speed", type=float, default=0.4)
    parser.add_argument("--prt", type=float, default=1.0 / 1600.0)
    parser.add_argument("--wavelength", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-model",
        choices=[IDEAL_TARGET_MODEL, REALISTIC_TARGET_MODEL],
        default=IDEAL_TARGET_MODEL,
    )
    parser.add_argument("--phase-noise-std-deg", type=float, default=10.0)
    parser.add_argument("--max-train-windows-per-scr", type=int, default=None)
    parser.add_argument("--max-test-windows-per-scr", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_mat = args.train_mat or args.raw_dir / TRAIN_MAT
    test_mat = args.test_mat or args.raw_dir / TEST_MAT
    paper_target_index = paper_cell_to_zero_based(args.paper_target_cell)
    crop_start = args.crop_start
    if crop_start is None:
        crop_start = paper_target_index - args.range_cells // 2
    crop_end = crop_start + args.range_cells
    local_target = paper_target_index - crop_start
    if not 0 <= local_target < args.range_cells:
        raise SystemExit(
            f"目标单元不在裁剪窗口内: one_based={args.paper_target_cell}, "
            f"zero_based={paper_target_index}, crop=[{crop_start},{crop_end})"
        )
    if args.min_target_gap < 1:
        raise ValueError(f"min_target_gap 必须为正整数，实际为 {args.min_target_gap}。")
    if args.train_targets_per_scr < 1:
        raise ValueError(f"train_targets_per_scr 必须为正整数，实际为 {args.train_targets_per_scr}。")

    train_clutter = load_clutter(train_mat, args.mat_key, crop_start, crop_end)
    test_clutter = load_clutter(test_mat, args.mat_key, crop_start, crop_end)
    rng = np.random.default_rng(args.seed)
    protocol_id = PROTOCOL_ID if args.target_model == IDEAL_TARGET_MODEL else REALISTIC_PROTOCOL_ID

    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    train_scr_parts: list[np.ndarray] = []
    train_targets: dict[str, list[int]] = {}
    train_speeds: dict[str, list[float]] = {}
    train_injection_audits: dict[str, dict[str, Any]] = {}

    for scr in TRAIN_SCR_VALUES:
        target_local = choose_train_targets(
            rng,
            args.range_cells,
            args.train_targets_per_scr,
            args.reference_cells,
            args.min_target_gap,
        )
        target_speeds = rng.uniform(args.train_speed_min, args.train_speed_max, size=len(target_local))
        train_targets[str(scr)] = [int(crop_start + pos + 1) for pos in target_local]
        train_speeds[str(scr)] = [float(speed) for speed in target_speeds]
        x_scr, y_scr, injection_audit = build_scr_samples(
            clutter=train_clutter,
            target_positions=target_local,
            target_speeds=target_speeds,
            scr_db=scr,
            pulses=args.pulses,
            reference_cells=args.reference_cells,
            prt=args.prt,
            wavelength=args.wavelength,
            max_windows=args.max_train_windows_per_scr,
            target_model=args.target_model,
            target_rng=np.random.default_rng(args.seed + 100_000 + scr),
            phase_noise_std_deg=args.phase_noise_std_deg,
        )
        train_x_parts.append(x_scr)
        train_y_parts.append(y_scr)
        train_scr_parts.append(np.full((len(x_scr),), scr, dtype=np.int16))
        train_injection_audits[str(scr)] = injection_audit

    x_train = np.concatenate(train_x_parts, axis=0)
    y_train = np.concatenate(train_y_parts, axis=0)
    scr_train = np.concatenate(train_scr_parts, axis=0)

    test_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    test_injection_audits: dict[str, dict[str, Any]] = {}
    for scr in TEST_SCR_VALUES:
        x_scr, y_scr, injection_audit = build_scr_samples(
            clutter=test_clutter,
            target_positions=[local_target],
            target_speeds=[args.test_speed],
            scr_db=scr,
            pulses=args.pulses,
            reference_cells=args.reference_cells,
            prt=args.prt,
            wavelength=args.wavelength,
            max_windows=args.max_test_windows_per_scr,
            target_model=args.target_model,
            target_rng=np.random.default_rng(args.seed + 200_000 + scr),
            phase_noise_std_deg=args.phase_noise_std_deg,
        )
        test_sets[scr] = (x_scr, y_scr)
        test_injection_audits[str(scr)] = injection_audit

    manifest = build_manifest(
        args=args,
        train_mat=train_mat,
        test_mat=test_mat,
        crop_start=crop_start,
        crop_end=crop_end,
        paper_target_index=paper_target_index,
        local_target=local_target,
        train_clutter_shape=train_clutter.shape,
        test_clutter_shape=test_clutter.shape,
        train_targets=train_targets,
        train_speeds=train_speeds,
        train_injection_audits=train_injection_audits,
        test_injection_audits=test_injection_audits,
        x_train=x_train,
        y_train=y_train,
        test_sets=test_sets,
        protocol_id=protocol_id,
    )

    print_audit(manifest)
    if args.dry_run:
        print("dry-run: 未写入任何文件。")
        return

    ensure_output_dir(args.output_dir, args.overwrite)
    np.savez_compressed(args.output_dir / "train.npz", X=x_train, y=y_train, scr=scr_train)
    for scr, (x_scr, y_scr) in test_sets.items():
        np.savez_compressed(
            args.output_dir / f"test_scr_{scr}.npz",
            X=x_scr,
            y=y_scr,
            scr=np.full((len(x_scr),), scr, dtype=np.int16),
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"已写入 SDRDSP strict 数据: {args.output_dir}")


def load_clutter(path: Path, mat_key: str, crop_start: int, crop_end: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"缺少 SDRDSP 原始文件: {path}")
    try:
        payload = sio.loadmat(path)
    except NotImplementedError as exc:
        raise RuntimeError(f"{path} 可能是 MATLAB v7.3 文件，当前环境缺少可用读取器。") from exc
    if mat_key not in payload:
        keys = sorted(k for k in payload if not k.startswith("__"))
        raise KeyError(f"{path} 中找不到 key={mat_key!r}，可用 key={keys}")
    clutter = np.asarray(payload[mat_key])
    if clutter.ndim != 2:
        raise ValueError(f"{path}:{mat_key} 应为二维 [pulse, range]，实际 shape={clutter.shape}")
    if crop_start < 0 or crop_end > clutter.shape[1]:
        raise ValueError(f"裁剪窗口 [{crop_start},{crop_end}) 超出 {path} 的 range 维度 {clutter.shape[1]}")
    return clutter[:, crop_start:crop_end].astype(np.complex64, copy=False)


def paper_cell_to_zero_based(cell_one_based: int) -> int:
    """把论文使用的一基距离单元编号转换为 Python 零基索引。"""
    if cell_one_based < 1:
        raise ValueError(f"论文距离单元编号必须从 1 开始，实际为 {cell_one_based}。")
    return cell_one_based - 1


def choose_train_targets(
    rng: np.random.Generator,
    range_cells: int,
    count: int,
    reference_cells: int,
    min_gap: int,
) -> list[int]:
    if range_cells <= reference_cells:
        raise ValueError(
            f"距离单元数必须大于参考单元数: range_cells={range_cells}, reference_cells={reference_cells}。"
        )
    available = np.arange(range_cells, dtype=np.int64)
    selected: list[int] = []
    for _ in range(count):
        if available.size == 0:
            raise ValueError(
                f"无法选择满足最小间隔的训练目标: range_cells={range_cells}, "
                f"count={count}, min_gap={min_gap}。"
            )
        pos = int(rng.choice(available))
        selected.append(pos)
        available = available[np.abs(available - pos) >= min_gap]

    selected.sort()
    excluded = set(selected)
    for pos in selected:
        refs = local_reference_bins(pos, range_cells, excluded, reference_cells)
        if len(refs) != reference_cells:
            raise ValueError(f"目标单元 {pos} 无法获得 {reference_cells} 个无目标参考单元。")
    return selected


def build_scr_samples(
    clutter: np.ndarray,
    target_positions: list[int],
    target_speeds: list[float] | np.ndarray,
    scr_db: int,
    pulses: int,
    reference_cells: int,
    prt: float,
    wavelength: float,
    max_windows: int | None,
    target_model: str = IDEAL_TARGET_MODEL,
    target_rng: np.random.Generator | None = None,
    phase_noise_std_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """先向完整距离像序列注入连续目标，再按脉冲维非重叠分段。"""
    if clutter.ndim != 2:
        raise ValueError(f"clutter 应为 [pulse, range]，实际为 {clutter.shape}。")
    if len(target_positions) != len(target_speeds):
        raise ValueError(
            f"目标位置数与速度数不一致: positions={len(target_positions)}, speeds={len(target_speeds)}。"
        )
    if not target_positions:
        raise ValueError("至少需要一个目标位置。")
    if pulses < 1:
        raise ValueError(f"pulses 必须为正整数，实际为 {pulses}。")
    if prt <= 0 or wavelength <= 0:
        raise ValueError(f"PRT 和波长必须为正数: prt={prt}, wavelength={wavelength}。")
    if target_model not in {IDEAL_TARGET_MODEL, REALISTIC_TARGET_MODEL}:
        raise ValueError(f"未知 target_model: {target_model!r}。")
    if target_model == REALISTIC_TARGET_MODEL and target_rng is None:
        raise ValueError("真实化目标模型需要显式 target_rng，避免不可复现。")
    if not np.isfinite(phase_noise_std_deg) or phase_noise_std_deg < 0:
        raise ValueError(f"phase_noise_std_deg 必须为非负有限数，实际为 {phase_noise_std_deg}。")

    starts = list(range(0, clutter.shape[0] - pulses + 1, pulses))
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError("没有可用 pulse 窗口。")

    excluded = set(target_positions)
    if len(excluded) != len(target_positions):
        raise ValueError(f"目标位置存在重复值: {target_positions}。")
    for pos in target_positions:
        if not 0 <= pos < clutter.shape[1]:
            raise ValueError(f"目标位置越界: pos={pos}, range_cells={clutter.shape[1]}。")
    ref_by_pos = {
        pos: local_reference_bins(pos, clutter.shape[1], excluded, reference_cells)
        for pos in target_positions
    }
    for pos, refs in ref_by_pos.items():
        if len(refs) != reference_cells:
            raise ValueError(f"目标单元 {pos} 的局部参考单元不足: {len(refs)} != {reference_cells}")

    injected = clutter.copy()
    pulse_indices = np.arange(clutter.shape[0], dtype=np.float64)
    target_records: list[dict[str, Any]] = []
    for pos, speed in zip(target_positions, target_speeds):
        refs = ref_by_pos[pos]
        per_cell_power = np.mean(np.abs(clutter[:, refs]) ** 2, axis=0, dtype=np.float64)
        reference_power_sum = float(np.sum(per_cell_power, dtype=np.float64))
        if not np.isfinite(reference_power_sum) or reference_power_sum <= 0:
            raise ValueError(f"目标单元 {pos} 的参考杂波功率无效: {reference_power_sum}。")
        target_amplitude = float(np.sqrt(reference_power_sum * (10.0 ** (scr_db / 10.0))))
        if target_model == IDEAL_TARGET_MODEL:
            phase = 4.0 * np.pi * float(speed) * prt * pulse_indices / wavelength
            target_signal = target_amplitude * np.exp(1j * phase)
            rcs_mean_power_gain = 1.0
        else:
            assert target_rng is not None
            target_signal, rcs_mean_power_gain = build_realistic_target_signal(
                num_pulses=clutter.shape[0],
                starts=starts,
                pulses=pulses,
                target_amplitude=target_amplitude,
                speed=float(speed),
                prt=prt,
                wavelength=wavelength,
                phase_noise_std_deg=phase_noise_std_deg,
                rng=target_rng,
            )
        injected[:, pos] += target_signal.astype(injected.dtype, copy=False)
        achieved_scr = 10.0 * np.log10((target_amplitude**2) / reference_power_sum)
        target_records.append(
            {
                "target_position_local_zero_based": int(pos),
                "reference_cells_local_zero_based": [int(ref) for ref in refs],
                "speed_mps": float(speed),
                "reference_power_sum": reference_power_sum,
                "target_amplitude": target_amplitude,
                "achieved_injected_scr_db": float(achieved_scr),
                "abs_scr_error_db": float(abs(achieved_scr - scr_db)),
                "rcs_mean_power_gain": float(rcs_mean_power_gain),
            }
        )

    labels = np.zeros(clutter.shape[1], dtype=np.int32)
    labels[target_positions] = 1
    x_parts = [
        np.stack(
            [injected[start : start + pulses].real, injected[start : start + pulses].imag],
            axis=0,
        ).astype(np.float32)
        for start in starts
    ]
    y = np.repeat(labels[None, :], len(starts), axis=0)
    audit = {
        "target_records": target_records,
        "num_windows": len(starts),
        "max_abs_scr_error_db": max(record["abs_scr_error_db"] for record in target_records),
        "target_model": target_model,
        "phase_progression": (
            "continuous_global_pulse_index"
            if target_model == IDEAL_TARGET_MODEL
            else "per_window_random_initial_phase_plus_doppler_plus_random_walk"
        ),
        "phase_noise_std_deg": float(phase_noise_std_deg) if target_model == REALISTIC_TARGET_MODEL else 0.0,
    }
    return np.stack(x_parts, axis=0), y, audit


def build_realistic_target_signal(
    num_pulses: int,
    starts: list[int],
    pulses: int,
    target_amplitude: float,
    speed: float,
    prt: float,
    wavelength: float,
    phase_noise_std_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """生成窗级 Swerling-I 起伏与窗内相位随机游走目标；平均注入功率保持不变。"""
    gains = rng.exponential(scale=1.0, size=len(starts)).astype(np.float64)
    mean_gain = float(gains.mean())
    if not np.isfinite(mean_gain) or mean_gain <= 0:
        raise ValueError(f"RCS 功率增益均值无效: {mean_gain}。")
    gains /= mean_gain
    omega = 4.0 * np.pi * speed * prt / wavelength
    noise_std = np.deg2rad(phase_noise_std_deg)
    signal = np.zeros(num_pulses, dtype=np.complex128)
    for window_idx, start in enumerate(starts):
        initial_phase = rng.uniform(-np.pi, np.pi)
        increments = rng.normal(0.0, noise_std, size=pulses)
        random_walk = np.cumsum(increments)
        phase = initial_phase + omega * np.arange(pulses, dtype=np.float64) + random_walk
        signal[start : start + pulses] = target_amplitude * np.sqrt(gains[window_idx]) * np.exp(1j * phase)
    return signal, float(gains.mean())


def local_reference_bins(pos: int, range_cells: int, excluded: set[int], count: int) -> list[int]:
    refs: list[int] = []
    radius = 1
    while len(refs) < count and (pos - radius >= 0 or pos + radius < range_cells):
        left = pos - radius
        right = pos + radius
        if left >= 0 and left not in excluded:
            refs.append(left)
        if len(refs) >= count:
            break
        if right < range_cells and right not in excluded:
            refs.append(right)
        radius += 1
    return refs


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.npz")) + list(output_dir.glob("manifest.json"))
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:5])
        raise FileExistsError(f"{output_dir} 已存在数据文件: {names}；如需覆盖请显式加 --overwrite。")


def build_manifest(
    args: argparse.Namespace,
    train_mat: Path,
    test_mat: Path,
    crop_start: int,
    crop_end: int,
    paper_target_index: int,
    local_target: int,
    train_clutter_shape: tuple[int, int],
    test_clutter_shape: tuple[int, int],
    train_targets: dict[str, list[int]],
    train_speeds: dict[str, list[float]],
    train_injection_audits: dict[str, dict[str, Any]],
    test_injection_audits: dict[str, dict[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    test_sets: dict[int, tuple[np.ndarray, np.ndarray]],
    protocol_id: str,
) -> dict[str, Any]:
    max_scr_error = max(
        [audit["max_abs_scr_error_db"] for audit in train_injection_audits.values()]
        + [audit["max_abs_scr_error_db"] for audit in test_injection_audits.values()]
    )
    min_train_gap = min(
        min(np.diff(sorted(cells))) for cells in train_targets.values() if len(cells) > 1
    )
    return {
        "dataset": (
            "SDRDSP Fig. 9 local-crop reproduction"
            if protocol_id == PROTOCOL_ID
            else "SDRDSP Fig. 9 local-crop target-model sensitivity"
        ),
        "source_files": {
            "train_background": str(train_mat),
            "test_background": str(test_mat),
            "train_sha256": file_sha256(train_mat),
            "test_sha256": file_sha256(test_mat),
            "mat_key": args.mat_key,
        },
        "protocol": {
            "id": protocol_id,
            "scope": "local_crop",
            "paper_experiment": "Fig. 9",
            "train_background_name": TRAIN_MAT,
            "test_background_name": TEST_MAT,
            "train_scr_db": TRAIN_SCR_VALUES,
            "test_scr_db": TEST_SCR_VALUES,
            "pulses": args.pulses,
            "range_cells": args.range_cells,
            "reference_cells": args.reference_cells,
            "scr_reference_power": "sum_of_full_profile_per_cell_mean_power",
            "target_injection_order": "full_profile_before_non_overlapping_segmentation",
            "train_targets_per_scr": args.train_targets_per_scr,
            "min_target_gap": args.min_target_gap,
            "train_target_cells_one_based_by_scr": train_targets,
            "train_target_speeds_mps_by_scr": train_speeds,
            "train_speed_mps": [args.train_speed_min, args.train_speed_max],
            "test_target_cell_one_based": args.paper_target_cell,
            "test_speed_mps": args.test_speed,
            "prt_seconds": args.prt,
            "wavelength_m": args.wavelength,
            "pulse_window": "non_overlapping_step_equals_P",
            "label_rule": "only_true_injected_range_cells_are_positive",
            "normalization": "none",
            "target_model": args.target_model,
            "phase_noise_model": (
                "none" if args.target_model == IDEAL_TARGET_MODEL else "within_window_random_walk"
            ),
            "phase_noise_std_deg": (
                0.0 if args.target_model == IDEAL_TARGET_MODEL else args.phase_noise_std_deg
            ),
            "random_initial_phase": (
                "none" if args.target_model == IDEAL_TARGET_MODEL else "per_4_pulse_window"
            ),
            "rcs_model": "none" if args.target_model == IDEAL_TARGET_MODEL else "swerling1_window",
            "rcs_power_distribution": (
                "constant_1" if args.target_model == IDEAL_TARGET_MODEL else "exponential_mean_1"
            ),
            "rcs_mean_power_normalized": True,
        },
        "crop": {
            "paper_target_cell_one_based": args.paper_target_cell,
            "paper_target_index_zero_based": paper_target_index,
            "crop_start_zero_based": crop_start,
            "crop_end_exclusive_zero_based": crop_end,
            "local_target_index_zero_based": local_target,
        },
        "source_shapes_after_crop": {
            "train": list(train_clutter_shape),
            "test": list(test_clutter_shape),
        },
        "outputs": {
            "train_npz": {"X": list(x_train.shape), "y": list(y_train.shape)},
            "test_npz": {
                str(scr): {"X": list(x.shape), "y": list(y.shape)}
                for scr, (x, y) in test_sets.items()
            },
        },
        "audit": {
            "train_positive_bins_per_window": summarize_positive_counts(y_train),
            "test_positive_bins_per_window": {
                str(scr): summarize_positive_counts(y) for scr, (_, y) in test_sets.items()
            },
            "min_train_target_gap": int(min_train_gap),
            "max_abs_injected_scr_error_db": float(max_scr_error),
            "train_injection_by_scr": train_injection_audits,
            "test_injection_by_scr": test_injection_audits,
            "train_x_mean": float(x_train.mean()),
            "train_x_std": float(x_train.std()),
            "test_x_std_by_scr": {str(scr): float(x.std()) for scr, (x, _) in test_sets.items()},
        },
        "seed": args.seed,
    }


def summarize_positive_counts(y: np.ndarray) -> dict[str, float]:
    counts = y.sum(axis=1)
    return {
        "min": float(counts.min()),
        "mean": float(counts.mean()),
        "max": float(counts.max()),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_audit(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    audit = manifest["audit"]
    print("SDRDSP strict preprocess audit")
    print(f"  train X/y: {outputs['train_npz']['X']} / {outputs['train_npz']['y']}")
    print(f"  train positives/window: {audit['train_positive_bins_per_window']}")
    first_scr = sorted(outputs["test_npz"], key=lambda item: int(item))[0]
    print(f"  test SCR files: {len(outputs['test_npz'])}; first SCR {first_scr}: {outputs['test_npz'][first_scr]}")
    print(f"  test positives/window: {audit['test_positive_bins_per_window'][first_scr]}")
    print(f"  min train target gap: {audit['min_train_target_gap']}")
    print(f"  max injected SCR error: {audit['max_abs_injected_scr_error_db']:.3e} dB")
    print(f"  crop: {manifest['crop']}")


if __name__ == "__main__":
    main()
