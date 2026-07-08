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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict SDRDSP SCR npz files from raw .mat clutter.")
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/sdrdsp/raw"))
    parser.add_argument("--train-mat", type=Path, default=None)
    parser.add_argument("--test-mat", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sdrdsp_strict_256_v2"))
    parser.add_argument("--mat-key", type=str, default=MAT_KEY)
    parser.add_argument("--pulses", type=int, default=4)
    parser.add_argument("--range-cells", type=int, default=256)
    parser.add_argument("--reference-cells", type=int, default=20)
    parser.add_argument("--paper-target-cell", type=int, default=2083)
    parser.add_argument("--crop-start", type=int, default=None)
    parser.add_argument("--train-targets-per-scr", type=int, default=5)
    parser.add_argument("--train-speed-min", type=float, default=0.1)
    parser.add_argument("--train-speed-max", type=float, default=0.5)
    parser.add_argument("--test-speed", type=float, default=0.4)
    parser.add_argument("--prt", type=float, default=1.0 / 1600.0)
    parser.add_argument("--wavelength", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-windows-per-scr", type=int, default=None)
    parser.add_argument("--max-test-windows-per-scr", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_mat = args.train_mat or args.raw_dir / TRAIN_MAT
    test_mat = args.test_mat or args.raw_dir / TEST_MAT
    crop_start = args.crop_start
    if crop_start is None:
        crop_start = args.paper_target_cell - args.range_cells // 2
    crop_end = crop_start + args.range_cells
    local_target = args.paper_target_cell - crop_start
    if not 0 <= local_target < args.range_cells:
        raise SystemExit(
            f"目标单元不在裁剪窗口内: target={args.paper_target_cell}, crop=[{crop_start},{crop_end})"
        )

    train_clutter = load_clutter(train_mat, args.mat_key, crop_start, crop_end)
    test_clutter = load_clutter(test_mat, args.mat_key, crop_start, crop_end)
    rng = np.random.default_rng(args.seed)

    train_scr_values = list(range(-12, 15, 2))
    test_scr_values = list(range(-24, 15, 2))
    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    train_scr_parts: list[np.ndarray] = []
    train_targets: dict[str, list[int]] = {}

    for scr in train_scr_values:
        target_local = choose_train_targets(
            rng,
            args.range_cells,
            args.train_targets_per_scr,
            args.reference_cells,
        )
        train_targets[str(scr)] = [int(crop_start + pos) for pos in target_local]
        x_scr, y_scr = build_scr_samples(
            clutter=train_clutter,
            target_positions=target_local,
            scr_db=scr,
            speed_fn=lambda size, r=rng: r.uniform(args.train_speed_min, args.train_speed_max, size=size),
            pulses=args.pulses,
            reference_cells=args.reference_cells,
            prt=args.prt,
            wavelength=args.wavelength,
            max_windows=args.max_train_windows_per_scr,
        )
        train_x_parts.append(x_scr)
        train_y_parts.append(y_scr)
        train_scr_parts.append(np.full((len(x_scr),), scr, dtype=np.int16))

    x_train = np.concatenate(train_x_parts, axis=0)
    y_train = np.concatenate(train_y_parts, axis=0)
    scr_train = np.concatenate(train_scr_parts, axis=0)

    test_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for scr in test_scr_values:
        x_scr, y_scr = build_scr_samples(
            clutter=test_clutter,
            target_positions=[local_target],
            scr_db=scr,
            speed_fn=lambda size, speed=args.test_speed: np.full(size, speed, dtype=np.float64),
            pulses=args.pulses,
            reference_cells=args.reference_cells,
            prt=args.prt,
            wavelength=args.wavelength,
            max_windows=args.max_test_windows_per_scr,
        )
        test_sets[scr] = (x_scr, y_scr)

    manifest = build_manifest(
        args=args,
        train_mat=train_mat,
        test_mat=test_mat,
        crop_start=crop_start,
        crop_end=crop_end,
        local_target=local_target,
        train_clutter_shape=train_clutter.shape,
        test_clutter_shape=test_clutter.shape,
        train_targets=train_targets,
        x_train=x_train,
        y_train=y_train,
        test_sets=test_sets,
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


def choose_train_targets(
    rng: np.random.Generator,
    range_cells: int,
    count: int,
    reference_cells: int,
) -> list[int]:
    candidates = [
        pos
        for pos in range(range_cells)
        if len(local_reference_bins(pos, range_cells, {pos}, reference_cells)) == reference_cells
    ]
    if len(candidates) < count:
        raise ValueError(f"可选训练目标单元不足: need={count}, available={len(candidates)}")
    return sorted(int(pos) for pos in rng.choice(candidates, size=count, replace=False))


def build_scr_samples(
    clutter: np.ndarray,
    target_positions: list[int],
    scr_db: int,
    speed_fn,
    pulses: int,
    reference_cells: int,
    prt: float,
    wavelength: float,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    starts = list(range(0, clutter.shape[0] - pulses + 1, pulses))
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError("没有可用 pulse 窗口。")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    excluded = set(target_positions)
    ref_by_pos = {
        pos: local_reference_bins(pos, clutter.shape[1], excluded, reference_cells)
        for pos in target_positions
    }
    for pos, refs in ref_by_pos.items():
        if len(refs) != reference_cells:
            raise ValueError(f"目标单元 {pos} 的局部参考单元不足: {len(refs)} != {reference_cells}")

    for start in starts:
        segment = clutter[start : start + pulses, :].copy()
        speeds = speed_fn(len(target_positions))
        for pos, speed in zip(target_positions, speeds):
            refs = ref_by_pos[pos]
            clutter_power = float(np.mean(np.abs(segment[:, refs]) ** 2))
            target_amp = np.sqrt(clutter_power * (10.0 ** (scr_db / 10.0)))
            for p in range(pulses):
                phase = 4.0 * np.pi * float(speed) * prt * p / wavelength
                segment[p, pos] += target_amp * np.exp(1j * phase)

        labels = np.zeros(clutter.shape[1], dtype=np.int32)
        for pos in target_positions:
            labels[pos] = 1
        x_parts.append(np.stack([segment.real, segment.imag], axis=0).astype(np.float32))
        y_parts.append(labels)

    return np.stack(x_parts, axis=0), np.stack(y_parts, axis=0)


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
    local_target: int,
    train_clutter_shape: tuple[int, int],
    test_clutter_shape: tuple[int, int],
    train_targets: dict[str, list[int]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    test_sets: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    return {
        "dataset": "SDRDSP strict SCR local-crop reproduction",
        "source_files": {
            "train_background": str(train_mat),
            "test_background": str(test_mat),
            "train_sha256": file_sha256(train_mat),
            "test_sha256": file_sha256(test_mat),
            "mat_key": args.mat_key,
        },
        "protocol": {
            "train_background_name": TRAIN_MAT,
            "test_background_name": TEST_MAT,
            "train_scr_db": list(range(-12, 15, 2)),
            "test_scr_db": list(range(-24, 15, 2)),
            "pulses": args.pulses,
            "range_cells": args.range_cells,
            "reference_cells": args.reference_cells,
            "scr_reference_power": "mean_power_of_local_20_reference_cells",
            "train_targets_per_scr": args.train_targets_per_scr,
            "train_target_cells_global_by_scr": train_targets,
            "train_speed_mps": [args.train_speed_min, args.train_speed_max],
            "test_speed_mps": args.test_speed,
            "prt_seconds": args.prt,
            "wavelength_m": args.wavelength,
            "pulse_window": "non_overlapping_step_equals_P",
            "label_rule": "only_true_injected_range_cells_are_positive",
        },
        "crop": {
            "paper_global_target_cell": args.paper_target_cell,
            "array_index_convention": (
                "Python zero-based indices; paper_global_target_cell is preserved as the cited cell number. "
                "This local crop follows the prior repo convention local_target_index=paper_target_cell-crop_start."
            ),
            "crop_start": crop_start,
            "crop_end_exclusive": crop_end,
            "local_target_index": local_target,
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
    print(f"  crop: {manifest['crop']}")


if __name__ == "__main__":
    main()
