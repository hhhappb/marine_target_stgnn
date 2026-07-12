from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "1").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from paper_modules.experiments.train import git_commit, load_checkpoint
from paper_modules.datasets.ipix_window import list_split_files, load_ipix_arrays, parse_source_and_pol, seed_everything
from paper_modules.datasets.scr_npz import (
    SDRDSP_LOCAL_CROP_PROTOCOLS,
    SDRDSP_V2_PROTOCOL,
    list_test_scr_files,
    load_scr_arrays,
    validate_sdrdsp_v2_manifest,
)
from paper_modules.models import build_model
from utils.config import get_config_value, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已有 checkpoint 诊断 IPIX/SDRDSP 数据捷径。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-pfa", type=float, default=0.001)
    parser.add_argument("--pfa-values", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument(
        "--probes",
        choices=["d1", "d2", "d3", "d4", "cross"],
        nargs="+",
        default=["d1", "d2", "d3"],
    )
    parser.add_argument("--cross-test-data-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--shifts", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
    parser.add_argument("--max-threshold-windows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(get_config_value(config, "train.seed"))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cfg = config.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "ipix_window"))
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if dataset_type == "scr_npz":
        run_sdrdsp_probes(args, config, device, seed)
        return
    if dataset_type != "ipix_window":
        raise ValueError(f"leakage_probe 不支持 dataset.type={dataset_type!r}。")
    if args.probes != ["d1", "d2", "d3"] or args.pfa_values != [0.0001, 0.001, 0.01]:
        raise ValueError("--probes/--pfa-values 仅用于 SDRDSP scr_npz 诊断。")
    data_dir = Path(dataset_cfg.get("data_dir", get_config_value(config, "paths.data_dir")))
    pols = _as_list(dataset_cfg.get("polarizations", get_config_value(config, "ipix.polarizations"))) or []
    sources = _as_list(dataset_cfg.get("sources", dataset_cfg.get("source")))
    batch_size = args.batch_size or int(get_config_value(config, "train.batch_size"))
    train_files = list_split_files(data_dir, "train", pols, sources=sources)
    test_files = list_split_files(data_dir, "test", pols, sources=sources)
    if not train_files:
        raise SystemExit(f"No train files found under {data_dir}")
    if not test_files:
        raise SystemExit(f"No test files found under {data_dir}")
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    threshold, threshold_bins = compute_train_clutter_threshold(
        model,
        train_files,
        batch_size,
        device,
        args.target_pfa,
        max_total_windows=args.max_threshold_windows,
        seed=seed,
    )

    baseline = evaluate_variant(
        model,
        test_files,
        batch_size,
        device,
        threshold,
        label_mode="original",
        max_windows_per_file=args.max_test_windows_per_file,
        seed=seed,
    )
    shift_results = [
        evaluate_circular_shift(
            model,
            test_files,
            batch_size,
            device,
            threshold,
            shift=shift,
            max_windows_per_file=args.max_test_windows_per_file,
            seed=seed,
        )
        for shift in args.shifts
    ]
    replacement = evaluate_replacement(
        model,
        test_files,
        batch_size,
        device,
        threshold,
        max_windows_per_file=args.max_test_windows_per_file,
        seed=seed,
    )

    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "git_commit": git_commit(),
        "data_dir": str(data_dir),
        "sources": sources or [],
        "polarizations": pols,
        "train_files": [str(path) for path in train_files],
        "test_files": [str(path) for path in test_files],
        "num_train_files": len(train_files),
        "num_test_files": len(test_files),
        "target_pfa": args.target_pfa,
        "threshold_source": "train_clutter_original_inputs",
        "threshold": threshold,
        "num_clutter_bins_for_threshold": threshold_bins,
        "max_test_windows_per_file": args.max_test_windows_per_file,
        "max_threshold_windows": args.max_threshold_windows,
        "baseline": baseline,
        "circular_shift": shift_results,
        "target_cell_replacement": replacement,
        "interpretation_notes": [
            "circular_shift: shifted-label metrics high and original-label metrics low means alarms follow shifted echoes.",
            "circular_shift: original-label metrics high after input shift means alarms stayed near memorized range positions.",
            "target_cell_replacement uses same-window farthest non-target cell; it is a diagnostic proxy, not proof of pure clutter.",
        ],
    }

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"Saved leakage probe results: {output_path}", flush=True)


def run_sdrdsp_probes(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> None:
    """在冻结 checkpoint 上运行 SDRDSP D1-D3 诊断，不改变训练数据或模型。"""
    dataset_cfg = config.get("dataset", {})
    data_dir = Path(dataset_cfg.get("data_dir", get_config_value(config, "paths.data_dir")))
    protocol = str(dataset_cfg.get("protocol", ""))
    if protocol not in SDRDSP_LOCAL_CROP_PROTOCOLS:
        raise ValueError(f"SDRDSP 捷径诊断不支持 protocol={protocol!r}。")
    if any(probe in args.probes for probe in ("d1", "d2", "d3", "d4")) and protocol != SDRDSP_V2_PROTOCOL:
        raise ValueError("D1-D4 必须以理想目标 sdrdsp_fig9_local_crop_v2 为训练/测试协议。")
    if "cross" in args.probes and not args.cross_test_data_dirs:
        raise ValueError("cross 探针需要 --cross-test-data-dirs。")
    pfa_values = validate_pfa_values(args.pfa_values)
    if args.target_pfa not in pfa_values:
        raise ValueError("--target-pfa 必须同时出现在 --pfa-values 中。")
    if "d2" in args.probes:
        if not args.shifts or any(shift == 0 for shift in args.shifts):
            raise ValueError("D2 的 --shifts 必须是非零距离平移量。")
        if len(set(args.shifts)) != len(args.shifts):
            raise ValueError("D2 的 --shifts 不允许重复。")

    pulses = int(get_config_value(config, "model.pulses"))
    range_cells = int(get_config_value(config, "model.range_cells"))
    if "d2" in args.probes:
        normalized_shifts = [shift % range_cells for shift in args.shifts]
        if any(shift == 0 for shift in normalized_shifts):
            raise ValueError(f"D2 shift 不能是 range_cells={range_cells} 的整数倍。")
        if len(set(normalized_shifts)) != len(normalized_shifts):
            raise ValueError("D2 shifts 在循环距离维上存在等价重复。")
    manifest = validate_sdrdsp_v2_manifest(data_dir, pulses, range_cells)
    batch_size = args.batch_size or int(get_config_value(config, "train.batch_size"))
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，实际为 {batch_size}。")

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    train_x, train_y, _ = load_scr_arrays(
        data_dir,
        "train",
        max_windows=args.max_threshold_windows,
        rng=np.random.default_rng(seed),
        require_scr_metadata=True,
    )
    train_scores = infer_sdrdsp_score_spaces(model, train_x, batch_size, device)
    train_clutter = {name: values[train_y == 0] for name, values in train_scores.items()}
    del train_x, train_y, train_scores
    if any(values.size == 0 for values in train_clutter.values()):
        raise ValueError("SDRDSP 训练集没有可用杂波分数，无法校准阈值。")

    test_files = list_test_scr_files(data_dir)
    baseline_records = collect_sdrdsp_records(
        model,
        test_files,
        batch_size,
        device,
        args.max_test_windows_per_file,
        seed,
        variant="original",
    )

    calibrations = {
        score_name: {
            str(pfa): strict_rank_threshold(clutter_scores, pfa)
            for pfa in pfa_values
        }
        for score_name, clutter_scores in train_clutter.items()
    }
    results: dict[str, Any] = {
        "protocol": "sdrdsp_shortcut_probe_v1",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "git_commit": git_commit(),
        "device": str(device),
        "data_dir": str(data_dir),
        "dataset_protocol": protocol,
        "probes": args.probes,
        "pfa_values": pfa_values,
        "target_pfa": args.target_pfa,
        "score_direction": "score <= threshold means target",
        "primary_score": "logit_clutter_minus_target",
        "threshold_source": "train_clutter",
        "threshold_rule": "strict_rank_floor_budget_excluding_boundary_ties",
        "num_clutter_bins_for_threshold": int(train_clutter["logit_margin"].size),
        "max_threshold_windows": args.max_threshold_windows,
        "max_test_windows_per_file": args.max_test_windows_per_file,
        "calibration": calibrations,
        "baseline_logit_margin": {
            "pfa": {
                str(pfa): evaluate_sdrdsp_records(
                    baseline_records,
                    "logit_margin",
                    calibrations["logit_margin"][str(pfa)]["threshold"],
                )
                for pfa in pfa_values
            }
        },
    }

    if "d1" in args.probes:
        results["d1_stable_threshold"] = build_d1_results(
            train_clutter,
            baseline_records,
            calibrations,
            pfa_values,
        )
    if "d2" in args.probes:
        results["d2_range_position"] = build_d2_results(
            model,
            test_files,
            batch_size,
            device,
            args.max_test_windows_per_file,
            seed,
            args.shifts,
            calibrations["logit_margin"],
            pfa_values,
        )
    if "d3" in args.probes:
        results["d3_target_phase"] = build_d3_results(
            model,
            test_files,
            batch_size,
            device,
            args.max_test_windows_per_file,
            seed,
            calibrations["logit_margin"],
            pfa_values,
        )
    if "d4" in args.probes:
        results["d4_exact_target_removal"] = build_d4_results(
            model,
            test_files,
            batch_size,
            device,
            args.max_test_windows_per_file,
            calibrations["logit_margin"],
            pfa_values,
            manifest,
        )
    if "cross" in args.probes:
        results["cross_domain"] = build_cross_domain_results(
            model=model,
            train_data_dir=data_dir,
            test_data_dirs=args.cross_test_data_dirs,
            batch_size=batch_size,
            device=device,
            max_windows_per_file=args.max_test_windows_per_file,
            seed=seed,
            calibrations=calibrations["logit_margin"],
            pfa_values=pfa_values,
            pulses=pulses,
            range_cells=range_cells,
        )

    output_path = args.output or default_sdrdsp_output_path()
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有诊断结果: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print_sdrdsp_summary(results)
    print(f"Saved SDRDSP shortcut probe results: {output_path}", flush=True)


def validate_pfa_values(values: list[float]) -> list[float]:
    pfa_values = [float(value) for value in values]
    if not pfa_values or any(not 0.0 < value < 1.0 for value in pfa_values):
        raise ValueError(f"Pfa 必须全部位于 (0, 1)，实际为 {pfa_values}。")
    if len(set(pfa_values)) != len(pfa_values):
        raise ValueError(f"Pfa 不允许重复，实际为 {pfa_values}。")
    return pfa_values


def infer_sdrdsp_score_spaces(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"SDRDSP X shape 应为 [B, 2, P, N]，实际为 {x.shape}。")
    margin_parts: list[np.ndarray] = []
    softmax_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            real = torch.from_numpy(batch[:, 0].astype(np.float32, copy=False)).to(device)
            imag = torch.from_numpy(batch[:, 1].astype(np.float32, copy=False)).to(device)
            logits = model(torch.complex(real, imag))
            if logits.ndim != 3 or logits.shape[1] != 2:
                raise ValueError(f"模型输出应为 [B, 2, N]，实际为 {tuple(logits.shape)}。")
            margin_parts.append((logits[:, 0, :] - logits[:, 1, :]).cpu().numpy().astype(np.float64))
            softmax_parts.append(
                torch.softmax(logits, dim=1)[:, 0, :].cpu().numpy().astype(np.float64)
            )
    if not margin_parts:
        raise ValueError("SDRDSP 输入为空，无法推理分数。")
    return {
        "logit_margin": np.concatenate(margin_parts, axis=0),
        "softmax_clutter": np.concatenate(softmax_parts, axis=0),
    }


def strict_rank_threshold(scores: np.ndarray, target_pfa: float) -> dict[str, Any]:
    """选择不超过虚警预算的阈值；边界并列时整体排除该并列组。"""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("阈值分数必须是非空有限数组。")
    if not 0.0 < target_pfa < 1.0:
        raise ValueError(f"target_pfa 必须位于 (0, 1)，实际为 {target_pfa}。")
    ordered = np.sort(flat)
    budget = int(np.floor(target_pfa * len(ordered)))
    if budget == 0:
        boundary = float(ordered[0])
        threshold = float(np.nextafter(boundary, -np.inf))
        boundary_tie_count = int(np.count_nonzero(ordered == boundary))
    else:
        boundary = float(ordered[budget - 1])
        inclusive_count = int(np.searchsorted(ordered, boundary, side="right"))
        boundary_tie_count = int(np.count_nonzero(ordered == boundary))
        threshold = boundary if inclusive_count <= budget else float(np.nextafter(boundary, -np.inf))
    selected = int(np.count_nonzero(flat <= threshold))
    return {
        "target_pfa": float(target_pfa),
        "threshold": threshold,
        "num_scores": int(flat.size),
        "false_alarm_budget": budget,
        "selected_clutter_bins": selected,
        "calibration_actual_pf": selected / flat.size,
        "boundary_value": boundary,
        "boundary_tie_count": boundary_tie_count,
        "boundary_tie_excluded": bool(threshold < boundary),
    }


def collect_sdrdsp_records(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    max_windows_per_file: int | None,
    seed: int,
    variant: str,
    shift: int | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    transform_rng = np.random.default_rng(seed + 10_003)
    records: list[dict[str, Any]] = []
    for path in files:
        scr = int(path.stem.replace("test_scr_", ""))
        x, y, _ = load_scr_arrays(
            path.parent,
            "test",
            scr=scr,
            max_windows=max_windows_per_file,
            rng=rng,
            require_scr_metadata=True,
        )
        original_labels = y
        labels = y
        if variant == "range_shift":
            if shift is None or shift == 0:
                raise ValueError("range_shift 需要非零 shift。")
            x = np.roll(x, shift=shift, axis=3)
            labels = np.roll(y, shift=shift, axis=1)
        elif variant in {"target_pulse_shuffle", "target_phase_random"}:
            x = transform_target_cells(x, y, variant, transform_rng)
        elif variant != "original":
            raise ValueError(f"未知 SDRDSP probe variant: {variant}")
        scores = infer_sdrdsp_score_spaces(model, x, batch_size, device)
        records.append(
            {
                "scr_db": scr,
                "labels": labels,
                "original_labels": original_labels,
                **scores,
            }
        )
    return records


def transform_target_cells(
    x: np.ndarray,
    labels: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """仅变换标签目标单元；非目标单元保持逐值不变。"""
    if x.ndim != 4 or x.shape[1] != 2 or labels.shape != (len(x), x.shape[-1]):
        raise ValueError(f"目标相位变换 shape 不匹配: X={x.shape}, y={labels.shape}。")
    transformed = x.copy()
    complex_x = x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64)
    for window_idx in range(len(x)):
        targets = np.flatnonzero(labels[window_idx] == 1)
        for target in targets:
            values = complex_x[window_idx, :, target]
            if mode == "target_pulse_shuffle":
                permutation = rng.permutation(len(values))
                if len(values) > 1 and np.array_equal(permutation, np.arange(len(values))):
                    permutation = np.roll(permutation, 1)
                changed = values[permutation]
            elif mode == "target_phase_random":
                phases = rng.uniform(-np.pi, np.pi, size=len(values))
                changed = np.abs(values) * np.exp(1j * phases)
            else:
                raise ValueError(f"未知目标相位变换: {mode}")
            transformed[window_idx, 0, :, target] = changed.real.astype(np.float32)
            transformed[window_idx, 1, :, target] = changed.imag.astype(np.float32)
    return transformed


def evaluate_sdrdsp_records(
    records: list[dict[str, Any]],
    score_name: str,
    threshold: float,
    label_name: str = "labels",
) -> dict[str, Any]:
    total = empty_counts()
    per_scr: list[dict[str, Any]] = []
    for item in records:
        labels = item[label_name]
        det = (item[score_name] <= threshold).astype(np.uint8)
        counts = count_detection(det, labels)
        add_counts(total, counts)
        per_scr.append({"scr_db": item["scr_db"], **pd_pf(counts), **counts})
    return {**pd_pf(total), **total, "per_scr": per_scr}


def build_d1_results(
    train_clutter: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    calibrations: dict[str, dict[str, dict[str, Any]]],
    pfa_values: list[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {"score_spaces": {}}
    for score_name, clutter_scores in train_clutter.items():
        score_result: dict[str, Any] = {
            "num_unique_train_clutter_scores": int(np.unique(clutter_scores).size),
            "unique_fraction": float(np.unique(clutter_scores).size / clutter_scores.size),
            "num_exact_one": int(np.count_nonzero(clutter_scores == 1.0)),
            "pfa": {},
        }
        for pfa in pfa_values:
            calibration = calibrations[score_name][str(pfa)]
            score_result["pfa"][str(pfa)] = {
                **calibration,
                "test": evaluate_sdrdsp_records(records, score_name, calibration["threshold"]),
            }
        result["score_spaces"][score_name] = score_result
    result["interpretation"] = (
        "正式诊断使用 logit_margin；softmax_clutter 仅用于量化概率饱和。"
        "若 softmax 大量等于 1 而 logit_margin 保持可排序，则原 Pfa=0.01 坍缩属于数值/并列阈值问题。"
    )
    return result


def build_d2_results(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    max_windows_per_file: int | None,
    seed: int,
    shifts: list[int],
    calibrations: dict[str, dict[str, Any]],
    pfa_values: list[float],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for shift in shifts:
        records = collect_sdrdsp_records(
            model,
            files,
            batch_size,
            device,
            max_windows_per_file,
            seed,
            variant="range_shift",
            shift=shift,
        )
        pfa_result: dict[str, Any] = {}
        for pfa in pfa_values:
            threshold = calibrations[str(pfa)]["threshold"]
            pfa_result[str(pfa)] = {
                "metrics_vs_shifted_labels": evaluate_sdrdsp_records(
                    records, "logit_margin", threshold, "labels"
                ),
                "metrics_vs_original_labels": evaluate_sdrdsp_records(
                    records, "logit_margin", threshold, "original_labels"
                ),
            }
        items.append({"shift": shift, "pfa": pfa_result})
    return {
        "variant": "circular_range_shift_all_echoes",
        "items": items,
        "interpretation": (
            "平移标签 PD 高且原标签 PD 低，说明报警跟随回波；原标签 PD 仍高则提示固定距离位置记忆。"
            "该探针循环平移整幅距离像，因此只用于位置记忆诊断，不代表真实目标迁移性能。"
        ),
    }


def build_d3_results(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    max_windows_per_file: int | None,
    seed: int,
    calibrations: dict[str, dict[str, Any]],
    pfa_values: list[float],
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in ("target_pulse_shuffle", "target_phase_random"):
        records = collect_sdrdsp_records(
            model,
            files,
            batch_size,
            device,
            max_windows_per_file,
            seed,
            variant=variant,
        )
        variants[variant] = {
            "pfa": {
                str(pfa): evaluate_sdrdsp_records(
                    records,
                    "logit_margin",
                    calibrations[str(pfa)]["threshold"],
                )
                for pfa in pfa_values
            }
        }
    return {
        "variants": variants,
        "non_target_cells_unchanged": True,
        "interpretation": (
            "pulse_shuffle 保留目标单元四个复数样本但打乱顺序；phase_random 保留目标单元总幅度并随机化相位。"
            "两者只改标签目标单元，是诊断代理，不等价于重新生成物理目标。"
        ),
    }


def build_d4_results(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    max_windows_per_file: int | None,
    calibrations: dict[str, dict[str, Any]],
    pfa_values: list[float],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """精确减去理想注入项，并与原始测试背景逐值核验。"""
    raw_windows = load_raw_test_windows(manifest, max_windows_per_file)
    records: list[dict[str, Any]] = []
    max_abs_residual = 0.0
    squared_residual = 0.0
    squared_raw = 0.0
    strict_allclose = True
    ulp_aware_allclose = True
    max_residual_ulps = 0.0
    for path in files:
        scr = int(path.stem.replace("test_scr_", ""))
        x, y, _ = load_scr_arrays(
            path.parent,
            "test",
            scr=scr,
            max_windows=max_windows_per_file,
            rng=np.random.default_rng(0),
            require_scr_metadata=True,
        )
        removed = remove_ideal_test_target(x, scr, manifest)
        raw = raw_windows[: len(removed)]
        residual = removed.astype(np.float64) - raw.astype(np.float64)
        max_abs_residual = max(max_abs_residual, float(np.max(np.abs(residual))))
        squared_residual += float(np.sum(residual * residual, dtype=np.float64))
        squared_raw += float(np.sum(raw.astype(np.float64) ** 2, dtype=np.float64))
        strict_allclose = strict_allclose and bool(np.allclose(removed, raw, rtol=1e-6, atol=1e-3))
        scale = np.maximum(np.abs(x), np.abs(raw)).astype(np.float32)
        ulp = np.spacing(scale).astype(np.float64)
        ulp_limit = 2.0 * ulp + np.finfo(np.float32).tiny
        ulp_aware_allclose = ulp_aware_allclose and bool(np.all(np.abs(residual) <= ulp_limit))
        positive_ulp = ulp > 0
        if np.any(positive_ulp):
            max_residual_ulps = max(
                max_residual_ulps,
                float(np.max(np.abs(residual[positive_ulp]) / ulp[positive_ulp])),
            )
        scores = infer_sdrdsp_score_spaces(model, removed, batch_size, device)
        records.append({"scr_db": scr, "labels": np.zeros_like(y), "former_labels": y, **scores})
    if not ulp_aware_allclose:
        raise ValueError(
            f"D4 删除注入项后超出 2 ULP 舍入上限: max_abs_residual={max_abs_residual:.6g}, "
            f"max_residual_ulps={max_residual_ulps:.3f}。"
        )

    pfa_result: dict[str, Any] = {}
    for pfa in pfa_values:
        threshold = calibrations[str(pfa)]["threshold"]
        clutter_metrics = evaluate_sdrdsp_records(records, "logit_margin", threshold)
        former_target_alarms = 0
        former_target_bins = 0
        per_scr = []
        for item in records:
            det = item["logit_margin"] <= threshold
            mask = item["former_labels"] == 1
            alarms = int(np.count_nonzero(det & mask))
            bins = int(np.count_nonzero(mask))
            former_target_alarms += alarms
            former_target_bins += bins
            per_scr.append({"scr_db": item["scr_db"], "alarm_rate": alarms / bins, "alarms": alarms, "bins": bins})
        pfa_result[str(pfa)] = {
            "threshold": threshold,
            "all_background_metrics": clutter_metrics,
            "former_target_alarm_rate": former_target_alarms / former_target_bins,
            "former_target_alarms": former_target_alarms,
            "former_target_bins": former_target_bins,
            "per_scr": per_scr,
        }
    return {
        "removal": "subtract_reconstructed_ideal_complex_target_component",
        "raw_background_allclose_rtol": 1e-6,
        "raw_background_allclose_atol": 1e-3,
        "raw_background_strict_allclose": strict_allclose,
        "raw_background_ulp_aware_allclose": ulp_aware_allclose,
        "raw_background_ulp_limit": 2.0,
        "max_abs_residual": max_abs_residual,
        "max_residual_ulps": max_residual_ulps,
        "relative_l2_residual": float(np.sqrt(squared_residual / squared_raw)),
        "pfa": pfa_result,
    }


def load_raw_test_windows(manifest: dict[str, Any], max_windows: int | None) -> np.ndarray:
    source = Path(manifest["source_files"]["test_background"])
    key = str(manifest["source_files"]["mat_key"])
    payload = sio.loadmat(source)
    if key not in payload:
        raise KeyError(f"D4 原始测试文件缺少 {key!r}: {source}。")
    crop = manifest["crop"]
    clutter = np.asarray(payload[key])[:, int(crop["crop_start_zero_based"]) : int(crop["crop_end_exclusive_zero_based"])]
    pulses = int(manifest["protocol"]["pulses"])
    starts = list(range(0, len(clutter) - pulses + 1, pulses))
    if max_windows is not None:
        starts = starts[:max_windows]
    return np.stack(
        [np.stack([clutter[start : start + pulses].real, clutter[start : start + pulses].imag]) for start in starts]
    ).astype(np.float32)


def remove_ideal_test_target(x: np.ndarray, scr: int, manifest: dict[str, Any]) -> np.ndarray:
    audit = manifest["audit"]["test_injection_by_scr"][str(scr)]
    record = audit["target_records"][0]
    pulses = int(manifest["protocol"]["pulses"])
    speed = float(record["speed_mps"])
    amplitude = float(record["target_amplitude"])
    prt = float(manifest["protocol"]["prt_seconds"])
    wavelength = float(manifest["protocol"]["wavelength_m"])
    target = int(record["target_position_local_zero_based"])
    pulse_indices = np.arange(len(x) * pulses, dtype=np.float64)
    signal = amplitude * np.exp(1j * 4.0 * np.pi * speed * prt * pulse_indices / wavelength)
    removed = x.copy()
    values = removed[:, 0, :, target].astype(np.float64) + 1j * removed[:, 1, :, target].astype(np.float64)
    values -= signal.reshape(len(x), pulses)
    removed[:, 0, :, target] = values.real.astype(np.float32)
    removed[:, 1, :, target] = values.imag.astype(np.float32)
    return removed


def build_cross_domain_results(
    model: nn.Module,
    train_data_dir: Path,
    test_data_dirs: list[Path],
    batch_size: int,
    device: torch.device,
    max_windows_per_file: int | None,
    seed: int,
    calibrations: dict[str, dict[str, Any]],
    pfa_values: list[float],
    pulses: int,
    range_cells: int,
) -> dict[str, Any]:
    train_manifest = validate_sdrdsp_v2_manifest(train_data_dir, pulses, range_cells)
    domains: list[dict[str, Any]] = []
    for test_dir in test_data_dirs:
        test_manifest = validate_sdrdsp_v2_manifest(test_dir, pulses, range_cells)
        records = collect_sdrdsp_records(
            model,
            list_test_scr_files(test_dir),
            batch_size,
            device,
            max_windows_per_file,
            seed,
            variant="original",
        )
        pfa_payload: dict[str, Any] = {}
        for pfa in pfa_values:
            metrics = evaluate_sdrdsp_records(records, "logit_margin", calibrations[str(pfa)]["threshold"])
            low = [item for item in metrics["per_scr"] if item["scr_db"] in {-24, -22, -20, -18, -16}]
            pfa_payload[str(pfa)] = {
                "threshold": calibrations[str(pfa)]["threshold"],
                "metrics": metrics,
                "low_scr_mean_pd": float(np.mean([item["PD"] for item in low])),
                "low_scr_pd_auc": float(
                    np.trapezoid([item["PD"] for item in low], [item["scr_db"] for item in low])
                ),
                "pf_guardrail_pass": bool(metrics["PF"] <= 2.0 * pfa),
            }
        domains.append(
            {
                "test_data_dir": str(test_dir),
                "test_protocol": test_manifest["protocol"]["id"],
                "pfa": pfa_payload,
            }
        )
    return {
        "train_data_dir": str(train_data_dir),
        "train_protocol": train_manifest["protocol"]["id"],
        "threshold_source": "train_clutter_logit_margin",
        "domains": domains,
    }


def default_sdrdsp_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / "training" / f"{stamp}_sdrdsp_shortcut_probe" / "probe_results.json"


def print_sdrdsp_summary(results: dict[str, Any]) -> None:
    print("SDRDSP shortcut probe summary", flush=True)
    print(
        f"checkpoint={results['checkpoint']} | threshold_source={results['threshold_source']} "
        f"| threshold_bins={results['num_clutter_bins_for_threshold']}",
        flush=True,
    )
    d1 = results.get("d1_stable_threshold")
    if d1:
        for score_name, score_result in d1["score_spaces"].items():
            print(
                f"D1 {score_name} | unique_fraction={score_result['unique_fraction']:.6f} "
                f"| exact_one={score_result['num_exact_one']}",
                flush=True,
            )
            for pfa, item in score_result["pfa"].items():
                test = item["test"]
                print(
                    f"  Pfa={pfa} | threshold={item['threshold']:.8g} "
                    f"| calibration_PF={item['calibration_actual_pf']:.6g} "
                    f"| test_PD={test['PD']:.4f} | test_PF={test['PF']:.6f}",
                    flush=True,
                )
    d2 = results.get("d2_range_position")
    if d2:
        pfa = str(results["target_pfa"])
        for item in d2["items"]:
            shifted = item["pfa"][pfa]["metrics_vs_shifted_labels"]
            original = item["pfa"][pfa]["metrics_vs_original_labels"]
            print(
                f"D2 shift={item['shift']} | shifted-label PD={shifted['PD']:.4f}, PF={shifted['PF']:.6f} "
                f"| original-label PD={original['PD']:.4f}",
                flush=True,
            )
    d3 = results.get("d3_target_phase")
    if d3:
        pfa = str(results["target_pfa"])
        for variant, item in d3["variants"].items():
            metrics = item["pfa"][pfa]
            print(f"D3 {variant} | PD={metrics['PD']:.4f} | PF={metrics['PF']:.6f}", flush=True)
    d4 = results.get("d4_exact_target_removal")
    if d4:
        item = d4["pfa"][str(results["target_pfa"])]
        print(
            f"D4 removal | residual={d4['max_abs_residual']:.6g} "
            f"| global_PF={item['all_background_metrics']['PF']:.6f} "
            f"| former-target alarm={item['former_target_alarm_rate']:.6f}",
            flush=True,
        )
    cross = results.get("cross_domain")
    if cross:
        pfa = str(results["target_pfa"])
        for domain in cross["domains"]:
            item = domain["pfa"][pfa]
            print(
                f"cross {cross['train_protocol']} -> {domain['test_protocol']} "
                f"| low-SCR PD={item['low_scr_mean_pd']:.4f} "
                f"| PF={item['metrics']['PF']:.6f} | guardrail={item['pf_guardrail_pass']}",
                flush=True,
            )


def compute_train_clutter_threshold(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    target_pfa: float,
    max_total_windows: int | None,
    seed: int,
) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    clutter_parts: list[np.ndarray] = []
    remaining = max_total_windows
    for path in files:
        if remaining is not None and remaining <= 0:
            break
        limit = remaining
        x, y = load_ipix_arrays(path, max_windows=limit, rng=rng)
        if remaining is not None:
            remaining -= len(x)
        o0 = infer_o0(model, x, batch_size, device)
        clutter_parts.append(o0[y == 0])
    if not clutter_parts:
        raise ValueError("没有训练集杂波分数，无法计算 train_clutter 阈值。")
    clutter = np.concatenate(clutter_parts)
    if clutter.size == 0:
        raise ValueError("训练集杂波分数为空，无法计算 train_clutter 阈值。")
    ordered = np.sort(clutter)
    idx = int(np.ceil(target_pfa * len(ordered))) - 1
    threshold = float(ordered[max(0, min(idx, len(ordered) - 1))])
    return threshold, int(clutter.shape[0])


def evaluate_variant(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    threshold: float,
    label_mode: str,
    max_windows_per_file: int | None,
    seed: int,
    input_shift: int | None = None,
    replace_targets: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    total = empty_counts()
    per_file: list[dict[str, Any]] = []
    target_rates: list[float] = []
    shifted_target_rates: list[float] = []

    for path in files:
        x, y = load_ipix_arrays(path, max_windows=max_windows_per_file, rng=rng)
        labels = y
        shifted_labels = None
        if input_shift is not None:
            x = np.roll(x, shift=input_shift, axis=2)
            shifted_labels = np.roll(y, shift=input_shift, axis=1)
            labels = shifted_labels if label_mode == "shifted" else y
        if replace_targets:
            x = replace_target_cells_with_farthest_clutter(x, y)

        o0 = infer_o0(model, x, batch_size, device)
        det = (o0 <= threshold).astype(np.uint8)
        counts = count_detection(det, labels)
        add_counts(total, counts)
        source, pol = parse_source_and_pol(path)
        per_record = {
            "source": source,
            "polarization": pol,
            **pd_pf(counts),
            **counts,
        }
        if input_shift is not None and shifted_labels is not None:
            per_record["det_rate_on_original_target_cells"] = masked_rate(det, y == 1)
            per_record["det_rate_on_shifted_target_cells"] = masked_rate(det, shifted_labels == 1)
            target_rates.append(per_record["det_rate_on_original_target_cells"])
            shifted_target_rates.append(per_record["det_rate_on_shifted_target_cells"])
        per_file.append(per_record)

    result = {
        "label_mode": label_mode,
        **pd_pf(total),
        **total,
        "per_file": per_file,
    }
    if target_rates:
        result["mean_det_rate_on_original_target_cells"] = float(np.mean(target_rates))
        result["mean_det_rate_on_shifted_target_cells"] = float(np.mean(shifted_target_rates))
    return result


def evaluate_circular_shift(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    threshold: float,
    shift: int,
    max_windows_per_file: int | None,
    seed: int,
) -> dict[str, Any]:
    return {
        "shift": shift,
        "metrics_vs_original_labels": evaluate_variant(
            model,
            files,
            batch_size,
            device,
            threshold,
            "original",
            max_windows_per_file,
            seed,
            input_shift=shift,
        ),
        "metrics_vs_shifted_labels": evaluate_variant(
            model,
            files,
            batch_size,
            device,
            threshold,
            "shifted",
            max_windows_per_file,
            seed,
            input_shift=shift,
        ),
    }


def evaluate_replacement(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    threshold: float,
    max_windows_per_file: int | None,
    seed: int,
) -> dict[str, Any]:
    result = evaluate_variant(
        model,
        files,
        batch_size,
        device,
        threshold,
        "original",
        max_windows_per_file,
        seed,
        replace_targets=True,
    )
    result["replacement_mode"] = "same_window_farthest_non_target_cell"
    return result


def infer_o0(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            real = torch.from_numpy(batch.real.astype(np.float32, copy=False)).to(device)
            imag = torch.from_numpy(batch.imag.astype(np.float32, copy=False)).to(device)
            logits = model(torch.complex(real, imag))
            parts.append(torch.softmax(logits, dim=1)[:, 0, :].cpu().numpy())
    return np.concatenate(parts, axis=0)


def replace_target_cells_with_farthest_clutter(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y0 = labels[0]
    targets = np.flatnonzero(y0 == 1)
    clutters = np.flatnonzero(y0 == 0)
    if targets.size == 0 or clutters.size == 0:
        return x.copy()

    replaced = x.copy()
    for target in targets:
        source = int(clutters[np.argmax(np.abs(clutters - target))])
        replaced[:, :, target] = x[:, :, source]
    return replaced


def count_detection(det: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(((det == 1) & (labels == 1)).sum()),
        "FN": int(((det == 0) & (labels == 1)).sum()),
        "FP": int(((det == 1) & (labels == 0)).sum()),
        "TN": int(((det == 0) & (labels == 0)).sum()),
    }


def empty_counts() -> dict[str, int]:
    return {"TP": 0, "FN": 0, "FP": 0, "TN": 0}


def add_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    for key in total:
        total[key] += counts[key]


def pd_pf(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    return {
        "PD": tp / (tp + fn) if (tp + fn) else 0.0,
        "PF": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def masked_rate(det: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(det[mask].mean())


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / "training" / f"{stamp}_ipix_leakage_probe" / "leakage_probe_results.json"


def print_summary(results: dict[str, Any]) -> None:
    baseline = results["baseline"]
    print("Leakage probe summary", flush=True)
    print(
        f"threshold_source={results['threshold_source']} | target_pfa={results['target_pfa']} "
        f"| threshold={results['threshold']:.8f} | threshold_bins={results['num_clutter_bins_for_threshold']}",
        flush=True,
    )
    print(f"baseline | PD={baseline['PD']:.4f} | PF={baseline['PF']:.6f}", flush=True)
    for item in results["circular_shift"]:
        original = item["metrics_vs_original_labels"]
        shifted = item["metrics_vs_shifted_labels"]
        print(
            f"shift={item['shift']} | original-label PD={original['PD']:.4f}, PF={original['PF']:.6f} "
            f"| shifted-label PD={shifted['PD']:.4f}, PF={shifted['PF']:.6f}",
            flush=True,
        )
    repl = results["target_cell_replacement"]
    print(f"replacement | PD={repl['PD']:.4f} | PF={repl['PF']:.6f} | mode={repl['replacement_mode']}", flush=True)


if __name__ == "__main__":
    main()
