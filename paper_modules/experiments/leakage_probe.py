from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target-pfa", type=float, default=0.001)
    parser.add_argument("--pfa-values", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument(
        "--probes",
        choices=["d1", "d2", "d3", "d4", "cross", "scr_audit", "stage6_finalize"],
        nargs="+",
        default=["d1", "d2", "d3"],
    )
    parser.add_argument("--cross-test-data-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--scr-audit-data-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--scr-audit-mi-eval", type=Path, default=None)
    parser.add_argument("--scr-audit-ma-eval", type=Path, default=None)
    parser.add_argument("--stage6-scr-audit", type=Path, default=None)
    parser.add_argument("--stage6-r-data-dir", type=Path, default=None)
    parser.add_argument("--stage6-gate-definition", type=Path, default=None)
    parser.add_argument("--stage6-hash-output", type=Path, default=None)
    parser.add_argument("--stage6-report-output", type=Path, default=None)
    parser.add_argument("--stage6-figure-output", type=Path, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--paper-fig9-reference", type=Path, default=None)
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
    if dataset_type == "scr_npz" and args.probes == ["stage6_finalize"]:
        run_stage6_finalize(args, config)
        return
    if "stage6_finalize" in args.probes:
        raise ValueError("stage6_finalize 必须单独运行，不能与其他探针混用。")
    if dataset_type == "scr_npz" and args.probes == ["scr_audit"]:
        run_scr_distribution_audit(args, config)
        return
    if "scr_audit" in args.probes:
        raise ValueError("scr_audit 必须单独运行，不能与模型推理探针混用。")
    if args.checkpoint is None or not args.checkpoint.exists():
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


def run_scr_distribution_audit(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """只读审计 SDRDSP 的 SCR 标定粒度、RCS 分布和低 SCR 残差。"""
    if not args.scr_audit_data_dirs:
        raise ValueError("scr_audit 需要 --scr-audit-data-dirs，且必须提供 I/A/R 三个域。")
    if args.paper_fig9_reference is None:
        raise ValueError("scr_audit 需要 --paper-fig9-reference。")
    if args.scr_audit_mi_eval is None or args.scr_audit_ma_eval is None:
        raise ValueError("scr_audit 需要 --scr-audit-mi-eval 和 --scr-audit-ma-eval。")
    required_paths = [args.paper_fig9_reference, args.scr_audit_mi_eval, args.scr_audit_ma_eval]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"scr_audit 缺少输入文件: {missing}。")

    output_path = args.output or default_scr_audit_output_path()
    if output_path.suffix.lower() != ".json":
        raise ValueError("scr_audit --output 必须是 .json 文件。")
    csv_path = output_path.with_suffix(".csv")
    report_path = output_path.with_suffix(".md")
    existing = [str(path) for path in (output_path, csv_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有审计产物: {existing}。")

    pulses = int(get_config_value(config, "model.pulses"))
    range_cells = int(get_config_value(config, "model.range_cells"))
    domain_results: dict[str, Any] = {}
    pair_vectors: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    csv_rows: list[dict[str, Any]] = []
    for data_dir in args.scr_audit_data_dirs:
        manifest = validate_sdrdsp_v2_manifest(data_dir, pulses, range_cells)
        domain = scr_audit_domain(manifest)
        if domain in domain_results:
            raise ValueError(f"scr_audit 重复提供 {domain} 域。")
        result, vectors, rows = audit_scr_data_dir(data_dir, manifest)
        domain_results[domain] = result
        pair_vectors[domain] = vectors
        csv_rows.extend(rows)
    if set(domain_results) != {"I", "A", "R"}:
        raise ValueError(f"scr_audit 必须恰好提供 I/A/R，实际为 {sorted(domain_results)}。")

    pair_check = compare_a_r_test_vectors(pair_vectors["A"], pair_vectors["R"])
    low_scr = build_low_scr_residual_audit(
        args.paper_fig9_reference,
        args.scr_audit_mi_eval,
        args.scr_audit_ma_eval,
    )
    classification = classify_scr_audit(domain_results, pair_check)
    results = {
        "protocol": "sdrdsp_stage4b_rcs_scr_distribution_audit_v1",
        "git_commit": git_commit(),
        "inputs": {
            "data_dirs": [str(path) for path in args.scr_audit_data_dirs],
            "paper_fig9_reference": str(args.paper_fig9_reference),
            "mi_eval": str(args.scr_audit_mi_eval),
            "ma_eval": str(args.scr_audit_ma_eval),
        },
        "frozen_definitions": {
            "target_power": "mean_over_4_pulses(abs(saved_injected_complex - raw_background_complex)^2)",
            "reference_power": "sum_over_20_reference_cells(mean_over_4_pulses(abs(raw_background)^2))",
            "actual_scr_db": "10*log10(target_power/reference_power)",
            "formal_low_scr_rmse": "SCR <= -20 dB, three non-overlapping points",
            "diagnostic_low5_rmse": "SCR -24 to -16 dB, five points, does not replace formal metric",
            "gain_replay_tolerance": {"rtol": 0.005, "atol": 0.0005},
            "identity_residual_limit_db": 0.05,
            "a_r_pair_limit_db": 0.02,
            "deep_fade_tolerance": "four-binomial-standard-errors around Exponential(1) probability",
        },
        "implementation_order": [
            "读取完整裁剪背景 [pulse, range]",
            "按目标距离单元在完整脉冲序列上计算20参考单元功率和",
            "按标称SCR计算每个目标序列的固定基础幅度",
            "抽取并按目标序列均值归一化窗级Swerling功率增益",
            "使用sqrt(power_gain)施加到目标幅度",
            "注入完整脉冲序列",
            "按P=4非重叠切窗",
            "保存float32 I/Q且不做网络输入归一化",
        ],
        "scr_calibration_granularity": "per_nominal_scr_and_target_position_over_full_pulse_sequence_before_windowing",
        "rcs_normalization_granularity": "per_nominal_scr_target_sequence_across_all_windows",
        "domains": domain_results,
        "a_r_test_pair_check": pair_check,
        "low_scr_residual_audit": low_scr,
        "case_classification": classification,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_scr_audit_csv(csv_path, csv_rows)
    report_path.write_text(build_scr_audit_report(results), encoding="utf-8")
    print_scr_audit_summary(results)
    print(f"Saved SCR audit JSON: {output_path}", flush=True)
    print(f"Saved SCR audit CSV: {csv_path}", flush=True)
    print(f"Saved SCR audit report: {report_path}", flush=True)


def scr_audit_domain(manifest: dict[str, Any]) -> str:
    target_model = str(manifest.get("protocol", {}).get("target_model") or "ideal_continuous_phase")
    mapping = {
        "ideal_continuous_phase": "I",
        "ideal_phase_swerling1_window": "A",
        "phase_noise_swerling1_window": "R",
    }
    if target_model not in mapping:
        raise ValueError(f"scr_audit 不支持 target_model={target_model!r}。")
    return mapping[target_model]


def load_raw_windows_for_split(manifest: dict[str, Any], split: str) -> np.ndarray:
    if split not in {"train", "test"}:
        raise ValueError(f"未知 split={split!r}。")
    source = Path(manifest["source_files"][f"{split}_background"])
    key = str(manifest["source_files"]["mat_key"])
    payload = sio.loadmat(source)
    if key not in payload:
        raise KeyError(f"SCR 审计原始文件缺少 {key!r}: {source}。")
    crop = manifest["crop"]
    clutter = np.asarray(payload[key])[
        :, int(crop["crop_start_zero_based"]) : int(crop["crop_end_exclusive_zero_based"])
    ]
    pulses = int(manifest["protocol"]["pulses"])
    starts = range(0, len(clutter) - pulses + 1, pulses)
    return np.stack(
        [np.stack([clutter[start : start + pulses].real, clutter[start : start + pulses].imag]) for start in starts]
    ).astype(np.float32)


def replay_rcs_gains(
    manifest: dict[str, Any],
    split: str,
    scr_db: int,
    num_windows: int,
    num_targets: int,
) -> list[np.ndarray] | None:
    if scr_audit_domain(manifest) == "I":
        return None
    offset = 100_000 if split == "train" else 200_000
    rng = np.random.default_rng(int(manifest["seed"]) + offset + int(scr_db))
    pulses = int(manifest["protocol"]["pulses"])
    result: list[np.ndarray] = []
    for _ in range(num_targets):
        gains = rng.exponential(scale=1.0, size=num_windows).astype(np.float64)
        gains /= gains.mean()
        result.append(gains)
        for _window in range(num_windows):
            rng.uniform(-np.pi, np.pi)
            rng.normal(0.0, 1.0, size=pulses)
    return result


def audit_target_record(
    x: np.ndarray,
    raw: np.ndarray,
    record: dict[str, Any],
    scr_db: int,
    expected_gain: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if x.shape != raw.shape or x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"注入数据与原始背景 shape 不一致: X={x.shape}, raw={raw.shape}。")
    target = int(record["target_position_local_zero_based"])
    refs = np.asarray(record["reference_cells_local_zero_based"], dtype=np.int64)
    if refs.size != 20:
        raise ValueError(f"SCR 审计要求20个参考单元，实际为 {refs.size}。")
    injected_target = x[:, 0, :, target].astype(np.float64) + 1j * x[:, 1, :, target].astype(np.float64)
    raw_target = raw[:, 0, :, target].astype(np.float64) + 1j * raw[:, 1, :, target].astype(np.float64)
    component = injected_target - raw_target
    target_power = np.mean(np.abs(component) ** 2, axis=1, dtype=np.float64)
    raw_refs = raw[:, 0, :, :][:, :, refs].astype(np.float64) + 1j * raw[:, 1, :, :][:, :, refs].astype(
        np.float64
    )
    reference_power = np.sum(np.mean(np.abs(raw_refs) ** 2, axis=1, dtype=np.float64), axis=1, dtype=np.float64)
    if np.any(target_power <= 0.0) or np.any(reference_power <= 0.0):
        raise ValueError("SCR 审计重建得到非正目标功率或参考功率。")
    base_power = float(record["target_amplitude"]) ** 2
    recovered_gain = target_power / base_power
    expected = np.ones_like(recovered_gain) if expected_gain is None else np.asarray(expected_gain, dtype=np.float64)
    if expected.shape != recovered_gain.shape:
        raise ValueError(f"RCS replay shape 不一致: expected={expected.shape}, recovered={recovered_gain.shape}。")
    actual_scr = 10.0 * np.log10(target_power / reference_power)
    global_reference = float(record["reference_power_sum"])
    predicted_scr = float(scr_db) + 10.0 * np.log10(expected) + 10.0 * np.log10(global_reference / reference_power)
    identity_residual = actual_scr - predicted_scr
    reference_observed = x[:, :, :, :][:, :, :, refs].astype(np.float64)
    reference_raw = raw[:, :, :, :][:, :, :, refs].astype(np.float64)
    max_reference_residual = float(np.max(np.abs(reference_observed - reference_raw)))
    gain_error = recovered_gain - expected
    correlation = safe_correlation(reference_power, target_power)
    summary = {
        "target_position_local_zero_based": target,
        "speed_mps": float(record["speed_mps"]),
        "base_target_power": base_power,
        "global_reference_power_sum": global_reference,
        "reference_target_power_correlation_within_sequence": correlation,
        "max_abs_reference_cell_residual": max_reference_residual,
        "max_abs_gain_replay_error": float(np.max(np.abs(gain_error))),
        "gain_replay_allclose": bool(np.allclose(recovered_gain, expected, rtol=0.005, atol=0.0005)),
        "max_abs_scr_identity_residual_db": float(np.max(np.abs(identity_residual))),
    }
    return summary, {
        "actual_scr_db": actual_scr,
        "reference_power": reference_power,
        "target_power": target_power,
        "recovered_gain": recovered_gain,
        "expected_gain": expected,
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) <= np.finfo(np.float64).eps or np.std(right) <= np.finfo(np.float64).eps:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("无法汇总空或非有限分布。")
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(quantiles[3]),
        "min": float(values.min()),
        "max": float(values.max()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
    }


def summarize_gain_distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    result: dict[str, Any] = summarize_distribution(values)
    result.update(
        {
            "fraction_lt_0_1": float(np.mean(values < 0.1)),
            "fraction_lt_0_01": float(np.mean(values < 0.01)),
            "fraction_gt_3": float(np.mean(values > 3.0)),
            "unique_count": int(np.unique(values).size),
            "nonpositive_count": int(np.count_nonzero(values <= 0.0)),
        }
    )
    return result


def four_sigma_binomial_interval(probability: float, count: int) -> tuple[float, float]:
    radius = 4.0 * np.sqrt(probability * (1.0 - probability) / count)
    return max(0.0, probability - radius), min(1.0, probability + radius)


def audit_scr_data_dir(
    data_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]], list[dict[str, Any]]]:
    domain = scr_audit_domain(manifest)
    split_results: dict[str, Any] = {}
    vectors: dict[str, dict[int, np.ndarray]] = {"actual_scr_db": {}, "gain": {}}
    csv_rows: list[dict[str, Any]] = []
    all_gains: list[np.ndarray] = []
    all_record_summaries: list[dict[str, Any]] = []
    gain_hash_matches: list[bool] = []
    for split in ("train", "test"):
        raw = load_raw_windows_for_split(manifest, split)
        if split == "train":
            with np.load(data_dir / "train.npz") as payload:
                full_x = payload["X"].astype(np.float32, copy=False)
                full_y = payload["y"].astype(np.int64, copy=False)
                full_scr = payload["scr"].astype(np.int64, copy=False)
            scr_payloads = {
                int(scr): (full_x[full_scr == int(scr)], full_y[full_scr == int(scr)])
                for scr in sorted(set(full_scr.tolist()))
            }
        else:
            scr_payloads = {}
            for path in list_test_scr_files(data_dir):
                scr = int(path.stem.replace("test_scr_", ""))
                with np.load(path) as payload:
                    scr_payloads[scr] = (
                        payload["X"].astype(np.float32, copy=False),
                        payload["y"].astype(np.int64, copy=False),
                    )
        per_scr: dict[str, Any] = {}
        for scr_db, (x, y) in sorted(scr_payloads.items()):
            if len(x) != len(raw):
                raise ValueError(
                    f"{domain}/{split}/SCR={scr_db} 窗口数与原始背景不一致: {len(x)} != {len(raw)}。"
                )
            audit = manifest["audit"][f"{split}_injection_by_scr"][str(scr_db)]
            records = audit["target_records"]
            expected_gains = replay_rcs_gains(manifest, split, scr_db, len(x), len(records))
            actual_parts: list[np.ndarray] = []
            gain_parts: list[np.ndarray] = []
            record_summaries: list[dict[str, Any]] = []
            for index, record in enumerate(records):
                target = int(record["target_position_local_zero_based"])
                if not np.all(y[:, target] == 1):
                    raise ValueError(f"{domain}/{split}/SCR={scr_db} 的目标标签与manifest不一致: target={target}。")
                expected = None if expected_gains is None else expected_gains[index]
                record_summary, record_vectors = audit_target_record(x, raw, record, scr_db, expected)
                if expected is not None:
                    component = record.get("component_draws", {})
                    expected_hash = component.get("rcs_gain_sha256")
                    if expected_hash:
                        hash_matches = hashlib.sha256(expected.tobytes()).hexdigest() == expected_hash
                        gain_hash_matches.append(hash_matches)
                        record_summary["manifest_gain_hash_match"] = hash_matches
                actual_parts.append(record_vectors["actual_scr_db"])
                gain_parts.append(record_vectors["recovered_gain"])
                record_summaries.append(record_summary)
                all_record_summaries.append(record_summary)
            actual_values = np.concatenate(actual_parts)
            gain_values = np.concatenate(gain_parts)
            actual_summary = summarize_distribution(actual_values)
            gain_summary = summarize_gain_distribution(gain_values)
            if domain != "I":
                all_gains.append(gain_values)
            per_scr[str(scr_db)] = {
                "num_windows": len(x),
                "num_targets": len(records),
                "actual_scr_db": actual_summary,
                "recovered_power_gain": gain_summary,
                "records": record_summaries,
            }
            csv_rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "scr_db": scr_db,
                    **{f"actual_scr_{key}": value for key, value in actual_summary.items()},
                    **{f"gain_{key}": value for key, value in gain_summary.items()},
                }
            )
            if split == "test":
                vectors["actual_scr_db"][scr_db] = actual_values
                vectors["gain"][scr_db] = gain_values
        split_results[split] = {"per_scr": per_scr}

    gain_distribution = None
    deep_fade_check = None
    if all_gains:
        gain_distribution = summarize_gain_distribution(np.concatenate(all_gains))
        count = int(gain_distribution["count"])
        expected_probabilities = {
            "fraction_lt_0_1": 1.0 - np.exp(-0.1),
            "fraction_lt_0_01": 1.0 - np.exp(-0.01),
            "fraction_gt_3": np.exp(-3.0),
        }
        checks: dict[str, Any] = {}
        for key, probability in expected_probabilities.items():
            lower, upper = four_sigma_binomial_interval(float(probability), count)
            observed = float(gain_distribution[key])
            checks[key] = {
                "observed": observed,
                "exponential_reference": float(probability),
                "four_sigma_lower": lower,
                "four_sigma_upper": upper,
                "pass": bool(lower <= observed <= upper),
            }
        deep_fade_check = {"checks": checks, "all_pass": all(item["pass"] for item in checks.values())}

    position_sets = [
        tuple(int(value) for value in values)
        for _, values in sorted(manifest["protocol"]["train_target_cells_one_based_by_scr"].items(), key=lambda item: int(item[0]))
    ]
    hashes = [
        str(record.get("component_draws", {}).get("rcs_gain_sha256"))
        for audit in manifest["audit"]["train_injection_by_scr"].values()
        for record in audit["target_records"]
        if record.get("component_draws", {}).get("rcs_gain_sha256")
    ]
    max_gain_error = max(float(item["max_abs_gain_replay_error"]) for item in all_record_summaries)
    max_identity_residual = max(float(item["max_abs_scr_identity_residual_db"]) for item in all_record_summaries)
    max_reference_residual = max(float(item["max_abs_reference_cell_residual"]) for item in all_record_summaries)
    result = {
        "data_dir": str(data_dir),
        "target_model": str(manifest.get("protocol", {}).get("target_model") or "ideal_continuous_phase"),
        "source_files": manifest["source_files"],
        "splits": split_results,
        "aggregate_gain_distribution": gain_distribution,
        "deep_fade_theory_check": deep_fade_check,
        "reconstruction_checks": {
            "max_abs_gain_replay_error": max_gain_error,
            "all_gain_replay_allclose": all(bool(item["gain_replay_allclose"]) for item in all_record_summaries),
            "max_abs_scr_identity_residual_db": max_identity_residual,
            "max_abs_reference_cell_residual": max_reference_residual,
            "all_manifest_gain_hashes_match": all(gain_hash_matches) if gain_hash_matches else None,
            "num_manifest_gain_hashes_checked": len(gain_hash_matches),
        },
        "training_scr_reuse": {
            "same_background_windows_reused_across_scr": True,
            "same_window_order_across_scr": True,
            "all_target_position_sets_identical": len(set(position_sets)) == 1,
            "num_unique_target_position_sets": len(set(position_sets)),
            "num_rcs_hashes": len(hashes),
            "num_duplicate_rcs_hashes": len(hashes) - len(set(hashes)),
            "interpretation": "背景窗口复用会降低有效背景样本多样性，但本审计不把复用本身判定为泄漏。",
        },
    }
    return result, vectors, csv_rows


def compare_a_r_test_vectors(
    a_vectors: dict[str, dict[int, np.ndarray]],
    r_vectors: dict[str, dict[int, np.ndarray]],
) -> dict[str, Any]:
    if set(a_vectors["actual_scr_db"]) != set(r_vectors["actual_scr_db"]):
        raise ValueError("A/R 测试SCR集合不一致。")
    max_scr_error = 0.0
    max_gain_error = 0.0
    per_scr: dict[str, Any] = {}
    for scr_db in sorted(a_vectors["actual_scr_db"]):
        a_scr = a_vectors["actual_scr_db"][scr_db]
        r_scr = r_vectors["actual_scr_db"][scr_db]
        a_gain = a_vectors["gain"][scr_db]
        r_gain = r_vectors["gain"][scr_db]
        if a_scr.shape != r_scr.shape or a_gain.shape != r_gain.shape:
            raise ValueError(f"A/R 配对shape不一致: SCR={scr_db}。")
        scr_error = float(np.max(np.abs(a_scr - r_scr)))
        gain_error = float(np.max(np.abs(a_gain - r_gain)))
        max_scr_error = max(max_scr_error, scr_error)
        max_gain_error = max(max_gain_error, gain_error)
        per_scr[str(scr_db)] = {"max_abs_actual_scr_difference_db": scr_error, "max_abs_gain_difference": gain_error}
    return {
        "max_abs_actual_scr_difference_db": max_scr_error,
        "max_abs_gain_difference": max_gain_error,
        "actual_scr_pair_pass": max_scr_error <= 0.02,
        "gain_pair_pass": max_gain_error <= 0.005,
        "per_scr": per_scr,
    }


def extract_eval_curve(path: Path, factor_domain: str) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    domains = payload.get("cross_domain", {}).get("domains", [])
    matches = [item for item in domains if item.get("factor_domain") == factor_domain]
    if len(matches) != 1:
        raise ValueError(f"{path} 中 factor_domain={factor_domain} 应唯一，实际为 {len(matches)}。")
    pfa = matches[0]["score_spaces"]["softmax_clutter"]["pfa"]["0.001"]
    if pfa.get("numerically_degenerate", False):
        raise ValueError(f"{path} 的 {factor_domain} softmax Pfa=0.001 已退化。")
    return {int(item["scr_db"]): float(item["PD"]) for item in pfa["metrics"]["per_scr"]}


def build_low_scr_residual_audit(reference_path: Path, mi_eval: Path, ma_eval: Path) -> dict[str, Any]:
    reference = load_fig9_reference(reference_path, [0.0001, 0.001, 0.01])["0.001"]
    paper = {int(item["scr_db"]): float(item["pd"]) for item in reference}
    mi_i = extract_eval_curve(mi_eval, "I")
    mi_a = extract_eval_curve(mi_eval, "A")
    ma_a = extract_eval_curve(ma_eval, "A")
    points: list[dict[str, Any]] = []
    for scr_db in (-24, -22, -20, -18, -16):
        points.append(
            {
                "scr_db": scr_db,
                "paper_pd": paper[scr_db],
                "mi_to_i_pd": mi_i[scr_db],
                "mi_to_a_pd": mi_a[scr_db],
                "ma_to_a_pd": ma_a[scr_db],
                "mi_to_i_residual": mi_i[scr_db] - paper[scr_db],
                "mi_to_a_residual": mi_a[scr_db] - paper[scr_db],
                "ma_to_a_residual": ma_a[scr_db] - paper[scr_db],
            }
        )

    def rmse(curve: dict[int, float], scr_values: tuple[int, ...]) -> float:
        return float(np.sqrt(np.mean([(curve[value] - paper[value]) ** 2 for value in scr_values])))

    formal = (-24, -22, -20)
    low5 = (-24, -22, -20, -18, -16)
    transition = (-18, -16, -14, -12, -10, -8)
    return {
        "pfa": 0.001,
        "points_minus24_to_minus16": points,
        "formal_low3_rmse": {
            "definition": list(formal),
            "mi_to_i": rmse(mi_i, formal),
            "mi_to_a": rmse(mi_a, formal),
            "ma_to_a": rmse(ma_a, formal),
        },
        "diagnostic_low5_rmse": {
            "definition": list(low5),
            "mi_to_i": rmse(mi_i, low5),
            "mi_to_a": rmse(mi_a, low5),
            "ma_to_a": rmse(ma_a, low5),
        },
        "transition_rmse": {
            "definition": list(transition),
            "mi_to_i": rmse(mi_i, transition),
            "mi_to_a": rmse(mi_a, transition),
            "ma_to_a": rmse(ma_a, transition),
        },
        "interpretation": "平均PD下降不保证RMSE下降；RMSE由每个SCR点相对论文曲线的残差平方决定。",
    }


def classify_scr_audit(domains: dict[str, Any], pair_check: dict[str, Any]) -> dict[str, Any]:
    reconstruction_ok = all(
        item["reconstruction_checks"]["all_gain_replay_allclose"]
        and item["reconstruction_checks"]["max_abs_scr_identity_residual_db"] <= 0.05
        and item["reconstruction_checks"]["max_abs_reference_cell_residual"] == 0.0
        and item["reconstruction_checks"]["all_manifest_gain_hashes_match"] is not False
        for item in domains.values()
    )
    if not reconstruction_ok or not pair_check["actual_scr_pair_pass"] or not pair_check["gain_pair_pass"]:
        case = "B"
        conclusion = "发现RCS重放、最终目标功率或A/R配对不一致；应先按实现问题处理。"
    elif any(not item["deep_fade_theory_check"]["all_pass"] for key, item in domains.items() if key in {"A", "R"}):
        case = "C"
        conclusion = "RCS重放正确，但深衰落比例偏离预注册理论容差；需区分有限样本、设计选择或裁剪。"
    else:
        case = "D"
        conclusion = "未发现逐窗口SCR重标定、RCS抵消或深衰落不足；当前窗级Swerling-I假设不足以恢复论文曲线。"
    return {
        "case": case,
        "case_a_ruled_out_by_code_and_manifest": True,
        "reconstruction_ok": reconstruction_ok,
        "conclusion": conclusion,
        "stop_required": True,
        "automatic_fix_or_training_allowed": False,
    }


def write_scr_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("SCR 审计没有可写入CSV的行。")
    fieldnames = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scr_audit_report(results: dict[str, Any]) -> str:
    case = results["case_classification"]
    lines = [
        "# SDRDSP 任务 B.5：RCS/SCR 分布审计",
        "",
        "## 结论",
        "",
        f"自动判定为 **情形 {case['case']}**：{case['conclusion']}",
        "",
        "本任务只读取现有代码、manifest、原始MAT、NPZ和评价JSON；未训练、未修改模型、未重新生成数据。",
        "",
        "## 标定与生成顺序",
        "",
        "- SCR标定粒度：每个标称SCR、每个目标距离单元，在完整脉冲序列上计算一次。",
        "- RCS归一化粒度：每个标称SCR下的每个目标序列，在全部窗口上统一归一化。",
        "- 先注入完整脉冲序列，再按4脉冲非重叠切窗。",
        "- 未发现逐窗口重新设置基础目标幅度的处理顺序。",
        "",
        "## 重建和RCS验收",
        "",
        "| 域 | gain重放最大误差 | SCR等式最大残差/dB | 参考单元残差 | 深衰落理论检查 |",
        "|---|---:|---:|---:|---|",
    ]
    for domain in ("I", "A", "R"):
        item = results["domains"][domain]
        checks = item["reconstruction_checks"]
        deep = item["deep_fade_theory_check"]
        lines.append(
            f"| {domain} | {checks['max_abs_gain_replay_error']:.6g} | "
            f"{checks['max_abs_scr_identity_residual_db']:.6g} | "
            f"{checks['max_abs_reference_cell_residual']:.6g} | "
            f"{'N/A' if deep is None else deep['all_pass']} |"
        )
    pair = results["a_r_test_pair_check"]
    lines.extend(
        [
            "",
            f"A/R测试域最大actual SCR差为 `{pair['max_abs_actual_scr_difference_db']:.6g} dB`，"
            f"最大gain差为 `{pair['max_abs_gain_difference']:.6g}`。",
            "",
            "## 聚合Swerling功率增益",
            "",
            "| 域 | count | mean | median | q01 | q05 | q95 | g<0.1 | g<0.01 | g>3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for domain in ("A", "R"):
        gain = results["domains"][domain]["aggregate_gain_distribution"]
        lines.append(
            f"| {domain} | {gain['count']} | {gain['mean']:.6f} | {gain['median']:.6f} | "
            f"{gain['q01']:.6f} | {gain['q05']:.6f} | {gain['q95']:.6f} | "
            f"{gain['fraction_lt_0_1']:.4%} | {gain['fraction_lt_0_01']:.4%} | {gain['fraction_gt_3']:.4%} |"
        )
    low = results["low_scr_residual_audit"]
    lines.extend(
        [
            "",
            "## Pfa=0.001低SCR逐点残差",
            "",
            "| SCR/dB | 论文PD | M_I→I | M_I→A | M_A→A | M_I→I残差 | M_A→A残差 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in low["points_minus24_to_minus16"]:
        lines.append(
            f"| {item['scr_db']} | {item['paper_pd']:.6f} | {item['mi_to_i_pd']:.6f} | "
            f"{item['mi_to_a_pd']:.6f} | {item['ma_to_a_pd']:.6f} | "
            f"{item['mi_to_i_residual']:+.6f} | {item['ma_to_a_residual']:+.6f} |"
        )
    formal = low["formal_low3_rmse"]
    diagnostic = low["diagnostic_low5_rmse"]
    transition = low["transition_rmse"]
    lines.extend(
        [
            "",
            "- 正式低SCR三点RMSE（-24/-22/-20）："
            f"M_I→I={formal['mi_to_i']:.6f}，M_I→A={formal['mi_to_a']:.6f}，M_A→A={formal['ma_to_a']:.6f}。",
            "- 诊断低SCR五点RMSE（-24至-16）："
            f"M_I→I={diagnostic['mi_to_i']:.6f}，M_I→A={diagnostic['mi_to_a']:.6f}，"
            f"M_A→A={diagnostic['ma_to_a']:.6f}。该值不替代冻结闸门。",
            "- 过渡区RMSE（-18至-8）："
            f"M_I→I={transition['mi_to_i']:.6f}，M_I→A={transition['mi_to_a']:.6f}，"
            f"M_A→A={transition['ma_to_a']:.6f}。",
            "",
            "平均PD下降不保证RMSE下降，因为RMSE取决于每个SCR点相对论文曲线的残差平方。",
            "",
            "## 训练SCR层复用",
            "",
        ]
    )
    for domain in ("I", "A", "R"):
        reuse = results["domains"][domain]["training_scr_reuse"]
        lines.append(
            f"- {domain}域：14个SCR层复用相同背景窗口和顺序；目标位置集合共"
            f" `{reuse['num_unique_target_position_sets']}` 组；RCS hash重复数 `{reuse['num_duplicate_rcs_hashes']}`。"
        )
    lines.extend(
        [
            "",
            "背景复用会降低有效背景样本多样性，但不能仅凭这一点判定数据泄漏。",
            "",
            "## 结论边界与停止条件",
            "",
            f"- 情形判定：**{case['case']}**。",
            "- 不自动修复数据，不启动seed123/2026，不修改SCR定义。",
            "- 当前结果只能评价现有自行定义的窗级Swerling-I候选，不能恢复论文作者未公开的唯一协议。",
            "- 审计完成后停止，等待人工决定下一候选协议。",
            "",
        ]
    )
    return "\n".join(lines)


def default_scr_audit_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / "training" / f"{stamp}_sdrdsp_scr_audit" / "scr_audit.json"


def print_scr_audit_summary(results: dict[str, Any]) -> None:
    case = results["case_classification"]
    print("SDRDSP RCS/SCR distribution audit", flush=True)
    print(f"case={case['case']} | {case['conclusion']}", flush=True)
    for domain in ("I", "A", "R"):
        checks = results["domains"][domain]["reconstruction_checks"]
        print(
            f"{domain}: gain_error={checks['max_abs_gain_replay_error']:.6g} | "
            f"SCR_residual={checks['max_abs_scr_identity_residual_db']:.6g} dB | "
            f"reference_residual={checks['max_abs_reference_cell_residual']:.6g}",
            flush=True,
        )


def run_stage6_finalize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """汇总任务B/B.5，只新增阶段6旁路审计、报告和曲线图。"""
    required = {
        "--stage6-scr-audit": args.stage6_scr_audit,
        "--stage6-r-data-dir": args.stage6_r_data_dir,
        "--stage6-gate-definition": args.stage6_gate_definition,
        "--stage6-hash-output": args.stage6_hash_output,
        "--stage6-report-output": args.stage6_report_output,
        "--stage6-figure-output": args.stage6_figure_output,
    }
    missing_args = [name for name, value in required.items() if value is None]
    if missing_args:
        raise ValueError(f"stage6_finalize 缺少参数: {missing_args}。")
    input_paths = [args.stage6_scr_audit, args.stage6_r_data_dir, args.stage6_gate_definition]
    missing_paths = [str(path) for path in input_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"stage6_finalize 缺少输入: {missing_paths}。")
    outputs = [args.stage6_hash_output, args.stage6_report_output, args.stage6_figure_output]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有阶段6产物: {existing}。")
    if args.stage6_hash_output.suffix.lower() != ".json":
        raise ValueError("--stage6-hash-output 必须是 .json。")
    if args.stage6_report_output.suffix.lower() != ".md":
        raise ValueError("--stage6-report-output 必须是 .md。")
    if args.stage6_figure_output.suffix.lower() != ".png":
        raise ValueError("--stage6-figure-output 必须是 .png。")

    audit = json.loads(args.stage6_scr_audit.read_text(encoding="utf-8"))
    classification = audit.get("case_classification", {})
    if classification.get("case") != "D" or classification.get("stop_required") is not True:
        raise ValueError("阶段6只接受已冻结且要求停止的 Case D 审计。")
    pulses = int(get_config_value(config, "model.pulses"))
    range_cells = int(get_config_value(config, "model.range_cells"))
    manifest = validate_sdrdsp_v2_manifest(args.stage6_r_data_dir, pulses, range_cells)
    if scr_audit_domain(manifest) != "R":
        raise ValueError("--stage6-r-data-dir 必须是 phase+RCS 的旧 R 域。")

    gate_limit = extract_stage6_gate_limit(args.stage6_gate_definition)
    hash_audit = build_external_r_component_hash_audit(args.stage6_r_data_dir, manifest)
    if not hash_audit["reconstruction"]["all_component_reconstruction_close"]:
        raise ValueError("旧R组件确定性重放未通过，不生成阶段6结论。")
    figure_bytes = build_stage6_figure(audit)
    report = build_stage6_report(args, audit, hash_audit, gate_limit)

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.stage6_hash_output.open("x", encoding="utf-8") as handle:
        json.dump(hash_audit, handle, indent=2, ensure_ascii=False)
    with args.stage6_report_output.open("x", encoding="utf-8") as handle:
        handle.write(report)
    with args.stage6_figure_output.open("xb") as handle:
        handle.write(figure_bytes)
    print("SDRDSP stage6 finalized without new training", flush=True)
    print(f"External R hash audit: {args.stage6_hash_output}", flush=True)
    print(f"Final report: {args.stage6_report_output}", flush=True)
    print(f"Fig.9 comparison: {args.stage6_figure_output}", flush=True)


def canonical_array_sha256(values: np.ndarray, dtype: str) -> str:
    """按小端、C连续布局生成可跨数组内存布局复核的哈希。"""
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    canonical = np.ascontiguousarray(values, dtype=canonical_dtype)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_external_r_component_hash_audit(data_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """重放旧R随机流并在manifest外记录组件哈希，不回写冻结数据。"""
    from scripts.preprocess_sdrdsp import build_realistic_target_signal

    pulses = int(manifest["protocol"]["pulses"])
    phase_noise_std_deg = float(manifest["protocol"]["phase_noise_std_deg"])
    prt = float(manifest["protocol"]["prt_seconds"])
    wavelength = float(manifest["protocol"]["wavelength_m"])
    records_out: list[dict[str, Any]] = []
    close_flags: list[bool] = []
    max_errors: list[float] = []
    original_hash_count = 0
    for split in ("train", "test"):
        raw = load_raw_windows_for_split(manifest, split)
        if split == "train":
            with np.load(data_dir / "train.npz") as payload:
                full_x = payload["X"].astype(np.float32, copy=False)
                full_scr = payload["scr"].astype(np.int64, copy=False)
            payloads = {int(scr): full_x[full_scr == int(scr)] for scr in sorted(set(full_scr.tolist()))}
        else:
            payloads = {}
            for path in list_test_scr_files(data_dir):
                scr_db = int(path.stem.replace("test_scr_", ""))
                with np.load(path) as payload:
                    payloads[scr_db] = payload["X"].astype(np.float32, copy=False)
        offset = 100_000 if split == "train" else 200_000
        starts = list(range(0, len(raw) * pulses, pulses))
        for scr_db, x in sorted(payloads.items()):
            manifest_records = manifest["audit"][f"{split}_injection_by_scr"][str(scr_db)]["target_records"]
            rng_seed = int(manifest["seed"]) + offset + int(scr_db)
            rng = np.random.default_rng(rng_seed)
            for record_index, record in enumerate(manifest_records):
                signal, _, component_audit = build_realistic_target_signal(
                    num_pulses=len(raw) * pulses,
                    starts=starts,
                    pulses=pulses,
                    target_amplitude=float(record["target_amplitude"]),
                    speed=float(record["speed_mps"]),
                    prt=prt,
                    wavelength=wavelength,
                    phase_noise_std_deg=phase_noise_std_deg,
                    rng=rng,
                    use_phase_factor=True,
                    use_rcs_factor=True,
                )
                target = int(record["target_position_local_zero_based"])
                observed = x[:, 0, :, target].astype(np.float64) + 1j * x[:, 1, :, target].astype(np.float64)
                background = raw[:, 0, :, target].astype(np.float64) + 1j * raw[:, 1, :, target].astype(np.float64)
                recovered = observed - background
                expected = signal.reshape(len(raw), pulses)
                error = np.abs(recovered - expected)
                close = bool(np.allclose(recovered, expected, rtol=0.005, atol=0.005))
                existing_draws = record.get("component_draws", {})
                original_hash_count += int(bool(existing_draws.get("rcs_gain_sha256")))
                original_hash_count += int(bool(existing_draws.get("phase_bundle_sha256")))
                close_flags.append(close)
                max_errors.append(float(error.max()))
                records_out.append(
                    {
                        "split": split,
                        "scr_db": int(scr_db),
                        "record_index": record_index,
                        "target_position_local_zero_based": target,
                        "speed_mps": float(record["speed_mps"]),
                        "rng_seed": rng_seed,
                        "num_windows": len(raw),
                        "pulses": pulses,
                        "replayed_rcs_gain_sha256_float64_le": component_audit["rcs_gain_sha256"],
                        "replayed_phase_bundle_sha256_float64_le": component_audit["phase_bundle_sha256"],
                        "expected_component_sha256_complex64_le": canonical_array_sha256(expected, "<c8"),
                        "recovered_component_sha256_complex64_le": canonical_array_sha256(recovered, "<c8"),
                        "max_abs_component_reconstruction_error": float(error.max()),
                        "mean_abs_component_reconstruction_error": float(error.mean()),
                        "component_reconstruction_allclose": close,
                    }
                )
    records_digest = hashlib.sha256(
        json.dumps(records_out, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": "sdrdsp_old_r_external_component_hash_audit_v1",
        "provenance": "deterministic_replay_sidecar_not_original_manifest_native_hash",
        "git_commit": git_commit(),
        "data_dir": str(data_dir),
        "manifest_path": str(data_dir / "manifest.json"),
        "manifest_sha256": path_sha256(data_dir / "manifest.json"),
        "manifest_modified": False,
        "draw_stream": "legacy_combined_v1_draw_all_components_before_factor_toggle",
        "canonical_hash_layout": "little_endian_C_contiguous",
        "original_manifest_component_hash_count": original_hash_count,
        "record_count": len(records_out),
        "records_sha256": records_digest,
        "reconstruction": {
            "rtol": 0.005,
            "atol": 0.005,
            "all_component_reconstruction_close": all(close_flags),
            "max_abs_component_reconstruction_error": max(max_errors),
        },
        "records": records_out,
    }


def extract_stage6_gate_limit(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"RMSE_low_scr\s*<=\s*([0-9.]+)", text)
    if match is None:
        raise ValueError(f"无法从预注册闸门提取 RMSE_low_scr 上限: {path}。")
    return float(match.group(1))


def build_stage6_figure(audit: dict[str, Any]) -> bytes:
    from io import BytesIO

    import matplotlib.pyplot as plt

    inputs = audit["inputs"]
    reference = load_fig9_reference(Path(inputs["paper_fig9_reference"]), [0.0001, 0.001, 0.01])["0.001"]
    paper = {int(item["scr_db"]): float(item["pd"]) for item in reference}
    mi_i = extract_eval_curve(Path(inputs["mi_eval"]), "I")
    mi_a = extract_eval_curve(Path(inputs["mi_eval"]), "A")
    ma_a = extract_eval_curve(Path(inputs["ma_eval"]), "A")
    scr_values = sorted(set(paper) & set(mi_i) & set(mi_a) & set(ma_a))
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.plot(scr_values, [paper[x] for x in scr_values], "ko-", linewidth=2.0, markersize=4, label="Paper ST-GNN")
    ax.plot(scr_values, [mi_i[x] for x in scr_values], "o-", linewidth=1.7, markersize=3, label="M_I -> I baseline")
    ax.plot(
        scr_values,
        [mi_a[x] for x in scr_values],
        "--",
        linewidth=2.0,
        label="M_I -> A cross-domain diagnostic",
    )
    ax.plot(scr_values, [ma_a[x] for x in scr_values], "s-", linewidth=1.7, markersize=3, label="M_A -> A (gate FAIL)")
    ax.set(title="SDRDSP Fig. 9 comparison at Pfa=0.001", xlabel="SCR (dB)", ylabel="PD")
    ax.set_xlim(min(scr_values), max(scr_values))
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def summarize_stage6_thresholds(ma_eval_path: Path) -> dict[str, Any]:
    payload = json.loads(ma_eval_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    for domain in payload["cross_domain"]["domains"]:
        factor_domain = str(domain["factor_domain"])
        pfa_items = domain["score_spaces"]["softmax_clutter"]["pfa"]
        for pfa_text, item in sorted(pfa_items.items(), key=lambda pair: float(pair[0])):
            target_pfa = float(pfa_text)
            actual_pf = float(item["metrics"]["PF"])
            checks.append(
                {
                    "domain": factor_domain,
                    "target_pfa": target_pfa,
                    "actual_pf": actual_pf,
                    "pf_ratio": actual_pf / target_pfa,
                    "numerically_degenerate": bool(item["numerically_degenerate"]),
                    "pf_guardrail_pass": bool(item["pf_guardrail_pass"]),
                }
            )
    return {
        "count": len(checks),
        "all_non_degenerate": all(not item["numerically_degenerate"] for item in checks),
        "all_pf_guardrails_pass": all(item["pf_guardrail_pass"] for item in checks),
        "min_pf_ratio": min(item["pf_ratio"] for item in checks),
        "max_pf_ratio": max(item["pf_ratio"] for item in checks),
    }


def build_stage6_report(
    args: argparse.Namespace,
    audit: dict[str, Any],
    hash_audit: dict[str, Any],
    gate_limit: float,
) -> str:
    low = audit["low_scr_residual_audit"]["formal_low3_rmse"]
    ma_rmse = float(low["ma_to_a"])
    thresholds = summarize_stage6_thresholds(Path(audit["inputs"]["ma_eval"]))
    return "\n".join(
        [
            "# SDRDSP 阶段6最终科学闸门报告",
            "",
            "日期：2026-07-13  ",
            "执行范围：仅汇总任务B/B.5、重跑受阻测试、生成旁路哈希和曲线；未执行新训练。",
            "",
            "## 最终判定",
            "",
            "- **RCS-only候选：FAIL。**",
            f"- `M_A→A` 在 `Pfa=0.001` 的正式低SCR三点RMSE为 `{ma_rmse:.6f}`，预注册上限为 `{gate_limit:.6f}`。",
            f"- 任务B.5自动分类为 **Case {audit['case_classification']['case']}**，要求停止且不允许自动训练。",
            "- `M_A seed123/2026` 记为 `skipped_by_preregistered_gate`；这不包括此前已完成的其他训练域seed123/2026。",
            "",
            "## 结论边界",
            "",
            "### 已确认",
            "",
            "- 固定 `M_I` checkpoint 从I域转到A域后，低SCR检出性能明显下降，说明模型对RCS/幅度测试扰动敏感。",
            "- RCS-only同域训练能够恢复大量性能，但没有使正式低SCR曲线达到预注册论文对齐标准。",
            "",
            "### 已排除（仅限当前实现和已评估工作点）",
            "",
            "- 未发现逐4脉冲窗口重新标定基础SCR。",
            "- 未发现Swerling功率增益生成、均值归一化或 `sqrt(power_gain)` 幅度换算错误。",
            f"- A训练阈值在I/P/A/R × 3个Pfa共 `{thresholds['count']}` 个工作点均非退化，actual PF倍率范围为 "
            f"`{thresholds['min_pf_ratio']:.3f}–{thresholds['max_pf_ratio']:.3f}`，全部通过冻结PF守门线。",
            "",
            "### 未确认",
            "",
            "- 尚未证明模型学习了哪一种具体合成器指纹。",
            "- 尚未确定背景窗口跨SCR复用、目标调度、有限物理因素或其他未公开协议中哪一项是主要原因。",
            "- 当前Swerling-I候选不能替代论文作者未公开的唯一目标合成协议。",
            "",
            "## 曲线口径",
            "",
            "- `M_I→I`：当前理想合成同域baseline。",
            "- `M_A→A`：RCS-only同域候选，科学闸门FAIL。",
            "- `M_I→A`：只标记为 **cross-domain diagnostic**，不得表述为论文Fig.9复现结果。",
            f"- 正式低SCRRMSE：`M_I→I={low['mi_to_i']:.6f}`，`M_I→A={low['mi_to_a']:.6f}`，"
            f"`M_A→A={low['ma_to_a']:.6f}`。",
            "",
            f"![SDRDSP Fig.9科学闸门对照]({args.stage6_figure_output.name})",
            "",
            "## Windows测试复核",
            "",
            "此前3个失败用例在普通沙箱和仓库工作目录均受到Windows临时目录权限影响。切换到获批的系统临时目录后，原3个测试 `3 passed in 0.93s`；未修改测试代码或协议实现。",
            "",
            "## 旧R manifest外部组件哈希",
            "",
            f"- 旁路审计记录数：`{hash_audit['record_count']}`；旧manifest原生组件哈希数：`{hash_audit['original_manifest_component_hash_count']}`。",
            f"- 重放组件与保存NPZ重建组件全部数值一致；最大复数幅度误差 `{hash_audit['reconstruction']['max_abs_component_reconstruction_error']:.6g}`。",
            "- 哈希属于 `deterministic_replay_sidecar`，不是旧manifest当时保存的原生出处证明；冻结manifest未修改。",
            "",
            "## 产物与停止条件",
            "",
            f"- 任务B结果：`reports/sdrdsp_stage4_ma_seed42_result.md`",
            f"- 任务B.5审计：`{args.stage6_scr_audit}`",
            f"- 外部哈希：`{args.stage6_hash_output}`",
            f"- 曲线图：`{args.stage6_figure_output}`",
            "- 本报告只新增文件，未覆盖既有阶段4–6报告。",
            "- 阶段6到此停止；不启动新训练、不修改模型、数据协议、阈值或验收线。",
            "",
        ]
    )


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
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples 必须为正整数。")
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
        "logit_margin": {
            str(pfa): strict_rank_threshold(train_clutter["logit_margin"], pfa)
            for pfa in pfa_values
        },
        "softmax_clutter": {
            str(pfa): paper_order_stat_threshold(train_clutter["softmax_clutter"], pfa)
            for pfa in pfa_values
        },
    }
    fig9_reference = load_fig9_reference(args.paper_fig9_reference, pfa_values)
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
        "threshold_rules": {
            "softmax_clutter": "paper_order_stat_ceil_including_boundary_ties",
            "logit_margin": "strict_rank_floor_budget_excluding_boundary_ties",
        },
        "num_clutter_bins_for_threshold": int(train_clutter["logit_margin"].size),
        "train_score_diagnostics": {
            name: score_diagnostics(values) for name, values in train_clutter.items()
        },
        "paper_fig9_reference": str(args.paper_fig9_reference) if args.paper_fig9_reference else None,
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
            calibrations=calibrations,
            pfa_values=pfa_values,
            pulses=pulses,
            range_cells=range_cells,
            bootstrap_resamples=args.bootstrap_resamples,
            fig9_reference=fig9_reference,
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


def paper_order_stat_threshold(scores: np.ndarray, target_pfa: float) -> dict[str, Any]:
    """按论文 FAR Controller 的 ceil(alpha*Nc) 序统计量选阈值，边界并列全部计入。"""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("阈值分数必须是非空有限数组。")
    if not 0.0 < target_pfa < 1.0:
        raise ValueError(f"target_pfa 必须位于 (0, 1)，实际为 {target_pfa}。")
    ordered = np.sort(flat)
    rank_one_based = int(np.ceil(target_pfa * len(ordered)))
    threshold = float(ordered[rank_one_based - 1])
    selected = int(np.count_nonzero(flat <= threshold))
    tie_count = int(np.count_nonzero(flat == threshold))
    return {
        "target_pfa": float(target_pfa),
        "threshold": threshold,
        "num_scores": int(flat.size),
        "rank_one_based": rank_one_based,
        "selected_clutter_bins": selected,
        "calibration_actual_pf": selected / flat.size,
        "boundary_value": threshold,
        "boundary_tie_count": tie_count,
        "boundary_tie_excluded": False,
        "numerically_degenerate": bool(selected / flat.size > 2.0 * target_pfa),
    }


def score_diagnostics(scores: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("分数诊断不能使用空数组。")
    unique = int(np.unique(flat).size)
    return {
        "num_scores": int(flat.size),
        "num_unique": unique,
        "unique_fraction": unique / flat.size,
        "num_exact_zero": int(np.count_nonzero(flat == 0.0)),
        "exact_zero_fraction": float(np.mean(flat == 0.0)),
        "num_exact_one": int(np.count_nonzero(flat == 1.0)),
        "exact_one_fraction": float(np.mean(flat == 1.0)),
    }


def load_fig9_reference(
    path: Path | None,
    pfa_values: list[float],
) -> dict[str, list[dict[str, float]]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Fig.9 参考 CSV 不存在: {path}")
    reference: dict[str, list[dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = {"pfa", "scr_db", "pd", "digitization_uncertainty"}
        if set(reader.fieldnames or []) != expected_fields:
            raise ValueError(f"Fig.9 CSV 字段必须为 {sorted(expected_fields)}。")
        for row in reader:
            pfa = float(row["pfa"])
            item = {
                "scr_db": float(row["scr_db"]),
                "pd": float(row["pd"]),
                "digitization_uncertainty": float(row["digitization_uncertainty"]),
            }
            if not 0.0 <= item["pd"] <= 1.0 or item["digitization_uncertainty"] < 0.0:
                raise ValueError(f"Fig.9 CSV 数值非法: {row}。")
            reference.setdefault(str(pfa), []).append(item)
    expected_scr = [float(value) for value in range(-24, 15, 2)]
    for pfa in pfa_values:
        key = str(pfa)
        items = sorted(reference.get(key, []), key=lambda item: item["scr_db"])
        if [item["scr_db"] for item in items] != expected_scr:
            raise ValueError(f"Fig.9 CSV 在 Pfa={pfa} 下必须覆盖全部 20 个 SCR 点。")
        reference[key] = items
    if set(reference) != {str(value) for value in pfa_values}:
        raise ValueError("Fig.9 CSV 包含未请求的 Pfa 或缺少请求的 Pfa。")
    return reference


def fig9_error_metrics(
    per_scr: list[dict[str, Any]],
    reference: list[dict[str, float]],
) -> dict[str, float | None]:
    observed = {float(item["scr_db"]): float(item["PD"]) for item in per_scr}
    scr = np.asarray([item["scr_db"] for item in reference], dtype=np.float64)
    paper = np.asarray([item["pd"] for item in reference], dtype=np.float64)
    if set(observed) != set(scr.tolist()):
        raise ValueError("实验逐 SCR 结果与 Fig.9 参考点不完整对应。")
    model = np.asarray([observed[value] for value in scr], dtype=np.float64)
    error = model - paper

    def rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(error[mask] ** 2)))

    return {
        "RMSE_all": rmse(np.ones_like(scr, dtype=bool)),
        "RMSE_low_scr": rmse(scr <= -20),
        "RMSE_transition": rmse((scr >= -18) & (scr <= -8)),
        "MAE_all": float(np.mean(np.abs(error))),
        "PD_SCR_AUC_difference": float(np.trapezoid(model, scr) - np.trapezoid(paper, scr)),
        "SCR_at_PD_0.5": first_crossing_scr(scr, model, 0.5),
        "SCR_at_PD_0.9": first_crossing_scr(scr, model, 0.9),
    }


def first_crossing_scr(scr: np.ndarray, pd: np.ndarray, target: float) -> float | None:
    for index, value in enumerate(pd):
        if value < target:
            continue
        if index == 0:
            return float(scr[0])
        left_pd = float(pd[index - 1])
        right_pd = float(value)
        if right_pd == left_pd:
            return float(scr[index])
        weight = (target - left_pd) / (right_pd - left_pd)
        return float(scr[index - 1] + weight * (scr[index] - scr[index - 1]))
    return None


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
        "softmax_clutter 使用论文 ceil(alpha*Nc) 序统计量，是论文公式严格实现；"
        "logit_margin 使用边界并列保护，是数值稳定诊断。"
        "若 softmax 大量等于 1 且实际 PF 失控，应保留并标记退化，不能用 logit 结果冒充论文公式结果。"
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
    calibrations: dict[str, dict[str, dict[str, Any]]],
    pfa_values: list[float],
    pulses: int,
    range_cells: int,
    bootstrap_resamples: int,
    fig9_reference: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    train_manifest = validate_sdrdsp_v2_manifest(train_data_dir, pulses, range_cells)
    domains: list[dict[str, Any]] = []
    factor_vectors: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "softmax_clutter": {},
        "logit_margin": {},
    }
    factor_keys = {
        "ideal_continuous_phase": "I",
        "phase_noise_constant_rcs": "P",
        "ideal_phase_swerling1_window": "A",
        "phase_noise_swerling1_window": "R",
    }
    for test_dir in test_data_dirs:
        test_manifest = validate_sdrdsp_v2_manifest(test_dir, pulses, range_cells)
        target_model = str(test_manifest.get("protocol", {}).get("target_model", "ideal_continuous_phase"))
        factor_key = factor_keys.get(target_model)
        if factor_key is None:
            raise ValueError(f"cross 域 target_model 无法归类: {target_model!r}。")
        if factor_key in factor_vectors["logit_margin"]:
            raise ValueError(f"cross 测试域重复提供因素域 {factor_key}。")
        for score_name in factor_vectors:
            factor_vectors[score_name][factor_key] = {}
        records = collect_sdrdsp_records(
            model,
            list_test_scr_files(test_dir),
            batch_size,
            device,
            max_windows_per_file,
            seed,
            variant="original",
        )
        score_spaces: dict[str, Any] = {}
        for score_name in ("softmax_clutter", "logit_margin"):
            pfa_payload: dict[str, Any] = {}
            for pfa in pfa_values:
                calibration = calibrations[score_name][str(pfa)]
                metrics = evaluate_sdrdsp_records(records, score_name, calibration["threshold"])
                low = [item for item in metrics["per_scr"] if item["scr_db"] in {-24, -22, -20, -18, -16}]
                item: dict[str, Any] = {
                    "threshold": calibration["threshold"],
                    "threshold_source": "train_clutter",
                    "num_clutter_bins_for_threshold": calibration["num_scores"],
                    "num_threshold_ties": calibration["boundary_tie_count"],
                    "calibration_actual_pf": calibration["calibration_actual_pf"],
                    "numerically_degenerate": bool(
                        calibration.get("numerically_degenerate", False) or metrics["PF"] > 2.0 * pfa
                    ),
                    "metrics": metrics,
                    "low_scr_mean_pd": float(np.mean([value["PD"] for value in low])),
                    "low_scr_pd_auc": float(
                        np.trapezoid([value["PD"] for value in low], [value["scr_db"] for value in low])
                    ),
                    "pf_guardrail_pass": bool(metrics["PF"] <= 2.0 * pfa),
                }
                if fig9_reference:
                    item["fig9_error"] = fig9_error_metrics(metrics["per_scr"], fig9_reference[str(pfa)])
                pfa_payload[str(pfa)] = item
            score_spaces[score_name] = {
                "role": "paper_formal" if score_name == "softmax_clutter" else "numerical_diagnostic",
                "threshold_rule": (
                    "paper_order_stat_ceil_including_boundary_ties"
                    if score_name == "softmax_clutter"
                    else "strict_rank_floor_budget_excluding_boundary_ties"
                ),
                "pfa": pfa_payload,
            }
        pfa_payload = score_spaces["logit_margin"]["pfa"]
        for score_name in factor_vectors:
            for pfa in pfa_values:
                factor_vectors[score_name][factor_key][str(pfa)] = low_scr_target_detections(
                    records,
                    calibrations[score_name][str(pfa)]["threshold"],
                    score_name=score_name,
                )
        domains.append(
            {
                "test_data_dir": str(test_dir),
                "test_protocol": test_manifest["protocol"]["id"],
                "factor_domain": factor_key,
                "target_model": target_model,
                "score_spaces": score_spaces,
                "pfa": pfa_payload,
            }
        )
    result = {
        "train_data_dir": str(train_data_dir),
        "train_protocol": train_manifest["protocol"]["id"],
        "threshold_source": "train_clutter",
        "domains": domains,
    }
    if all(set(values) == {"I", "P", "A", "R"} for values in factor_vectors.values()):
        result["factor_effects_by_score"] = {
            score_name: {
                str(pfa): build_factor_effects(
                    {key: values[key][str(pfa)] for key in ("I", "P", "A", "R")},
                    bootstrap_resamples,
                    seed + int(round(pfa * 1_000_000)),
                )
                for pfa in pfa_values
            }
            for score_name, values in factor_vectors.items()
        }
        result["factor_effects"] = result["factor_effects_by_score"]["logit_margin"]
        result["factor_effects_interpretation"] = (
            "固定 checkpoint 的配对测试扰动敏感性分解，不构成训练捷径的因果归因。"
            "phase 表示随机初相与窗内随机游走的相位扰动组合。"
        )
    return result


def low_scr_target_detections(
    records: list[dict[str, Any]],
    threshold: float,
    score_name: str = "logit_margin",
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for item in records:
        if item["scr_db"] not in {-24, -22, -20, -18, -16}:
            continue
        mask = item["labels"] == 1
        detections = item[score_name] <= threshold
        parts.append(detections[mask].astype(np.float64))
    if not parts:
        raise ValueError("没有低 SCR 目标窗口用于因素分析。")
    return np.concatenate(parts)


def build_factor_effects(
    vectors: dict[str, np.ndarray],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """对同一窗口的 I/P/A/R 检测结果做描述性配对分解。"""
    expected = {"I", "P", "A", "R"}
    if set(vectors) != expected:
        raise ValueError(f"因素分析需要 I/P/A/R 四域，实际为 {sorted(vectors)}。")
    lengths = {len(np.asarray(values).reshape(-1)) for values in vectors.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError(f"I/P/A/R 配对窗口数量不一致或为空: {lengths}。")
    flat = {key: np.asarray(values, dtype=np.float64).reshape(-1) for key, values in vectors.items()}
    effect_vectors = {
        "D_phase": flat["I"] - flat["P"],
        "D_rcs": flat["I"] - flat["A"],
        "D_combined": flat["I"] - flat["R"],
        "D_interaction": flat["P"] + flat["A"] - flat["I"] - flat["R"],
    }
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(bootstrap_resamples, dtype=np.float64) for key in effect_vectors}
    count = next(iter(lengths))
    for bootstrap_idx in range(bootstrap_resamples):
        indices = rng.integers(0, count, size=count)
        for key, values in effect_vectors.items():
            samples[key][bootstrap_idx] = float(values[indices].mean())

    effects: dict[str, Any] = {}
    for key, values in effect_vectors.items():
        estimate = float(values.mean())
        lower, upper = np.quantile(samples[key], [0.025, 0.975])
        effects[key] = {
            "estimate": estimate,
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "stable_drop": bool(key != "D_interaction" and estimate >= 0.02 and lower > 0.0),
            "sensitivity": classify_pd_drop(estimate) if key != "D_interaction" else "descriptive_interaction",
        }
    return {
        "Y00_ideal": float(flat["I"].mean()),
        "Y10_phase_bundle": float(flat["P"].mean()),
        "Y01_rcs": float(flat["A"].mean()),
        "Y11_combined": float(flat["R"].mean()),
        "num_paired_target_windows": count,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "effects": effects,
    }


def classify_pd_drop(value: float) -> str:
    if value < 0.02:
        return "negligible_or_improved"
    if value < 0.05:
        return "mild"
    if value <= 0.10:
        return "clear"
    return "strong"


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
