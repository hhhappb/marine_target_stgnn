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
import torch
import torch.nn as nn

from paper_modules.experiments.train import git_commit, load_checkpoint
from paper_modules.datasets.ipix_window import list_split_files, load_ipix_arrays, parse_source_and_pol, seed_everything
from paper_modules.models import build_model
from utils.config import get_config_value, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe IPIX position-memory leakage with an existing checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-pfa", type=float, default=0.001)
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
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

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
