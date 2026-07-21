from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np

if not os.environ.get("OMP_NUM_THREADS", "1").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from paper_modules.datasets import build_dataset, list_split_files, load_ipix_arrays, parse_source_and_pol, seed_everything
from paper_modules.datasets.scr_npz import ScrNpzDataset, list_test_scr_files
from paper_modules.losses import build_loss
from paper_modules.models import build_model
from utils.config import get_config_value, load_config


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    show_progress: bool,
    log_interval: int,
    grad_clip: float | None,
    gradient_accumulation_steps: int = 1,
    collect_temporal_diagnostics: bool = False,
) -> tuple[float, dict[str, float]]:
    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps 必须为正整数，实际为 {gradient_accumulation_steps}。"
        )
    model.train()
    total_loss = 0.0
    tp = fp = tn = fn = 0
    diagnostics: list[dict[str, float]] = []
    gradient_norms: list[float] = []
    progress = tqdm(
        total=len(loader),
        desc=f"Epoch {epoch:03d}/{epochs}",
        dynamic_ncols=True,
        leave=True,
        mininterval=2.0,
        file=sys.stdout,
        disable=not show_progress,
    )
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    try:
        for batch_idx, (real, imag, labels) in enumerate(loader, start=1):
            real = real.to(device, non_blocking=True)
            imag = imag.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            echoes = torch.complex(real, imag)

            logits = model(echoes)
            loss = criterion(logits, labels)
            if collect_temporal_diagnostics and hasattr(model, "get_temporal_diagnostics"):
                diagnostics.append(model.get_temporal_diagnostics())
            window_start = ((batch_idx - 1) // gradient_accumulation_steps) * gradient_accumulation_steps + 1
            window_size = min(gradient_accumulation_steps, len(loader) - window_start + 1)
            (loss / window_size).backward()
            should_step = batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(loader)
            if should_step:
                gradient_norms.append(_gradient_norm(model))
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            loss_value = float(loss.item())
            total_loss += loss_value
            pred = logits.argmax(dim=1)
            tp += int(((pred == 1) & (labels == 1)).sum().item())
            fp += int(((pred == 1) & (labels == 0)).sum().item())
            tn += int(((pred == 0) & (labels == 0)).sum().item())
            fn += int(((pred == 0) & (labels == 1)).sum().item())

            metrics = _metrics(tp, fp, tn, fn)
            if show_progress:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{total_loss / batch_idx:.4f}",
                    PD=f"{metrics['pd']:.4f}",
                    PF=f"{metrics['pf']:.4f}",
                )
            elif log_interval > 0 and batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch:03d}/{epochs} batch {batch_idx}/{len(loader)} "
                    f"| loss={loss_value:.6f} | avg={total_loss / batch_idx:.6f} "
                    f"| PD={metrics['pd']:.4f} | PF={metrics['pf']:.4f}",
                    flush=True,
                )
    finally:
        progress.close()
    metrics = _metrics(tp, fp, tn, fn)
    metrics["optimizer_steps"] = float(optimizer_steps)
    if gradient_norms:
        metrics["gradient_norm"] = float(sum(gradient_norms) / len(gradient_norms))
    if diagnostics:
        names = sorted({name for record in diagnostics for name in record})
        for name in names:
            values = [record[name] for record in diagnostics if name in record]
            if values:
                metrics[f"diagnostic_{name}"] = float(sum(values) / len(values))
    return total_loss / max(1, len(loader)), metrics


def probe_batch_sizes(
    model: nn.Module,
    dataset: Any,
    criterion: nn.Module,
    device: torch.device,
    candidates: list[int],
    memory_fraction: float,
) -> dict[str, Any]:
    """用真实单批前向和反向探测训练显存，不执行参数更新。"""
    if device.type != "cuda":
        raise RuntimeError("batch 自动探测需要 CUDA 设备。")
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError(f"batch candidates 必须为正整数，实际为 {candidates}。")
    if candidates != sorted(set(candidates)):
        raise ValueError("batch candidates 必须严格递增且不重复。")
    if not 0.0 < memory_fraction < 1.0:
        raise ValueError(f"memory_fraction 必须位于 (0, 1)，实际为 {memory_fraction}。")

    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    records: list[dict[str, Any]] = []
    model.train()
    for batch_size in candidates:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        real = imag = labels = echoes = logits = loss = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            real, imag, labels = next(iter(loader))
            real = real.to(device)
            imag = imag.to(device)
            labels = labels.to(device)
            echoes = torch.complex(real, imag)
            # 首次执行只用于 CUDA kernel/allocator 预热，不计入吞吐。
            model.zero_grad(set_to_none=True)
            logits = model(echoes)
            loss = criterion(logits, labels)
            loss.backward()
            torch.cuda.synchronize(device)
            model.zero_grad(set_to_none=True)
            del logits, loss
            logits = loss = None
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            logits = model(echoes)
            loss = criterion(logits, labels)
            loss.backward()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            within_limit = peak_reserved <= int(total_bytes * memory_fraction)
            records.append(
                {
                    "batch_size": batch_size,
                    "status": "pass" if within_limit else "over_memory_limit",
                    "elapsed_seconds": elapsed,
                    "windows_per_second": batch_size / elapsed,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_reserved_gib": peak_reserved / (1024**3),
                    "memory_fraction": peak_reserved / total_bytes,
                    "loss": float(loss.item()),
                }
            )
        except torch.OutOfMemoryError:
            records.append({"batch_size": batch_size, "status": "oom"})
        finally:
            model.zero_grad(set_to_none=True)
            del real, imag, labels, echoes, logits, loss, loader
            gc.collect()
            torch.cuda.empty_cache()

    passing = [item for item in records if item["status"] == "pass"]
    if not passing:
        raise RuntimeError(f"候选 batch 均未通过显存限制: {records}。")
    largest_safe = max(int(item["batch_size"]) for item in passing)
    recommended = max(passing, key=lambda item: float(item["windows_per_second"]))
    return {
        "device": torch.cuda.get_device_name(device),
        "total_memory_bytes": total_bytes,
        "memory_limit_fraction": memory_fraction,
        "selected_batch_size": int(recommended["batch_size"]),
        "selection_rule": "max_windows_per_second_within_memory_limit_after_one_warmup",
        "largest_safe_batch_size": largest_safe,
        "records": records,
    }


def evaluate_files(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    pfa_values: list[float],
    max_windows_per_file: int | None = None,
    threshold_files: list[Path] | None = None,
    threshold_source: str = "test_diagnostic_current_eval",
    max_threshold_windows: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    model.eval()
    import numpy as np

    eval_rng = np.random.default_rng(seed)
    threshold_rng = np.random.default_rng(seed)
    records, eval_clutter = collect_file_scores(
        model,
        files,
        batch_size,
        device,
        eval_rng,
        max_windows_per_file=max_windows_per_file,
    )

    if threshold_source == "test_diagnostic_current_eval":
        threshold_clutter = eval_clutter
        num_threshold_files = len(records)
    elif threshold_source == "train_clutter":
        if threshold_files is None:
            raise ValueError("threshold_source=train_clutter 需要提供 threshold_files。")
        threshold_records, threshold_clutter = collect_file_scores(
            model,
            threshold_files,
            batch_size,
            device,
            threshold_rng,
            max_total_windows=max_threshold_windows,
        )
        num_threshold_files = len(threshold_records)
    else:
        raise ValueError(f"未知 threshold_source: {threshold_source}")

    results: dict[str, object] = {
        "num_files": len(records),
        "num_clutter_bins": int(eval_clutter.shape[0]),
        "threshold_source": threshold_source,
        "num_threshold_files": num_threshold_files,
        "num_clutter_bins_for_threshold": int(threshold_clutter.shape[0]),
        "pfa": {},
    }
    for pfa in pfa_values:
        ordered = np.sort(threshold_clutter)
        idx = int(np.ceil(pfa * len(ordered))) - 1
        threshold = float(ordered[max(0, min(idx, len(ordered) - 1))])
        total = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        per_file = []
        for item in records:
            labels = item["labels"]
            det = (item["o0"] <= threshold).astype(np.uint8)
            counts = {
                "TP": int(((det == 1) & (labels == 1)).sum()),
                "FN": int(((det == 0) & (labels == 1)).sum()),
                "FP": int(((det == 1) & (labels == 0)).sum()),
                "TN": int(((det == 0) & (labels == 0)).sum()),
            }
            for key in total:
                total[key] += counts[key]
            per_file.append({"source": item["source"], "polarization": item["polarization"], **_pd_pf(counts), **counts})
        results["pfa"][str(pfa)] = {"threshold": threshold, **_pd_pf(total), **total, "per_file": per_file}
    return results


def collect_file_scores(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    rng: Any,
    max_windows_per_file: int | None = None,
    max_total_windows: int | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np

    records: list[dict[str, Any]] = []
    clutter_scores: list[np.ndarray] = []
    remaining = max_total_windows

    with torch.no_grad():
        for path in files:
            if remaining is not None and remaining <= 0:
                break
            limit = max_windows_per_file
            if remaining is not None:
                limit = remaining if limit is None else min(limit, remaining)
            x, y = load_ipix_arrays(path, max_windows=limit, rng=rng)
            if remaining is not None:
                remaining -= len(x)
            o0_parts: list[np.ndarray] = []
            for start in range(0, len(x), batch_size):
                batch = x[start : start + batch_size]
                real = torch.from_numpy(batch.real.astype(np.float32, copy=False)).to(device)
                imag = torch.from_numpy(batch.imag.astype(np.float32, copy=False)).to(device)
                logits = model(torch.complex(real, imag))
                probs = torch.softmax(logits, dim=1)[:, 0, :].cpu().numpy()
                o0_parts.append(probs)
            o0 = np.concatenate(o0_parts, axis=0)
            clutter_scores.append(o0[y == 0])
            source, pol = parse_source_and_pol(path)
            records.append({"source": source, "polarization": pol, "o0": o0, "labels": y})

    if not clutter_scores:
        raise ValueError("没有可用于阈值或评估的 clutter scores。")
    return records, np.concatenate(clutter_scores)


def evaluate_scr_curve(
    model: nn.Module,
    config: dict[str, Any],
    data_dir: Path,
    batch_size: int,
    device: torch.device,
    pfa_values: list[float],
    max_windows_per_scr: int | None = None,
    max_threshold_windows: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    model.eval()
    threshold_source = str(config.get("eval", {}).get("threshold_source", ""))
    score_space = str(config.get("eval", {}).get("score_space", "o0"))
    if score_space not in {"o0", "logit_margin"}:
        raise ValueError(f"pd_scr_curve 不支持 score_space={score_space!r}。")
    if threshold_source != "train":
        raise ValueError(f"pd_scr_curve 必须使用 threshold_source=train，实际为 {threshold_source!r}。")
    train_dataset = build_dataset(config, "train", max_windows=max_threshold_windows, seed=seed)
    if not isinstance(train_dataset, ScrNpzDataset):
        raise TypeError(f"pd_scr_curve 需要 ScrNpzDataset，实际为 {type(train_dataset).__name__}。")
    train_scores, train_labels = collect_dataset_scores(model, train_dataset, batch_size, device, score_space)
    threshold_clutter = train_scores[train_labels == 0]
    if threshold_clutter.size == 0:
        raise ValueError("SCR train.npz 中没有杂波样本，无法按训练集杂波计算阈值。")

    scr_files = list_test_scr_files(data_dir)
    if not scr_files:
        raise ValueError(f"没有找到 SCR 测试文件: {data_dir / 'test_scr_*.npz'}")

    scr_records: list[dict[str, object]] = []
    for path in scr_files:
        scr = int(path.stem.replace("test_scr_", ""))
        dataset = build_dataset(
            config,
            "test",
            scr=scr,
            max_windows=max_windows_per_scr,
            seed=seed,
            norm=train_dataset.norm,
        )
        if not isinstance(dataset, ScrNpzDataset):
            raise TypeError(f"pd_scr_curve 需要 ScrNpzDataset，实际为 {type(dataset).__name__}。")
        scores, labels = collect_dataset_scores(model, dataset, batch_size, device, score_space)
        scr_records.append({"scr_db": scr, "scores": scores, "labels": labels})

    results: dict[str, object] = {
        "protocol": "pd_scr_curve",
        "threshold_source": "train",
        "score_space": score_space,
        "num_test_scr_files": len(scr_records),
        "num_clutter_bins_for_threshold": int(threshold_clutter.shape[0]),
        "pfa": {},
    }
    ordered = np.sort(threshold_clutter)
    for pfa in pfa_values:
        if score_space == "o0":
            idx = int(np.ceil(pfa * len(ordered))) - 1
            threshold = float(ordered[max(0, min(idx, len(ordered) - 1))])
        else:
            idx = int(np.ceil((1.0 - pfa) * len(ordered))) - 1
            threshold = float(ordered[max(0, min(idx, len(ordered) - 1))])
        total = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        per_scr = []
        for item in scr_records:
            labels = item["labels"]
            if score_space == "o0":
                det = (item["scores"] <= threshold).astype(np.uint8)
            else:
                det = (item["scores"] >= threshold).astype(np.uint8)
            counts = {
                "TP": int(((det == 1) & (labels == 1)).sum()),
                "FN": int(((det == 0) & (labels == 1)).sum()),
                "FP": int(((det == 1) & (labels == 0)).sum()),
                "TN": int(((det == 0) & (labels == 0)).sum()),
            }
            for key in total:
                total[key] += counts[key]
            per_scr.append({"scr_db": item["scr_db"], **_pd_pf(counts), **counts})
        ordered_scr = sorted(per_scr, key=lambda item: int(item["scr_db"]))
        scr_axis = np.asarray([int(item["scr_db"]) for item in ordered_scr], dtype=np.float64)
        pd_axis = np.asarray([float(item["PD"]) for item in ordered_scr], dtype=np.float64)
        scr_span = float(scr_axis[-1] - scr_axis[0]) if scr_axis.size > 1 else 0.0
        pd_scr_auc = float(np.trapezoid(pd_axis, scr_axis) / scr_span) if scr_span > 0 else 0.0
        results["pfa"][str(pfa)] = {
            "threshold": threshold,
            **_pd_pf(total),
            **total,
            "PD_SCR_AUC": pd_scr_auc,
            "per_scr": per_scr,
        }
    return results


def collect_dataset_scores(
    model: nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    score_space: str = "o0",
) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    o0_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.no_grad():
        for real, imag, labels in loader:
            real = real.to(device, non_blocking=True)
            imag = imag.to(device, non_blocking=True)
            logits = model(torch.complex(real, imag))
            if score_space == "o0":
                scores = torch.softmax(logits, dim=1)[:, 0, :]
            elif score_space == "logit_margin":
                scores = logits[:, 1, :] - logits[:, 0, :]
            else:
                raise ValueError(f"未知 score_space: {score_space}")
            o0_parts.append(scores.cpu().numpy())
            label_parts.append(labels.numpy())
    if not o0_parts:
        raise ValueError("数据集为空，无法评估。")
    return np.concatenate(o0_parts, axis=0), np.concatenate(label_parts, axis=0)


def save_checkpoint(path: Path, model: nn.Module, config: dict[str, Any], extra: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, **extra}, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)


def build_optimizer(config: dict[str, Any], model: nn.Module) -> optim.Optimizer:
    train_cfg = config.get("train", {})
    optimizer_type = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(get_config_value(config, "train.learning_rate"))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4 if optimizer_type == "adamw" else 0.0))
    if optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知 optimizer: {optimizer_type}")


def build_scheduler(config: dict[str, Any], optimizer: optim.Optimizer, epochs: int):
    train_cfg = config.get("train", {})
    scheduler_type = str(train_cfg.get("scheduler", "cosine")).lower()
    if scheduler_type in {"none", "null", "off"}:
        return None
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    raise ValueError(f"未知 scheduler: {scheduler_type}")


def write_config_snapshot(path: Path, config: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: PyYAML. Install with `python -m pip install PyYAML`.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train configurable paper ST-GNN modules on IPIX windows.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper_modules/configs/real_imag_sfe_replacement_original_sfe.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--auto-batch-probe", action="store_true")
    parser.add_argument(
        "--batch-candidates",
        type=int,
        nargs="+",
        default=[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
    )
    parser.add_argument("--batch-memory-fraction", type=float, default=0.90)
    parser.add_argument("--batch-probe-output", type=Path, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-interval", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config["train"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    accumulation_steps = int(config.get("train", {}).get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError(f"train.gradient_accumulation_steps 必须为正整数，实际为 {accumulation_steps}。")
    if args.auto_batch_probe and args.eval_only:
        raise ValueError("--auto-batch-probe 与 --eval-only 不能同时使用。")

    seed = int(get_config_value(config, "train.seed"))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(get_config_value(config, "paths.data_dir"))
    save_dir = Path(get_config_value(config, "paths.save_dir"))
    eval_protocol = str(config.get("eval", {}).get("protocol", "per_file_pol"))
    dataset_type = str(config.get("dataset", {}).get("type", "ipix_window"))
    dataset_cfg = config.get("dataset", {})
    diagnostics_cfg = config.get("diagnostics", {})
    if not isinstance(diagnostics_cfg, dict):
        raise ValueError("diagnostics 配置必须是 mapping。")
    temporal_diagnostics_enabled = bool(diagnostics_cfg.get("temporal", False))
    pols = dataset_cfg.get("polarizations", get_config_value(config, "ipix.polarizations"))
    sources = _as_list(dataset_cfg.get("sources", dataset_cfg.get("source")))

    train_files: list[Path] = []
    test_files: list[Path] = []
    if dataset_type == "ipix_window":
        train_files = list_split_files(data_dir, "train", list(pols), sources=sources)
        test_files = list_split_files(data_dir, "test", list(pols), sources=sources)
        if not train_files:
            raise SystemExit(f"No train files found under {data_dir}")
        if not test_files:
            raise SystemExit(f"No test files found under {data_dir}")
    elif dataset_type == "scr_npz":
        if not (data_dir / "train.npz").exists():
            raise SystemExit(f"No SCR train file found under {data_dir}")
        train_files = [data_dir / "train.npz"]
        test_files = list_test_scr_files(data_dir)
        if not test_files:
            raise SystemExit(f"No SCR test files found under {data_dir}")
    else:
        raise SystemExit(f"Unknown dataset type: {dataset_type}")

    model = build_model(config).to(device)
    if args.auto_batch_probe:
        train_dataset = build_dataset(config, "train", max_windows=args.max_train_windows, seed=seed)
        use_class_weights = bool(config.get("loss", {}).get("use_class_weights", True))
        class_weights = train_dataset.class_weights().to(device) if use_class_weights else None
        criterion = build_loss(config, class_weights)
        probe = probe_batch_sizes(
            model,
            train_dataset,
            criterion,
            device,
            args.batch_candidates,
            args.batch_memory_fraction,
        )
        probe.update(
            {
                "config": str(args.config),
                "data_dir": str(data_dir),
                "dataset_protocol": dataset_cfg.get("protocol"),
                "model": config.get("model", {}),
            }
        )
        output = args.batch_probe_output
        if output is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = Path("logs") / "training" / f"{stamp}_batch_probe" / "batch_probe.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(probe, indent=2), encoding="utf-8")
        print(json.dumps(probe, indent=2), flush=True)
        print(f"Saved batch probe: {output}", flush=True)
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(save_dir / "config.yaml", config)
    print("=" * 72, flush=True)
    print("Paper module ST-GNN training/evaluation", flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Device: {device} | Data dir: {data_dir}", flush=True)
    print(f"Dataset: {dataset_type} | Eval protocol: {eval_protocol}", flush=True)
    if dataset_type == "ipix_window":
        print(f"Train files: {len(train_files)} | Test files: {len(test_files)} | Sources: {sources or 'all'} | Pols: {pols}", flush=True)
    else:
        print(f"Train file: {data_dir / 'train.npz'} | Test SCR files: {len(test_files)}", flush=True)
    print("=" * 72, flush=True)

    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--eval-only requires --checkpoint")
        load_checkpoint(args.checkpoint, model, device)
    else:
        train_dataset = build_dataset(config, "train", max_windows=args.max_train_windows, seed=seed)
        print(f"Loaded train windows: {len(train_dataset):,}", flush=True)
        print(f"Class weights: {train_dataset.class_weights().tolist()}", flush=True)
        loader = DataLoader(
            train_dataset,
            batch_size=int(get_config_value(config, "train.batch_size")),
            shuffle=True,
            num_workers=int(get_config_value(config, "train.num_workers")),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(get_config_value(config, "train.num_workers")) > 0,
        )
        print(
            f"Micro batch: {loader.batch_size} | gradient accumulation: {accumulation_steps} "
            f"| nominal effective batch: {loader.batch_size * accumulation_steps}",
            flush=True,
        )
        use_class_weights = bool(config.get("loss", {}).get("use_class_weights", True))
        class_weights = train_dataset.class_weights().to(device) if use_class_weights else None
        criterion = build_loss(config, class_weights)
        if use_class_weights:
            print(f"Loss class weights: {train_dataset.class_weights().tolist()}", flush=True)
        else:
            print("Loss class weights: disabled", flush=True)
        optimizer = build_optimizer(config, model)
        epochs = int(get_config_value(config, "train.epochs"))
        scheduler = build_scheduler(config, optimizer, epochs)
        grad_clip_raw = config.get("train", {}).get("grad_clip", 1.0)
        grad_clip = None if grad_clip_raw is None else float(grad_clip_raw)
        best_loss = float("inf")
        best_path = save_dir / "best_model.pth"
        t0 = time.time()
        training_history: list[dict[str, float]] = []
        for epoch in range(epochs):
            loss, metrics = train_one_epoch(
                model,
                loader,
                criterion,
                optimizer,
                device,
                epoch + 1,
                epochs,
                not args.no_progress,
                args.log_interval,
                grad_clip,
                accumulation_steps,
                temporal_diagnostics_enabled,
            )
            if scheduler is not None:
                scheduler.step()
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | loss={loss:.6f} "
                f"| PD={metrics['pd']:.4f} | PF={metrics['pf']:.4f} "
                f"| optimizer_steps={int(metrics['optimizer_steps'])} "
                f"| {time.time() - t0:.1f}s",
                flush=True,
            )
            training_history.append({"epoch": float(epoch + 1), "loss": float(loss), **metrics})
            if loss < best_loss:
                best_loss = loss
                save_checkpoint(best_path, model, config, {"best_loss": best_loss, "epoch": epoch + 1})
                print(f"  saved best checkpoint: {best_path}", flush=True)
        save_checkpoint(save_dir / "final_model.pth", model, config, {"best_loss": best_loss, "epoch": epochs})
        load_checkpoint(best_path, model, device)
    if args.eval_only:
        training_history = []
        previous_metrics = save_dir / "metrics.json"
        if previous_metrics.exists():
            previous_results = json.loads(previous_metrics.read_text(encoding="utf-8"))
            training_history = list(previous_results.get("training_history", []))

    if eval_protocol == "per_file_pol":
        if dataset_type != "ipix_window":
            raise ValueError("eval.protocol=per_file_pol 只支持 dataset.type=ipix_window。")
        results = evaluate_files(
            model,
            test_files,
            int(get_config_value(config, "train.batch_size")),
            device,
            get_config_value(config, "eval.pfa_values"),
            max_windows_per_file=args.max_test_windows_per_file,
            threshold_files=train_files,
            threshold_source=str(config.get("eval", {}).get("threshold_source", "test_diagnostic_current_eval")),
            max_threshold_windows=args.max_train_windows,
            seed=seed,
        )
    elif eval_protocol == "pd_scr_curve":
        if dataset_type != "scr_npz":
            raise ValueError("eval.protocol=pd_scr_curve 只支持 dataset.type=scr_npz。")
        results = evaluate_scr_curve(
            model,
            config,
            data_dir,
            int(get_config_value(config, "train.batch_size")),
            device,
            get_config_value(config, "eval.pfa_values"),
            max_windows_per_scr=args.max_test_windows_per_file,
            max_threshold_windows=args.max_train_windows,
            seed=seed,
        )
    else:
        raise ValueError(f"未知 eval.protocol: {eval_protocol}")

    checkpoint_path = args.checkpoint if args.eval_only else save_dir / "best_model.pth"
    results.update(
        run_metadata(
            args=args,
            config=config,
            data_dir=data_dir,
            dataset_type=dataset_type,
            eval_protocol=eval_protocol,
            sources=sources,
            pols=pols,
            train_files=train_files,
            test_files=test_files,
            checkpoint_path=checkpoint_path,
        )
    )
    results["training_history"] = training_history
    if temporal_diagnostics_enabled and hasattr(model, "get_temporal_diagnostics"):
        results["final_temporal_diagnostics"] = model.get_temporal_diagnostics()
        results["final_temporal_diagnostics_scope"] = "last_evaluation_batch"
    results_path = save_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nEvaluation summary", flush=True)
    for pfa, payload in results["pfa"].items():
        print(f"Pfa={pfa} | threshold={payload['threshold']:.8f} | PD={payload['PD']:.4f} | PF={payload['PF']:.6f}", flush=True)
        if eval_protocol == "pd_scr_curve":
            for item in payload["per_scr"]:
                print(f"  SCR={item['scr_db']:>3} dB | PD={item['PD']:.4f} | PF={item['PF']:.6f}", flush=True)
    print(f"Saved results: {results_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    return {
        "pd": tp / (tp + fn) if (tp + fn) else 0.0,
        "pf": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def _gradient_norm(model: nn.Module) -> float:
    """计算当前 optimizer step 前的全模型梯度二范数。"""
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return squared**0.5


def _pd_pf(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    return {"PD": tp / (tp + fn) if (tp + fn) else 0.0, "PF": fp / (fp + tn) if (fp + tn) else 0.0}


def run_metadata(
    args: argparse.Namespace,
    config: dict[str, Any],
    data_dir: Path,
    dataset_type: str,
    eval_protocol: str,
    sources: list[str] | None,
    pols: Any,
    train_files: list[Path],
    test_files: list[Path],
    checkpoint_path: Path,
) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    return {
        "protocol": eval_protocol,
        "dataset_type": dataset_type,
        "data_dir": str(data_dir),
        "dataset_data_dir": str(dataset_cfg.get("data_dir", data_dir)),
        "sources": sources or [],
        "polarizations": _as_list(pols) or [],
        "train_augmentation": dataset_cfg.get("augment", {}),
        "dataset_protocol": dataset_cfg.get("protocol") if dataset_type == "scr_npz" else None,
        "dataset_normalization": (
            dataset_cfg.get("normalization", "train_standardize_clip") if dataset_type == "scr_npz" else None
        ),
        "model_config": config.get("model", {}),
        "optimizer": config.get("train", {}).get("optimizer", "adamw"),
        "scheduler": config.get("train", {}).get("scheduler", "cosine"),
        "grad_clip": config.get("train", {}).get("grad_clip", 1.0),
        "gradient_accumulation_steps": int(config.get("train", {}).get("gradient_accumulation_steps", 1)),
        "nominal_effective_batch_size": int(config.get("train", {}).get("batch_size", 0))
        * int(config.get("train", {}).get("gradient_accumulation_steps", 1)),
        "loss_use_class_weights": config.get("loss", {}).get("use_class_weights", True),
        "num_train_files": len(train_files),
        "num_test_files": len(test_files),
        "train_files": [str(path) for path in train_files],
        "test_files": [str(path) for path in test_files],
        "config_path": str(args.config),
        "checkpoint": str(checkpoint_path),
        "git_commit": git_commit(),
    }


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


if __name__ == "__main__":
    main()
