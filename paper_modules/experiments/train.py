from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "1").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from paper_modules.losses import build_loss
from paper_modules.models import build_model
from train_ipix import IpixWindowDataset, list_split_files, load_ipix_arrays, parse_source_and_pol, seed_everything
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
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_bins = correct = 0
    tp = fp = tn = fn = 0
    progress = tqdm(
        total=len(loader),
        desc=f"Epoch {epoch:03d}/{epochs}",
        dynamic_ncols=True,
        leave=True,
        mininterval=2.0,
        file=sys.stdout,
        disable=not show_progress,
    )
    try:
        for batch_idx, (real, imag, labels) in enumerate(loader, start=1):
            real = real.to(device, non_blocking=True)
            imag = imag.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            echoes = torch.complex(real, imag)

            optimizer.zero_grad(set_to_none=True)
            logits = model(echoes)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_value = float(loss.item())
            total_loss += loss_value
            pred = logits.argmax(dim=1)
            total_bins += labels.numel()
            correct += int((pred == labels).sum().item())
            tp += int(((pred == 1) & (labels == 1)).sum().item())
            fp += int(((pred == 1) & (labels == 0)).sum().item())
            tn += int(((pred == 0) & (labels == 0)).sum().item())
            fn += int(((pred == 0) & (labels == 1)).sum().item())

            metrics = _metrics(correct, total_bins, tp, fp, tn, fn)
            if show_progress:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{total_loss / batch_idx:.4f}",
                    acc=f"{metrics['accuracy']:.4f}",
                    PD=f"{metrics['pd']:.4f}",
                    PF=f"{metrics['pf']:.4f}",
                )
            elif log_interval > 0 and batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch:03d}/{epochs} batch {batch_idx}/{len(loader)} "
                    f"| loss={loss_value:.6f} | avg={total_loss / batch_idx:.6f} "
                    f"| acc={metrics['accuracy']:.4f} | PD={metrics['pd']:.4f} | PF={metrics['pf']:.4f}",
                    flush=True,
                )
    finally:
        progress.close()
    return total_loss / max(1, len(loader)), _metrics(correct, total_bins, tp, fp, tn, fn)


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


def save_checkpoint(path: Path, model: nn.Module, config: dict[str, Any], extra: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, **extra}, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train configurable paper ST-GNN modules on IPIX windows.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper_modules/configs/real_imag_sfe_replacement_original_sfe.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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

    seed = int(get_config_value(config, "train.seed"))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(get_config_value(config, "paths.data_dir"))
    save_dir = Path(get_config_value(config, "paths.save_dir"))
    pols = get_config_value(config, "ipix.polarizations")
    train_files = list_split_files(data_dir, "train", pols)
    test_files = list_split_files(data_dir, "test", pols)
    if not train_files:
        raise SystemExit(f"No train files found under {data_dir}")
    if not test_files:
        raise SystemExit(f"No test files found under {data_dir}")

    model = build_model(config).to(device)
    print("=" * 72, flush=True)
    print("Paper module ST-GNN training/evaluation", flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Device: {device} | Data dir: {data_dir}", flush=True)
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)} | Pols: {pols}", flush=True)
    print("=" * 72, flush=True)

    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--eval-only requires --checkpoint")
        load_checkpoint(args.checkpoint, model, device)
    else:
        train_dataset = IpixWindowDataset(train_files, max_windows=args.max_train_windows, seed=seed)
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
        criterion = build_loss(config, train_dataset.class_weights().to(device))
        optimizer = optim.AdamW(model.parameters(), lr=float(get_config_value(config, "train.learning_rate")), weight_decay=1e-4)
        epochs = int(get_config_value(config, "train.epochs"))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        best_loss = float("inf")
        best_path = save_dir / "best_model.pth"
        t0 = time.time()
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
            )
            scheduler.step()
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | loss={loss:.6f} "
                f"| acc={metrics['accuracy']:.4f} | PD={metrics['pd']:.4f} | PF={metrics['pf']:.4f} "
                f"| {time.time() - t0:.1f}s",
                flush=True,
            )
            if loss < best_loss:
                best_loss = loss
                save_checkpoint(best_path, model, config, {"best_loss": best_loss, "epoch": epoch + 1})
                print(f"  saved best checkpoint: {best_path}", flush=True)
        save_checkpoint(save_dir / "final_model.pth", model, config, {"best_loss": best_loss, "epoch": epochs})
        load_checkpoint(best_path, model, device)

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
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nEvaluation summary", flush=True)
    for pfa, payload in results["pfa"].items():
        print(f"Pfa={pfa} | threshold={payload['threshold']:.8f} | PD={payload['PD']:.4f} | PF={payload['PF']:.6f}", flush=True)
    print(f"Saved results: {results_path}", flush=True)


def _metrics(correct: int, total_bins: int, tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    return {
        "accuracy": correct / total_bins if total_bins else 0.0,
        "pd": tp / (tp + fn) if (tp + fn) else 0.0,
        "pf": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def _pd_pf(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    return {"PD": tp / (tp + fn) if (tp + fn) else 0.0, "PF": fp / (fp + tn) if (fp + tn) else 0.0}


if __name__ == "__main__":
    main()
