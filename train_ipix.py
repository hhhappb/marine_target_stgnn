from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models.st_gnn import STGNNDetector
from utils.config import get_config_value, load_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_split_files(data_dir: Path, split: str, pols: list[str]) -> list[Path]:
    files: list[Path] = []
    for pol in pols:
        files.extend(data_dir.glob(f"*__{pol}__{split}.npz"))
    return sorted(files)


def parse_source_and_pol(path: Path) -> tuple[str, str]:
    parts = path.stem.split("__")
    if len(parts) < 3:
        return path.stem, "unknown"
    return parts[0], parts[1]


def load_ipix_arrays(path: Path, max_windows: int | None = None, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        x = data["E"]
        y = data["y_range"]
        if max_windows is not None and len(x) > max_windows:
            if rng is None:
                rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(len(x), size=max_windows, replace=False))
            x = x[idx]
            y = y[idx]
        return x.astype(np.complex64, copy=False), y.astype(np.int64, copy=False)


class IpixWindowDataset(Dataset):
    def __init__(self, files: list[Path], max_windows: int | None = None, seed: int = 42):
        self.files = files
        self.x_parts: list[np.ndarray] = []
        self.y_parts: list[np.ndarray] = []
        self.rng = np.random.default_rng(seed)
        remaining = max_windows

        for path in files:
            if remaining is not None and remaining <= 0:
                break
            limit = remaining
            x, y = load_ipix_arrays(path, max_windows=limit, rng=self.rng)
            self.x_parts.append(x)
            self.y_parts.append(y)
            if remaining is not None:
                remaining -= len(x)

        if not self.x_parts:
            raise ValueError("No IPIX windows were loaded. Check data_dir, split, polarizations, and max_windows.")

        self.x = np.concatenate(self.x_parts, axis=0)
        self.y = np.concatenate(self.y_parts, axis=0)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.x[idx]
        labels = self.y[idx]
        return (
            torch.from_numpy(sample.real.astype(np.float32, copy=False)),
            torch.from_numpy(sample.imag.astype(np.float32, copy=False)),
            torch.from_numpy(labels.astype(np.int64, copy=False)),
        )

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.y.reshape(-1), minlength=2).astype(np.float64)
        if np.any(counts == 0):
            return torch.ones(2, dtype=torch.float32)
        weights = counts.sum() / (2.0 * counts)
        return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: STGNNDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_bins = 0
    correct = 0
    tp = fp = tn = fn = 0

    for real, imag, labels in loader:
        real = real.to(device)
        imag = imag.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, logits, _, _ = model(torch.complex(real, imag), return_features=True)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        total_bins += labels.numel()
        correct += int((pred == labels).sum().item())
        tp += int(((pred == 1) & (labels == 1)).sum().item())
        fp += int(((pred == 1) & (labels == 0)).sum().item())
        tn += int(((pred == 0) & (labels == 0)).sum().item())
        fn += int(((pred == 0) & (labels == 1)).sum().item())

    metrics = {
        "accuracy": correct / total_bins if total_bins else 0.0,
        "pd": tp / (tp + fn) if (tp + fn) else 0.0,
        "pf": fp / (fp + tn) if (fp + tn) else 0.0,
    }
    return total_loss / max(1, len(loader)), metrics


def evaluate_files(
    model: STGNNDetector,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    pfa_values: list[float],
    max_windows_per_file: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    model.eval()
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    clutter_scores: list[np.ndarray] = []

    with torch.no_grad():
        for path in files:
            x, y = load_ipix_arrays(path, max_windows=max_windows_per_file, rng=rng)
            o0_parts: list[np.ndarray] = []
            for start in range(0, len(x), batch_size):
                batch = x[start : start + batch_size]
                real = torch.from_numpy(batch.real.astype(np.float32, copy=False)).to(device)
                imag = torch.from_numpy(batch.imag.astype(np.float32, copy=False)).to(device)
                _, logits, _, _ = model(torch.complex(real, imag), return_features=True)
                probs = torch.softmax(logits, dim=1)[:, 0, :].cpu().numpy()
                o0_parts.append(probs)
            o0 = np.concatenate(o0_parts, axis=0)
            clutter_scores.append(o0[y == 0])
            source, pol = parse_source_and_pol(path)
            records.append({"source": source, "polarization": pol, "o0": o0, "labels": y})

    all_clutter = np.concatenate(clutter_scores)
    results: dict[str, object] = {
        "num_files": len(files),
        "num_clutter_bins": int(all_clutter.shape[0]),
        "pfa": {},
    }

    for pfa in pfa_values:
        ordered = np.sort(all_clutter)
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
            tp_i, fn_i, fp_i, tn_i = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
            per_file.append(
                {
                    "source": item["source"],
                    "polarization": item["polarization"],
                    "PD": tp_i / (tp_i + fn_i) if (tp_i + fn_i) else 0.0,
                    "PF": fp_i / (fp_i + tn_i) if (fp_i + tn_i) else 0.0,
                    **counts,
                }
            )

        tp, fn, fp, tn = total["TP"], total["FN"], total["FP"], total["TN"]
        results["pfa"][str(pfa)] = {
            "threshold": threshold,
            "PD": tp / (tp + fn) if (tp + fn) else 0.0,
            "PF": fp / (fp + tn) if (fp + tn) else 0.0,
            **total,
            "per_file": per_file,
        }

    return results


def save_checkpoint(path: Path, model: STGNNDetector, args: argparse.Namespace, extra: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            **extra,
        },
        path,
    )


def load_checkpoint(path: Path, model: STGNNDetector, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)


def parse_args() -> argparse.Namespace:
    default_config = Path("configs/ipix_stgnn.toml")
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=default_config if default_config.exists() else None)
    config_args, remaining = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="Train and evaluate ST-GNN on preprocessed IPIX Dartmouth windows.")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--data-dir", type=Path, default=Path(get_config_value(config, "paths.data_dir")))
    parser.add_argument("--save-dir", type=Path, default=Path(get_config_value(config, "paths.save_dir")))
    parser.add_argument("--pols", nargs="+", default=get_config_value(config, "ipix.polarizations"))
    parser.add_argument("--P", type=int, default=get_config_value(config, "model.pulses"))
    parser.add_argument("--N", type=int, default=get_config_value(config, "model.range_cells"))
    parser.add_argument("--epochs", type=int, default=get_config_value(config, "train.epochs"))
    parser.add_argument("--batch-size", type=int, default=get_config_value(config, "train.batch_size"))
    parser.add_argument("--lr", type=float, default=get_config_value(config, "train.learning_rate"))
    parser.add_argument("--num-workers", type=int, default=get_config_value(config, "train.num_workers"))
    parser.add_argument("--seed", type=int, default=get_config_value(config, "train.seed"))
    parser.add_argument("--pfa-values", nargs="+", type=float, default=get_config_value(config, "eval.pfa_values"))
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files = list_split_files(args.data_dir, "train", args.pols)
    test_files = list_split_files(args.data_dir, "test", args.pols)
    if not train_files:
        raise SystemExit(f"No train files found under {args.data_dir}")
    if not test_files:
        raise SystemExit(f"No test files found under {args.data_dir}")

    print("=" * 72, flush=True)
    print("ST-GNN IPIX training/evaluation", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Data dir: {args.data_dir}", flush=True)
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)} | Pols: {args.pols}", flush=True)
    print(f"Model: P={args.P}, N={args.N}", flush=True)
    print("=" * 72, flush=True)

    model = STGNNDetector(P=args.P, N=args.N).to(device)

    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--eval-only requires --checkpoint")
        load_checkpoint(args.checkpoint, model, device)
    else:
        train_dataset = IpixWindowDataset(train_files, max_windows=args.max_train_windows, seed=args.seed)
        print(f"Loaded train windows: {len(train_dataset):,}", flush=True)
        print(f"Class weights: {train_dataset.class_weights().tolist()}", flush=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights().to(device))
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        best_loss = float("inf")
        best_path = args.save_dir / "ipix_best_model.pth"
        t0 = time.time()

        for epoch in range(args.epochs):
            loss, metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | loss={loss:.6f} "
                f"| acc={metrics['accuracy']:.4f} | PD={metrics['pd']:.4f} | PF={metrics['pf']:.4f} "
                f"| {elapsed:.1f}s",
                flush=True,
            )
            if loss < best_loss:
                best_loss = loss
                save_checkpoint(best_path, model, args, {"best_loss": best_loss, "epoch": epoch + 1})
                print(f"  saved best checkpoint: {best_path}", flush=True)

        final_path = args.save_dir / "ipix_final_model.pth"
        save_checkpoint(final_path, model, args, {"best_loss": best_loss, "epoch": args.epochs})
        load_checkpoint(best_path, model, device)

    results = evaluate_files(
        model,
        test_files,
        args.batch_size,
        device,
        args.pfa_values,
        max_windows_per_file=args.max_test_windows_per_file,
        seed=args.seed,
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.save_dir / "ipix_eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nEvaluation summary", flush=True)
    for pfa, payload in results["pfa"].items():
        print(
            f"Pfa={pfa} | threshold={payload['threshold']:.8f} "
            f"| PD={payload['PD']:.4f} | PF={payload['PF']:.6f}",
            flush=True,
        )
    print(f"Saved results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
