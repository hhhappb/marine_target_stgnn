from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.config import get_config_value, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper module experiments and summarize comparable metrics.")
    parser.add_argument("--suite", type=Path, default=None, help="Experiment suite yaml with configs, seeds, and run overrides.")
    parser.add_argument("--configs", nargs="*", type=Path, default=[])
    parser.add_argument("--all-configs", action="store_true", help="Run every yaml under paper_modules/configs.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
    parser.add_argument("--target-pfa", type=float, default=None)
    parser.add_argument("--run-root", type=Path, default=Path("logs/training"))
    parser.add_argument("--name", type=str, default="auto_modules")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_suite(args)
    configs = resolve_configs(args)
    if not configs:
        raise SystemExit("No configs selected. Pass --configs ... or --all-configs.")

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{safe_name(args.name)}"
    run_dir = args.run_root / run_id
    rows: list[dict[str, Any]] = []

    if not args.dry_run:
        (run_dir / "configs").mkdir(parents=True, exist_ok=True)
        (run_dir / "runs").mkdir(parents=True, exist_ok=True)

    for config_path in configs:
        base_cfg = load_config(config_path)
        seeds = args.seeds or [int(get_config_value(base_cfg, "train.seed"))]
        for seed in seeds:
            row = run_one(args, config_path, base_cfg, seed, run_dir)
            if row:
                rows.append(row)
                write_outputs(run_dir, rows, args.target_pfa)
            if row and row.get("status") == "failed" and args.stop_on_failure:
                raise SystemExit(f"Experiment failed: {row['run_name']}")

    if args.dry_run:
        return

    write_outputs(run_dir, rows, args.target_pfa)
    print(f"Saved automation summary: {run_dir / 'summary.md'}", flush=True)


def apply_suite(args: argparse.Namespace) -> None:
    if args.suite is None:
        return
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: PyYAML. Install with `python -m pip install PyYAML`.") from exc

    suite_path = (ROOT / args.suite).resolve() if not args.suite.is_absolute() else args.suite.resolve()
    if not suite_path.exists():
        raise SystemExit(f"Suite not found: {args.suite}")
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}

    args.configs = [Path(item) for item in suite.get("configs", args.configs)]
    args.all_configs = bool(suite.get("all_configs", args.all_configs))
    args.seeds = suite.get("seeds", args.seeds)
    args.target_pfa = suite.get("target_pfa", args.target_pfa)
    args.run_root = Path(suite.get("run_root", args.run_root))
    args.name = str(suite.get("name", args.name))
    args.stop_on_failure = bool(suite.get("stop_on_failure", args.stop_on_failure))

    overrides = suite.get("overrides", {})
    args.epochs = overrides.get("epochs", args.epochs)
    args.batch_size = overrides.get("batch_size", args.batch_size)
    args.num_workers = overrides.get("num_workers", args.num_workers)
    args.max_train_windows = overrides.get("max_train_windows", args.max_train_windows)
    args.max_test_windows_per_file = overrides.get("max_test_windows_per_file", args.max_test_windows_per_file)


def resolve_configs(args: argparse.Namespace) -> list[Path]:
    configs = [path for path in args.configs]
    if args.all_configs:
        configs.extend(sorted((ROOT / "paper_modules" / "configs").glob("*.yaml")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in configs:
        resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if resolved not in seen:
            if not resolved.exists():
                raise SystemExit(f"Config not found: {path}")
            unique.append(resolved)
            seen.add(resolved)
    return unique


def run_one(args: argparse.Namespace, config_path: Path, base_cfg: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any] | None:
    run_name = f"{safe_name(config_path.stem)}_seed{seed}"
    save_dir = run_dir / "runs" / run_name
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("paths", {})["save_dir"] = str(save_dir)
    cfg.setdefault("train", {})["seed"] = seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    config_snapshot = run_dir / "configs" / f"{run_name}.yaml"
    stdout_path = save_dir / "stdout.log"
    train_script = ROOT / "paper_modules" / "experiments" / "train.py"
    cmd = [sys.executable, str(train_script), "--config", str(config_snapshot), "--no-progress", "--log-interval", "100"]
    if args.max_train_windows is not None:
        cmd.extend(["--max-train-windows", str(args.max_train_windows)])
    if args.max_test_windows_per_file is not None:
        cmd.extend(["--max-test-windows-per-file", str(args.max_test_windows_per_file)])

    if args.dry_run:
        print(" ".join(cmd), flush=True)
        return None

    save_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(config_snapshot, cfg)
    t0 = time.time()
    status = "completed"
    returncode = 0
    with stdout_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8",
                errors="replace",
            )
            print(safe_line, end="", flush=True)
            log.write(line)
        returncode = proc.wait()
        if returncode != 0:
            status = "failed"

    duration = time.time() - t0
    row = base_row(run_name, config_path, config_snapshot, save_dir, stdout_path, seed, status, returncode, duration)
    if status == "completed":
        row.update(read_metrics(save_dir / "eval_results.json", args.target_pfa))
    return row


def read_metrics(results_path: Path, target_pfa: float | None) -> dict[str, Any]:
    if not results_path.exists():
        return {"status": "missing_results", "results_path": str(results_path)}
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    pfa_map = payload.get("pfa", {})
    key = select_pfa_key(pfa_map, target_pfa)
    result = pfa_map[key]
    tp, fn, fp, tn = int(result["TP"]), int(result["FN"]), int(result["FP"]), int(result["TN"])
    total = tp + fn + fp + tn
    per_file = result.get("per_file", [])
    worst = min(per_file, key=lambda item: float(item.get("PD", 0.0))) if per_file else {}
    return {
        "results_path": str(results_path),
        "target_pfa": float(key),
        "threshold_source": str(payload.get("threshold_source", "test_diagnostic_current_eval")),
        "threshold": float(result["threshold"]),
        "PD": float(result["PD"]),
        "PF": float(result["PF"]),
        "accuracy": (tp + tn) / total if total else 0.0,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "num_eval_clutter_bins": int(payload.get("num_clutter_bins", 0)),
        "num_clutter_bins_for_threshold": int(
            payload.get("num_clutter_bins_for_threshold", payload.get("num_clutter_bins", 0))
        ),
        "worst_source": worst.get("source", ""),
        "worst_polarization": worst.get("polarization", ""),
        "worst_PD": worst.get("PD", ""),
    }


def select_pfa_key(pfa_map: dict[str, Any], target_pfa: float | None) -> str:
    if not pfa_map:
        raise ValueError("eval_results.json has no pfa entries.")
    if target_pfa is None:
        target_pfa = 0.001
    return min(pfa_map, key=lambda key: abs(float(key) - float(target_pfa)))


def write_outputs(run_dir: Path, rows: list[dict[str, Any]], target_pfa: float | None) -> None:
    completed = [row for row in rows if row.get("status") == "completed" and "PD" in row]
    baseline = completed[0] if completed else None
    rows_for_output = []
    for row in rows:
        item = dict(row)
        if baseline and "PD" in item:
            item["delta_PD_vs_baseline"] = float(item["PD"]) - float(baseline["PD"])
            item["delta_PF_vs_baseline"] = float(item["PF"]) - float(baseline["PF"])
        rows_for_output.append(item)
    write_csv(run_dir / "metrics.csv", rows_for_output)
    write_summary(run_dir / "summary.md", rows_for_output, baseline, target_pfa)
    write_artifacts(run_dir / "artifacts.txt", rows_for_output)


def write_summary(path: Path, rows: list[dict[str, Any]], baseline: dict[str, Any] | None, target_pfa: float | None) -> None:
    ranked = sorted(
        [row for row in rows if row.get("status") == "completed" and "PD" in row],
        key=lambda row: (-float(row["PD"]), float(row["PF"])),
    )
    threshold_sources = sorted({str(row.get("threshold_source", "")) for row in ranked if row.get("threshold_source")})
    lines = [
        "# 自动化模块实验汇总",
        "",
        f"- target_pfa: {target_pfa if target_pfa is not None else 0.001}",
        f"- threshold_source: {', '.join(threshold_sources) if threshold_sources else 'unknown'}",
        "- 说明：train_clutter 使用训练集杂波分数定阈值；test_diagnostic_current_eval 使用测试集杂波分数定阈值，只适合作为模块筛选诊断。",
        "",
    ]
    if baseline:
        lines.extend(
            [
                "## Baseline",
                "",
                f"- {baseline['run_name']}: PD={float(baseline['PD']):.6f}, PF={float(baseline['PF']):.6f}, accuracy={float(baseline['accuracy']):.6f}",
                "",
            ]
        )
    lines.extend(["## Ranking", "", "| Rank | Run | PD | PF | Accuracy | Delta PD | Worst file/pol |", "|---:|---|---:|---:|---:|---:|---|"])
    for idx, row in enumerate(ranked, start=1):
        delta = row.get("delta_PD_vs_baseline", 0.0)
        worst = f"{row.get('worst_source', '')}/{row.get('worst_polarization', '')}"
        lines.append(
            f"| {idx} | {row['run_name']} | {float(row['PD']):.6f} | {float(row['PF']):.6f} | "
            f"{float(row['accuracy']):.6f} | {float(delta):+.6f} | {worst} |"
        )
    failed = [row for row in rows if row.get("status") != "completed"]
    if failed:
        lines.extend(["", "## Failed Runs", ""])
        for row in failed:
            lines.append(f"- {row['run_name']}: status={row.get('status')} returncode={row.get('returncode')} stdout={row.get('stdout_log')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = []
    for row in rows:
        lines.append(f"[{row['run_name']}]")
        lines.append(f"config_snapshot={row.get('config_snapshot', '')}")
        lines.append(f"stdout_log={row.get('stdout_log', '')}")
        lines.append(f"save_dir={row.get('save_dir', '')}")
        lines.append(f"results={row.get('results_path', '')}")
        lines.append(f"checkpoint={Path(row.get('save_dir', '')) / 'best_model.pth'}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def base_row(
    run_name: str,
    config_path: Path,
    config_snapshot: Path,
    save_dir: Path,
    stdout_path: Path,
    seed: int,
    status: str,
    returncode: int,
    duration: float,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "config": str(config_path),
        "config_snapshot": str(config_snapshot),
        "save_dir": str(save_dir),
        "stdout_log": str(stdout_path),
        "seed": seed,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(duration, 3),
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: PyYAML. Install with `python -m pip install PyYAML`.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "run"


if __name__ == "__main__":
    main()
