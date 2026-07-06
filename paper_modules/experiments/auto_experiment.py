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
    parser.add_argument("--validate-only", action="store_true", help="Validate selected per-file configs without launching training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_suite(args)
    configs = resolve_configs(args)
    if not configs:
        raise SystemExit("No configs selected. Pass --configs ... or --all-configs.")
    if args.validate_only:
        validate_configs(configs, args)
        return

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


def validate_configs(configs: list[Path], args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    require_train_only_stats = "per_file_fig7" in args.name

    for config_path in configs:
        base_cfg = load_config(config_path)
        seeds = args.seeds or [int(get_config_value(base_cfg, "train.seed"))]
        for seed in seeds:
            cfg = prepare_run_config(args, base_cfg, seed)
            meta = config_metadata(cfg)
            meta["config"] = str(config_path)
            meta["seed"] = seed
            rows.append(meta)

            label = f"{config_path.name} seed{seed}"
            if meta["dataset_type"] != "ipix_window":
                errors.append(f"{label}: dataset.type 必须是 ipix_window。")
            if len(meta["sources"]) != 1:
                errors.append(f"{label}: per-file 配置必须且只能声明 1 个 dataset.sources。")
            if len(meta["polarizations"]) != 1:
                errors.append(f"{label}: per-file 配置必须且只能声明 1 个 dataset.polarizations。")
            if meta["eval_protocol"] != "per_file_pol":
                errors.append(f"{label}: eval.protocol 必须是 per_file_pol。")
            if meta["threshold_source"] != "train_clutter":
                errors.append(f"{label}: eval.threshold_source 必须是 train_clutter。")
            if meta["paths_data_dir"] != meta["dataset_data_dir"]:
                errors.append(f"{label}: paths.data_dir 与 dataset.data_dir 不一致。")
            if require_train_only_stats and "stats_train_only" not in meta["data_dir"]:
                errors.append(f"{label}: Fig.7 hardpoint 正式配置必须使用 train-only stats 数据目录。")

    pairs: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if len(row["sources"]) == 1 and len(row["polarizations"]) == 1:
            pairs.setdefault((row["sources"][0], row["polarizations"][0], int(row["seed"])), []).append(row)

    for pair_key, pair_rows in sorted(pairs.items()):
        baseline_rows = [row for row in pair_rows if row["model_name"] == "original_stgnn"]
        candidate_rows = [row for row in pair_rows if row["model_name"] != "original_stgnn"]
        pair_label = f"{pair_key[0]}/{pair_key[1]}/seed{pair_key[2]}"
        if len(baseline_rows) != 1:
            errors.append(f"{pair_label}: 需要且只能有 1 个 original_stgnn baseline，实际 {len(baseline_rows)} 个。")
            continue
        if not candidate_rows:
            errors.append(f"{pair_label}: 缺少待比较模型配置。")
            continue
        baseline = baseline_rows[0]
        for candidate in candidate_rows:
            mismatches = comparable_mismatches(candidate, baseline)
            if mismatches:
                errors.append(f"{pair_label}: {Path(candidate['config']).name} 与 baseline 字段不一致: {', '.join(mismatches)}。")

    if args.seeds is None:
        warnings.append("当前 suite 只使用配置内 seed；seed42 结果只能作方向读数，正式结论建议至少 3 个 seed。")

    print("Config validation summary", flush=True)
    print(f"- configs: {len(configs)}", flush=True)
    print(f"- run units: {len(rows)}", flush=True)
    print(f"- pairs: {len(pairs)}", flush=True)
    if warnings:
        print("Warnings:", flush=True)
        for item in warnings:
            print(f"- {item}", flush=True)
    if errors:
        print("Errors:", flush=True)
        for item in errors:
            print(f"- {item}", flush=True)
        raise SystemExit("Config validation failed.")
    print("VALIDATION_OK", flush=True)


def prepare_run_config(args: argparse.Namespace, base_cfg: dict[str, Any], seed: int, save_dir: Path | None = None) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    if save_dir is not None:
        cfg.setdefault("paths", {})["save_dir"] = str(save_dir)
    cfg.setdefault("train", {})["seed"] = seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    return cfg


def config_metadata(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    eval_cfg = config.get("eval", {})
    train_cfg = config.get("train", {})
    paths_cfg = config.get("paths", {})
    pols = _as_list(dataset_cfg.get("polarizations", config.get("ipix", {}).get("polarizations", []))) or []
    sources = _as_list(dataset_cfg.get("sources", dataset_cfg.get("source"))) or []
    paths_data_dir = str(paths_cfg.get("data_dir", ""))
    dataset_data_dir = str(dataset_cfg.get("data_dir", paths_data_dir))
    augmentation = json.dumps(dataset_cfg.get("augment", {}), sort_keys=True, ensure_ascii=False)
    return {
        "model_name": str(config.get("model", {}).get("name", "")),
        "dataset_type": str(dataset_cfg.get("type", "ipix_window")),
        "data_dir": paths_data_dir,
        "paths_data_dir": paths_data_dir,
        "dataset_data_dir": dataset_data_dir,
        "train_augmentation": augmentation,
        "sources": sources,
        "source": sources[0] if len(sources) == 1 else "",
        "polarizations": pols,
        "polarization": pols[0] if len(pols) == 1 else "",
        "eval_protocol": str(eval_cfg.get("protocol", "per_file_pol")),
        "threshold_source": str(eval_cfg.get("threshold_source", "test_diagnostic_current_eval")),
        "epochs": int(train_cfg.get("epochs", 0)),
        "batch_size": int(train_cfg.get("batch_size", 0)),
        "learning_rate": float(train_cfg.get("learning_rate", 0.0)),
    }


def comparable_mismatches(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    fields = [
        "data_dir",
        "train_augmentation",
        "epochs",
        "batch_size",
        "learning_rate",
        "threshold_source",
        "eval_protocol",
    ]
    return [field for field in fields if candidate.get(field) != baseline.get(field)]


def run_one(args: argparse.Namespace, config_path: Path, base_cfg: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any] | None:
    run_name = f"{safe_name(config_path.stem)}_seed{seed}"
    save_dir = run_dir / "runs" / run_name
    cfg = prepare_run_config(args, base_cfg, seed, save_dir=save_dir)

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
    row.update(config_metadata(cfg))
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
        "protocol": str(payload.get("protocol", "")),
        "git_commit": str(payload.get("git_commit", "")),
        "train_files": ";".join(payload.get("train_files", [])),
        "test_files": ";".join(payload.get("test_files", [])),
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
    baselines = {
        pair_key(row): row
        for row in completed
        if row.get("model_name") == "original_stgnn" and pair_key(row) is not None
    }
    rows_for_output = []
    for row in rows:
        item = dict(row)
        key = pair_key(item)
        baseline = baselines.get(key) if key is not None else None
        if baseline and "PD" in item:
            item["pair_baseline_run"] = baseline["run_name"]
            item["delta_PD_vs_pair_baseline"] = float(item["PD"]) - float(baseline["PD"])
            item["delta_PF_vs_pair_baseline"] = float(item["PF"]) - float(baseline["PF"])
            item["pair_mismatch_fields"] = ", ".join(comparable_mismatches(item, baseline))
        rows_for_output.append(item)
    write_csv(run_dir / "metrics.csv", rows_for_output)
    write_summary(run_dir / "summary.md", rows_for_output, target_pfa)
    write_artifacts(run_dir / "artifacts.txt", rows_for_output)


def write_summary(path: Path, rows: list[dict[str, Any]], target_pfa: float | None) -> None:
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
    lines.extend(
        [
            "## Paired Comparison",
            "",
            "| Pair | Run | Model | PD | PF | Delta PD vs pair baseline | Delta PF vs pair baseline | Pair mismatch |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    paired = sorted(
        [row for row in rows if row.get("status") == "completed" and "PD" in row],
        key=lambda row: (str(row.get("source", "")), str(row.get("polarization", "")), str(row.get("run_name", ""))),
    )
    for row in paired:
        pair = f"{row.get('source', '')}/{row.get('polarization', '')}/seed{row.get('seed', '')}"
        delta_pd = row.get("delta_PD_vs_pair_baseline", 0.0)
        delta_pf = row.get("delta_PF_vs_pair_baseline", 0.0)
        mismatch = row.get("pair_mismatch_fields", "")
        lines.append(
            f"| {pair} | {row['run_name']} | {row.get('model_name', '')} | {float(row['PD']):.6f} | "
            f"{float(row['PF']):.6f} | {float(delta_pd):+.6f} | {float(delta_pf):+.6f} | {mismatch or 'ok'} |"
        )
    lines.extend(["", "## Diagnostic Ranking", "", "| Rank | Run | PD | PF | Accuracy | Worst file/pol |", "|---:|---|---:|---:|---:|---|"])
    for idx, row in enumerate(ranked, start=1):
        worst = f"{row.get('worst_source', '')}/{row.get('worst_polarization', '')}"
        lines.append(
            f"| {idx} | {row['run_name']} | {float(row['PD']):.6f} | {float(row['PF']):.6f} | "
            f"{float(row['accuracy']):.6f} | {worst} |"
        )
    failed = [row for row in rows if row.get("status") != "completed"]
    if failed:
        lines.extend(["", "## Failed Runs", ""])
        for row in failed:
            lines.append(f"- {row['run_name']}: status={row.get('status')} returncode={row.get('returncode')} stdout={row.get('stdout_log')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pair_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    source = str(row.get("source", ""))
    pol = str(row.get("polarization", ""))
    if not source or not pol or "seed" not in row:
        return None
    return source, pol, int(row["seed"])


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


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "run"


if __name__ == "__main__":
    main()
