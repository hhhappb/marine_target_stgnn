from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print paper module ablation commands.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("train.py")
    for config in args.configs:
        cmd = [sys.executable, str(script), "--config", str(config), "--no-progress", "--log-interval", "100"]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.max_train_windows is not None:
            cmd.extend(["--max-train-windows", str(args.max_train_windows)])
        if args.max_test_windows_per_file is not None:
            cmd.extend(["--max-test-windows-per-file", str(args.max_test_windows_per_file)])
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
