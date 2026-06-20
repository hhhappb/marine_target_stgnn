from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_modules.experiments.train import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper module checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.argv = [
        sys.argv[0],
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--eval-only",
    ]
    if args.batch_size is not None:
        sys.argv.extend(["--batch-size", str(args.batch_size)])
    if args.max_test_windows_per_file is not None:
        sys.argv.extend(["--max-test-windows-per-file", str(args.max_test_windows_per_file)])
    train_main()
