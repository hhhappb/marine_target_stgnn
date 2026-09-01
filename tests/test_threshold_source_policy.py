from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from paper_modules.experiments.train import evaluate_files


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "threshold_source",
    [None, "", "test_" + "diagnostic_current_eval", "test_" + "diagnostic_clutter"],
)
def test_evaluate_files_rejects_non_train_threshold_sources_before_reading_data(
    threshold_source: str | None,
) -> None:
    with pytest.raises(ValueError, match="必须显式设置"):
        evaluate_files(
            nn.Identity(),
            [],
            batch_size=1,
            device=torch.device("cpu"),
            pfa_values=[0.001],
            threshold_source=threshold_source,
        )


def test_active_experiment_sources_contain_no_forbidden_threshold_implementation() -> None:
    forbidden = (
        "matched" + "-PF",
        "matched_" + "pf",
        "matched" + "_test_" + "diagnostic",
        "lag_" + "matched" + "_" + "pf",
        "test_" + "diagnostic_current_eval",
        "test_" + "diagnostic_clutter",
    )
    roots = [
        ROOT / "AGENTS.md",
        ROOT / "docs" / "experiment_protocol.md",
        ROOT / "paper_modules",
        ROOT / "scripts",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".py", ".md", ".yaml", ".yml"}
            )

    residuals = {
        str(path.relative_to(ROOT)): token
        for path in files
        for token in forbidden
        if token.lower() in path.read_text(encoding="utf-8", errors="ignore").lower()
    }
    assert residuals == {}


def test_removed_test_threshold_generators_are_absent() -> None:
    assert not (ROOT / "paper_modules/experiments/ipix_threshold_migration_diagnostic.py").exists()
    assert not (ROOT / "scripts/build_ipix_threshold_fix_report.py").exists()
    assert not (ROOT / "scripts/extract_ipix_report.py").exists()
