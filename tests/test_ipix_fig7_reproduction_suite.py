from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import yaml

from paper_modules.experiments.auto_experiment import apply_suite, parse_args, resolve_configs, validate_configs
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "paper_modules/configs/suites/ipix_fig7_original_stgnn_56_fullstats.yaml"
DATA_DIR = "datasets/ipix_dartmouth/processed/window4_stride4_related_official_ipixload_auto_double"
PROCESSING_MODE = "official_ipixload_auto"
POLS = {"hh", "hv", "vv", "vh"}


def test_ipix_fig7_suite_covers_exactly_56_independent_units() -> None:
    labels = json.loads((ROOT / "datasets/ipix_dartmouth/labels.json").read_text(encoding="utf-8"))
    expected_sources = {Path(name).stem for name in labels["files"]}
    expected_units = set(product(expected_sources, POLS))
    suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))

    assert suite["baseline_only"] is True
    assert suite["stats_scope"] == "full_file"
    assert suite["expected_experiment_units"] == 56
    assert len(suite["configs"]) == 56

    actual_units: set[tuple[str, str]] = set()
    for relative_path in suite["configs"]:
        config = load_config(ROOT / relative_path)
        dataset = config["dataset"]
        assert dataset["data_dir"] == DATA_DIR
        assert dataset["expected_processing_mode"] == PROCESSING_MODE
        assert config["paths"]["data_dir"] == DATA_DIR
        assert config["model"]["name"] == "original_stgnn"
        assert config["eval"]["protocol"] == "per_file_pol"
        assert config["eval"]["threshold_source"] == "train_clutter"
        assert config["eval"]["pfa_values"] == [0.001]
        assert len(dataset["sources"]) == 1
        assert len(dataset["polarizations"]) == 1
        actual_units.add((dataset["sources"][0], dataset["polarizations"][0]))

    assert actual_units == expected_units


def test_ipix_fig7_suite_passes_baseline_only_validation(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["auto_experiment", "--suite", str(SUITE), "--validate-only"])
    args = parse_args()
    apply_suite(args)
    configs = resolve_configs(args)
    validate_configs(configs, args)
    output = capsys.readouterr().out
    assert "- experiment units: 56" in output
    assert "VALIDATION_OK" in output
