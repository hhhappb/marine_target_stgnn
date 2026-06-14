from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "data_dir": "data/simulated",
        "raw_dir": "datasets/ipix_dartmouth/raw",
        "labels": "datasets/ipix_dartmouth/labels.json",
        "processed_dir": "datasets/ipix_dartmouth/processed",
        "save_dir": "checkpoints",
    },
    "ipix": {
        "polarizations": ["hh", "hv", "vv", "vh"],
        "window": 4,
        "stride": 4,
        "train_fraction": 0.6,
        "target_policy": "related",
    },
    "model": {
        "pulses": 10,
        "range_cells": 256,
    },
    "train": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_workers": 0,
        "seed": 42,
    },
    "eval": {
        "pfa_values": [0.0001, 0.001, 0.01],
        "target_pfa": 0.001,
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    path = Path(config_path)
    with path.open("rb") as handle:
        user_config = tomllib.load(handle)
    return merge_config(config, user_config)


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
