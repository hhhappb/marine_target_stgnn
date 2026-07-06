from __future__ import annotations

from pathlib import Path
from typing import Any

from .ipix_window import IpixWindowDataset, list_split_files
from .scr_npz import ScrNpzDataset


def build_dataset(config: dict[str, Any], split: str, **overrides: Any):
    dataset_cfg = config.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "ipix_window"))
    seed = int(overrides.get("seed", config.get("train", {}).get("seed", 42)))
    max_windows = overrides.get("max_windows", dataset_cfg.get(f"max_{split}_windows"))

    if dataset_type == "ipix_window":
        data_dir = Path(dataset_cfg.get("data_dir", config.get("paths", {}).get("data_dir", "")))
        pols = dataset_cfg.get("polarizations", config.get("ipix", {}).get("polarizations", []))
        sources = _as_list(dataset_cfg.get("sources", dataset_cfg.get("source")))
        if not data_dir:
            raise ValueError("dataset.type=ipix_window 需要 dataset.data_dir 或 paths.data_dir。")
        if not pols:
            raise ValueError("dataset.type=ipix_window 需要 dataset.polarizations 或 ipix.polarizations。")
        files = list_split_files(data_dir, split, list(pols), sources=sources)
        if not files:
            raise ValueError(f"没有找到 IPIX {split} 文件：data_dir={data_dir}, sources={sources}, polarizations={pols}")
        return IpixWindowDataset(files, max_windows=max_windows, seed=seed)

    if dataset_type == "scr_npz":
        data_dir = Path(dataset_cfg.get("data_dir", config.get("paths", {}).get("data_dir", "")))
        if not data_dir:
            raise ValueError("dataset.type=scr_npz 需要 dataset.data_dir 或 paths.data_dir。")
        return ScrNpzDataset(
            data_dir=data_dir,
            split=split,
            scr=overrides.get("scr", dataset_cfg.get("scr")),
            max_windows=max_windows,
            seed=seed,
        )

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
