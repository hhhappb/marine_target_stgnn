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
        reject_retired_ipix_cross_file_split(dataset_cfg)
        sources = _as_list(dataset_cfg.get("sources", dataset_cfg.get("source")))
        if not data_dir:
            raise ValueError("dataset.type=ipix_window 需要 dataset.data_dir 或 paths.data_dir。")
        if not pols:
            raise ValueError("dataset.type=ipix_window 需要 dataset.polarizations 或 ipix.polarizations。")
        files = list_split_files(data_dir, split, list(pols), sources=sources)
        if not files:
            raise ValueError(f"没有找到 IPIX {split} 文件：data_dir={data_dir}, sources={sources}, polarizations={pols}")
        augment_cfg = dataset_cfg.get("augment", {}) if split == "train" else {}
        range_roll = augment_cfg.get("range_roll") if isinstance(augment_cfg, dict) else None
        window_fraction_range = dataset_cfg.get(f"{split}_window_fraction_range")
        default_label_policy = str(dataset_cfg.get("label_policy", "stored"))
        if split == "train":
            label_policy = str(dataset_cfg.get("train_label_policy", default_label_policy))
        else:
            label_policy = str(
                dataset_cfg.get("evaluation_label_policy", default_label_policy)
            )
        return IpixWindowDataset(
            files,
            max_windows=max_windows,
            seed=seed,
            range_roll=range_roll,
            window_fraction_range=window_fraction_range,
            label_policy=label_policy,
            secondary_echo_policy=str(
                dataset_cfg.get("secondary_echo_policy", "stored")
            ),
            expected_processing_mode=dataset_cfg.get("expected_processing_mode"),
        )

    if dataset_type == "scr_npz":
        data_dir = Path(dataset_cfg.get("data_dir", config.get("paths", {}).get("data_dir", "")))
        if not data_dir:
            raise ValueError("dataset.type=scr_npz 需要 dataset.data_dir 或 paths.data_dir。")
        model_cfg = config.get("model", {})
        return ScrNpzDataset(
            data_dir=data_dir,
            split=split,
            scr=overrides.get("scr", dataset_cfg.get("scr")),
            max_windows=max_windows,
            seed=seed,
            norm=overrides.get("norm"),
            normalization=str(dataset_cfg.get("normalization", "train_standardize_clip")),
            protocol=dataset_cfg.get("protocol"),
            expected_pulses=int(model_cfg["pulses"]) if "pulses" in model_cfg else None,
            expected_range_cells=int(model_cfg["range_cells"]) if "range_cells" in model_cfg else None,
        )

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]

def reject_retired_ipix_cross_file_split(dataset_cfg: dict[str, Any]) -> None:
    retired_keys = sorted(
        key
        for key in (
            "train_sources",
            "test_sources",
            "require_disjoint_train_test_sources",
        )
        if key in dataset_cfg
    )
    if retired_keys:
        raise ValueError(
            "IPIX 跨文件训练/测试协议已退役，禁止使用配置字段 "
            f"{retired_keys}。PAX-L1 仅允许同一文件-极化条件内的 60/40 划分。"
        )
