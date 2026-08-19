from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_DIR = ROOT / "paper_modules/configs/per_file_fig7_primary_strict_batch512"
OUTPUT_CONFIG_ROOT = ROOT / "paper_modules/configs/per_file_fig7_primary_strict_module_ablation"
SUITE_PATH = ROOT / "paper_modules/configs/suites/ipix_fig7_primary_strict_module_ablation_60ep.yaml"

COMMON_MODEL = {
    "name": "sfe_replacement_stgnn",
    "pulses": 4,
    "range_cells": 14,
}
COMMON_FEATURES = {
    "type": "real_imag",
    "hidden_channels": 32,
    "out_channels": 64,
}
ORIGINAL_SPATIAL = {
    "type": "original_stfe",
    "stage1_out_channels": 128,
    "stage2_out_channels": 512,
}
RADAR_PRIOR_SPATIAL = {
    "type": "radar_prior_dynamic_sfe",
    "stage1_out_channels": 128,
    "stage2_out_channels": 512,
    "static_gamma": 0.5,
    "static_delta": 5,
    "static_weight": 0.7,
    "dynamic_topk": 2,
    "dynamic_temperature": 0.2,
    "dropout": 0.1,
}
ORIGINAL_TEMPORAL = {
    "type": "stgnn_tfe",
    "stage1_out_channels": 256,
    "stage2_out_channels": 1024,
    "out_channels": 1024,
}
TEMPORAL_STAGE1 = {
    "type": "scale_normalized_difference_decomposition_tfe",
    "stage1_out_channels": 256,
    "beta_max": 0.1,
    "eps": 1e-6,
    "use_modulation": True,
    "collect_diagnostics": False,
}
TEMPORAL_STAGE2 = {
    "type": "scale_normalized_difference_decomposition_tfe",
    "stage2_out_channels": 1024,
    "out_channels": 1024,
    "beta_max": 0.1,
    "eps": 1e-6,
    "use_modulation": True,
    "collect_diagnostics": False,
}

VARIANTS = {
    "sfe_replacement_radar_prior_dynamic": {
        "question": "仅替换空间 SFE 是否改善 primary_strict IPIX 检测",
        "spatial_graph": RADAR_PRIOR_SPATIAL,
    },
    "tfe_replacement_scale_normalized_difference": {
        "question": "仅替换时间 TFE 是否改善 primary_strict IPIX 检测",
        "spatial_graph": ORIGINAL_SPATIAL,
        "temporal1": TEMPORAL_STAGE1,
        "temporal2": TEMPORAL_STAGE2,
    },
    "sfe_tfe_replacement_radar_prior_scale_normalized": {
        "question": "同时替换空间 SFE 与时间 TFE 是否改善 primary_strict IPIX 检测",
        "spatial_graph": RADAR_PRIOR_SPATIAL,
        "temporal1": TEMPORAL_STAGE1,
        "temporal2": TEMPORAL_STAGE2,
    },
}


def write_yaml_if_new(path: Path, payload: dict[str, object]) -> None:
    """只新增或复用完全一致的配置，禁止静默覆盖已有实验定义。"""
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"已有配置内容不同，拒绝覆盖：{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def build_variant(base: dict[str, object], variant_name: str) -> dict[str, object]:
    variant = VARIANTS[variant_name]
    config = deepcopy(base)
    experiment = dict(config.get("experiment", {}))
    experiment.update(
        {
            "protocol_family": "diagnostic",
            "protocol_version": "ipix_primary_strict_module_ablation_v1",
            "question": variant["question"],
        }
    )
    config["experiment"] = experiment
    config["model"] = deepcopy(COMMON_MODEL)
    config["radar_features"] = deepcopy(COMMON_FEATURES)
    config["spatial_graph"] = deepcopy(variant["spatial_graph"])
    config["temporal"] = deepcopy(ORIGINAL_TEMPORAL)
    config["clutter_gate"] = {"enabled": False}
    config["detection_head"] = {"hidden_channels": 512}
    if "temporal1" in variant:
        config["temporal1"] = deepcopy(variant["temporal1"])
        config["temporal2"] = deepcopy(variant["temporal2"])
    return config


def main() -> None:
    base_paths = sorted(BASE_CONFIG_DIR.glob("original_stgnn_label*.yaml"))
    if len(base_paths) != 56:
        raise RuntimeError(f"基准配置必须为 56 个，实际为 {len(base_paths)}。")

    suite_configs: list[str] = []
    for base_path in base_paths:
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        suite_configs.append(base_path.relative_to(ROOT).as_posix())
        suffix = base_path.stem.removeprefix("original_stgnn_")
        for variant_name in VARIANTS:
            output_path = OUTPUT_CONFIG_ROOT / variant_name / f"{variant_name}_{suffix}.yaml"
            config = build_variant(base, variant_name)
            paths = dict(config.get("paths", {}))
            paths["save_dir"] = (
                Path("logs/training/per_file_fig7_primary_strict_module_ablation")
                / variant_name
                / suffix
            ).as_posix()
            config["paths"] = paths
            write_yaml_if_new(output_path, config)
            suite_configs.append(output_path.relative_to(ROOT).as_posix())

    suite = {
        "name": "ipix_fig7_primary_strict_module_ablation_60ep",
        "description": "IPIX 56条件单seed配对比较：原始ST-GNN、仅空间、仅时间、时空联合",
        "run_root": "logs/training",
        "target_pfa": 0.001,
        "stop_on_failure": True,
        "stats_scope": "full_file",
        "expected_experiment_units": 56,
        "configs": suite_configs,
    }
    write_yaml_if_new(SUITE_PATH, suite)
    print(f"生成候选配置：{len(suite_configs) - len(base_paths)}")
    print(f"suite 配置总数：{len(suite_configs)}")
    print(SUITE_PATH.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
