# REMOVED.md — GDCM 包整理记录

本文件记录 `e4d2_gdcm/` 从 `e4d2_delta_cs/` 演进时移除/替换的内容。

## 核心改动（RDJ → GDCM 判决头）

| 位置 | 原内容 | 新内容 | 说明 |
|------|------|------|------|
| `models/e4d2.py` | RdjDetector（[Δ, \|Δ\|]→256ch） | **新增 GdcmDetector**（组归一化方向 + 通道级/组级幅度） | 128ch 按 GroupNorm 的 8 组划分；E4D2 增加 gdcm_g0..g4 |
| `models/__init__.py` | 导出 RdjDetector | 新增导出 GdcmDetector | — |
| `experiments/train_sdrdsp.py` | 幅相型输入 | 重写为统计型 E4D2 + `--detector` 参数 | 支持 rdj/gdcm_g0..g4 |

## 保留未动

- `E4Backbone`（STGNN 1024ch + PhaseEncoder/AmpEncoder/CVOCA 64ch + Global 128ch）
- `RdjDetector` 保留（作为 G0 baseline 与对照）
- **判决网络分类器与 RDJ 完全一致**：concat → Conv1d→128 → GELU → Dropout(0.1) → Conv1d(128→2)，保证性能变化可归因于 D_s/D_a 的重新定义
- `losses/`、`utils/`、`configs/`、`reports/`
- 包名 import：`e4d2_delta_cs.*` → `e4d2_gdcm.*`（sed 批量替换）

## 设计差异（GDCM vs RDJ / DA）

| 环节 | RDJ (G0) | DA（上轮） | GDCM (G1-G4) |
|------|------|------|------|
| D_s | Δ | Δ/ρ（全体 128ch 全局 L2） | Δ/ρ_g（**分组** L2，G=8 组各 16ch） |
| D_a | \|Δ\| | 单通道标量 ρ/(ρ+b) | **通道级** \|Δ\| / \|Δ\|/(\|U_d\|+\|U_h\|) / log(1+\|Δ\|) / ρ_g |
| D_a 维度 | 128 | 1 | 128（g4 为 8） |
| 分类器 | 相同 | 相同 | 相同（唯一不变约束） |

> **最终采用 A0（`gdcm_a0`，仅通道级 log 强度 D_a=log(1+|Δ|)，128ch）**：最简洁、最稳定的判决表示（AUC 0.9842±0.002 为全变体最高且方差最小，PdL@1e-3/1e-2 与 G3 持平）。已设为包默认（`E4D2()` → `gdcm_a0`，`configs/default.yaml`、`train_sdrdsp.py --detector` 默认值均已同步）。组方向变体 g0-g4 保留供消融对照。

## 已知遗留

- `experiments/train_ipix.py` 仍为旧风格 IPIX 训练入口，未适配 `--detector` 参数；
  如需 IPIX 训练请参考 `train_sdrdsp.py` 改造。
- `compute_polar_raw` / `compute_cs_polar` / `compute_delta_cs` 保留在 preprocess.py 中仅作对照。
