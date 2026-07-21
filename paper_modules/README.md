# 论文模块实验包说明

`paper_modules/` 只放严格对齐论文 ST-GNN 主干的可替换实验模块。当前主线不是重新设计一条模型流水线，而是在论文结构中做受控替换：

```text
Feature -> SFE1 -> TFE1 -> SFE2 -> TFE2 -> DetectionHead
```

## 主线入口

| 位置 | 文件 | 作用 |
|---|---|---|
| SFE 替换骨架 | `models/sfe_replacement_stgnn.py` | 保持两级 SFE/TFE 顺序，只替换 SFE 实现 |
| 原 SFE 复现 | `models/modules/spatial_graphs/original_stfe.py` | 固定相邻距离图 + additive GAT |
| 空间图候选 | `models/modules/spatial_graphs/` | 作为 SFE1/SFE2 的替换实现 |
| 输入特征候选 | `models/modules/radar_features/` | 只能在论文 FT/输入特征位置做受控替换 |

已删除旧的 `ExperimentalSTGNN` 自定义骨架和对应配置。旧骨架结果不能作为正式实验结论。

## 当前有效配置

| 实验 | 配置 |
|---|---|
| 原 SFE 对齐基线 | `configs/real_imag_sfe_replacement_original_sfe.yaml` |
| 雷达先验动态图替换 SFE | `configs/real_imag_sfe_replacement_radar_prior_dynamic_sfe.yaml` |
| 原 SFE 观察时间对照 | `configs/real_imag_sfe_replacement_original_sfe_p16.yaml`、`configs/real_imag_sfe_replacement_original_sfe_p32.yaml` |
| 雷达先验动态图观察时间对照 | `configs/real_imag_sfe_replacement_radar_prior_dynamic_sfe_p16.yaml`、`configs/real_imag_sfe_replacement_radar_prior_dynamic_sfe_p32.yaml` |
| 空间替换快筛 suite | `configs/suites/sfe_replacement_spatial_graph_screening.yaml` |
| 观察时间快筛 suite | `configs/suites/radar_prior_dynamic_sfe_observation_screening.yaml` |
| 原 I/Q 输入通道 | `configs/feature_replacement_original_iq.yaml` |
| I/Q + 幅相输入通道 | `configs/feature_replacement_iq_amp_phase.yaml` |
| I/Q + 幅相差分输入通道 | `configs/feature_replacement_iq_amp_phase_diffs.yaml` |
| 旧 SDRDSP 小样本诊断 | `configs/repro_original_stgnn_scr256.yaml` |
| ST-GNN SDRDSP Fig.9 v2 局部裁剪复现 | `configs/repro_original_stgnn_sdrdsp_strict256_v2.yaml` |
| IPIX Fig.7 困难点 per-file 对照 | `configs/suites/per_file_fig7_hardpoints_b.yaml` |
| IPIX Fig.7 困难点 range-roll 增广对照 | `configs/suites/per_file_fig7_hardpoints_shift_aug_b.yaml` |
| IPIX Fig.7 困难点三臂对照 | `configs/suites/per_file_fig7_hardpoints_three_arm.yaml` |
| 输入通道快筛 suite | `configs/suites/feature_replacement_input_channel_screening.yaml` |

## 统一接口

```python
logits = model(E_complex)
```

```text
E_complex: [B, P, N]
logits:    [B, 2, N]
```

## 常用命令

运行原 SFE 对齐基线：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\real_imag_sfe_replacement_original_sfe.yaml
```

运行 SFE 替换快筛：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\sfe_replacement_spatial_graph_screening.yaml
```

运行输入通道替换快筛：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\feature_replacement_input_channel_screening.yaml
```

单个替换模块小样本验证：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\real_imag_sfe_replacement_radar_prior_dynamic_sfe.yaml --epochs 1 --max-train-windows 64 --max-test-windows-per-file 5 --no-progress --log-interval 1
```

生成 SDRDSP Fig.9 v2 数据：

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_sdrdsp.py --raw-dir datasets\sdrdsp\raw
```

v2 明确采用 `N=256` 局部裁剪，并通过 manifest 固定 SCR 求和、连续目标注入、目标间距、无归一化和一基目标单元口径。运行完整数据 smoke：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --configs paper_modules\configs\repro_original_stgnn_sdrdsp_strict256_v2.yaml --seeds 42 --epochs 2 --target-pfa 0.001 --name sdrdsp_v2_full_smoke --stop-on-failure
```

运行 IPIX Fig.7 困难点 per-file 对照：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_b.yaml
```

运行前静态校验 per-file 配置：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_b.yaml --validate-only
```

运行 IPIX Fig.7 困难点 range-roll 增广对照：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_shift_aug_b.yaml --validate-only
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_shift_aug_b.yaml
```

运行 IPIX Fig.7 困难点三臂对照：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_three_arm.yaml
```
