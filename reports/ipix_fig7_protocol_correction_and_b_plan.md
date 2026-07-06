# IPIX Fig.7 协议更正与 B 档归因实验计划

## 结论更正

当前 pooled IPIX 结果不能用于证明“新模块优化了模型”。论文 Fig.7 的关键训练组织是每个 `dataset × polarization` 单独训练 detector；当前历史报告使用一个 pooled detector 汇总 56 个 train 段训练，和论文协议不是同一个实验。

因此：

- `reports/新空间新时间模块IPIX论文对比实验报告.md` 中的 pooled 高 PD 只能标记为 `pooled_diagnostic`；
- 与论文 Fig.7 数字化曲线的差值不能归因为 `radar_prior_dynamic_sfe` 或 `diff_bicam_tfe`；
- 正式归因必须比较同一 per-file 协议下的冻结 baseline 与新模块。

## B 档最小实验问题

在论文 ST-GNN 掉分最明显的 hard points 上，分别训练：

1. `original_stgnn`：冻结 `models/st_gnn.py` 的薄包装 baseline；
2. `sfe_tfe_radar_diffbic`：新空间 + 新时间模块。

每个 run 只使用一个 `source × polarization` 的 train split，评估同一个 `source × polarization` 的 test split。两模型共享同一数据、预处理、标签、阈值来源、seed、batch size 和训练入口，唯一变量是模型结构。

当前 hardpoint 配置已统一指向 `window4_stride4_related_stats_train_only`，避免全文件 auto-processing 统计混入测试段分布。该目录需要先在实验机上由 `scripts/preprocess_ipix.py --stats-scope train_only` 生成。

若需要拆分“系统级提升”和“SFE/TFE 模块贡献”，使用三臂 suite：

1. `original_stgnn`：原始 ST-GNN 整模型；
2. `sfe_replacement_original_modules`：同一替换骨架，但 SFE/TFE 取 `original_stfe + stgnn_tfe`；
3. `sfe_tfe_radar_diffbic`：新空间 + 新时间模块。

## Hard Points

| Label | Source | Pol | 论文 Fig.7 ST-GNN 数字化 PD | 历史 pooled PD | 风险说明 |
|---:|---|---|---:|---:|---|
| 1 | `19931107_135603_starea` | `vv` | 0.746 | 0.997807 | 论文 VV 明显掉分 |
| 3 | `19931107_145028_starea` | `vv` | 0.776 | 0.999975 | 论文 VV 明显掉分 |
| 6 | `19931109_191449_starea` | `hh` | 0.692 | 0.999059 | 最大 HH 差值 |
| 6 | `19931109_191449_starea` | `hv` | 0.782 | 0.999873 | 同一困难文件 |
| 6 | `19931109_191449_starea` | `vv` | 0.769 | 0.999568 | 同一困难文件 |
| 6 | `19931109_191449_starea` | `vh` | 0.791 | 0.999924 | 同一困难文件 |
| 12 | `19931118_162155_stareC0000` | `vv` | 0.649 | 0.999981 | 最大 VV 差值 |

## 已落地配置

所有配置位于：

```text
paper_modules/configs/per_file_fig7_hardpoints/
```

自动化 suite：

```text
paper_modules/configs/suites/per_file_fig7_hardpoints_b.yaml
paper_modules/configs/suites/per_file_fig7_hardpoints_three_arm.yaml
```

执行前静态校验：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_b.yaml --validate-only
```

执行命令：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_b.yaml
```

快速 smoke 可临时覆盖：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_b.yaml --epochs 1 --max-train-windows 128 --max-test-windows-per-file 8 --stop-on-failure
```

## 判定表

| 判据 | 结论 |
|---|---|
| 新模块 per-file 明显高于 `original_stgnn` per-file，且高于论文数字化值 | 可以初步支持模块有效，但仍需扩展到 56 点 |
| 新模块和 `original_stgnn` per-file 都接近论文低谷 | 历史 pooled 高分主要是协议红利 |
| 两者都远高于论文数字化值 | 还需继续排查标签口径、预处理统计、optimizer/loss 差异 |
| 新模块只在 pooled 高、per-file 不高 | 不得声称模块提升检测能力 |

## 剩余边界

B 档仍不是完整严格复现。它解决 pooled 训练归因问题，但仍需继续确认：

- `target_policy=related` 是否完全等价论文标签口径；
- 训练代码使用的 optimizer、class weights、weight decay 是否需要回到论文原始设置；
- 论文 Fig.7 数字化值不是作者提供的精确原表。
- 当前 suite 默认 seed=42；正式结论需扩展到至少 3 个 seed。
