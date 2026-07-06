# 仓库瘦身执行报告

## 任务结果总览

| 任务 | commit | 结果 |
|---|---|---|
| T0 | `33ef4e3` | 创建工作区外备份 `E:\marine_target_stgnn_backup_20260706` 并提交既有 tracked 删除。 |
| T0.5 | `9fcb626` | 在 `AGENTS.md` 新增第 0 节，明确保护路径、删除 SOP、git 操作闸和环境边界。 |
| T1 | `e8ff21b` | 清理非保护区 `__pycache__`，并将 `1.md` 整理为 `docs/ipix_related_papers_survey.md`。 |
| T2 | 无新 commit | 按裁决修正为 20 个 `test_scr_*.npz` 后验收通过，删除嵌套重复目录 `data/paper_strict_256/paper_strict_256/`。 |
| T4 | 跳过 | 首次因 `git status --short` 非空跳过；T8 修正后重试时 `git filter-repo` 不可用，安装 `.venv` 依赖的提权请求被拒绝，按规则跳过。 |
| T5 | `e723e53` | 新增 `paper_modules/datasets/` registry，迁移 IPIX 数据集逻辑并保留兼容导出。 |
| T6/T7 | `d7a03b0` | 接入 dataset registry、`pd_scr_curve`、`original_stgnn` wrapper 与 SCR256 复现配置，并完成双数据线 smoke。 |
| T8 | `149c73b` | 删除旧训练/评估入口与旧配置。 |
| T8 修正 | `ec88aba` | 补交 T0 未覆盖的未跟踪研究配置、模型源码和实验记录；三个 quick 配置按裁决直接删除。 |
| T9 | `83a9e56` | 同步 README、AGENTS、paper_modules README 的主线入口、SCR 配置和 smoke 命令。 |

## Git 瘦身状态

T4 未执行 history rewrite，因此没有瘦身前后对比结果。

- T4 重试前 HEAD：`ec88aba92185265088fd0ec28b0e8db736603f9c`
- T4 重试前对象体积：`size-pack: 435.80 MiB`
- 当前对象体积：`size-pack: 435.91 MiB`
- remote 未被 `git filter-repo` 移除，仍为 `origin https://github.com/hhhappb/marine_target_stgnn.git`

## 删除文件总清单

### T1 缓存和文档整理

- 删除非保护区 `__pycache__` 目录。
- 删除空目录 `.agents/`。
- `1.md` 移动为 `docs/ipix_related_papers_survey.md`。

### T2 数据重复目录

- 删除嵌套重复目录 `data/paper_strict_256/paper_strict_256/`。

### T8 旧入口和旧配置

- `train.py`
- `data/dataset.py`
- `run_train.py`
- `train_paper_reproduction_v2.py`
- `evaluate_final.py`
- `train_ipix.py`
- `paper_modules/experiments/run_ablation.py`
- `paper_modules/experiments/evaluate.py`
- `configs/ipix_stgnn.yaml`
- `paper_modules/configs/quick_real_imag_sfe_replacement_radar_prior_dynamic_sfe.yaml`
- `paper_modules/configs/quick_real_imag_sfe_tfe_replacement_radar_prior_dynamic_sfe_diff_bicam.yaml`
- `paper_modules/configs/quick_real_imag_tfe_replacement_diff_bicam.yaml`

## 验证记录

- T5 导入检查通过：`from paper_modules.datasets import build_dataset` 与旧兼容导入输出 `ok`。
- T7 IPIX smoke 通过，并生成 `config.yaml`、`eval_results.json`、`metrics.json`。
- T7 SCR smoke 通过，输出 20 个 SCR 点；`metrics.json` 中 `threshold_source=train`。
- `original_stgnn` 接口契约检查通过：`[2,4,256] complex -> [2,2,256]`。
- `paper_modules/experiments/leakage_probe.py --help` 通过。
- T8 删除后 IPIX/SCR smoke 已复跑通过。
- T9 文档关键词检查通过：`README.md`、`AGENTS.md`、`paper_modules/README.md` 中无旧入口关键词残留。
- T8 修正后 `git status --short` 曾达到空状态；T10 报告写入后仅本报告发生修改。

## 待人工确认文件

- 无无法归类的未跟踪文件。
- 当前 `??` 清单中没有 `*.pth`、`*.npz`、`*.cdf`、`*.log` 或临时文件。

## 遗留人工事项

- T4 未执行。若后续仍需清理 Git 历史，需要人工明确批准 `.venv` 安装 `git-filter-repo` 或提供可用的 `git filter-repo`，再执行 history rewrite。
- 如果未来成功执行 T4，需要人工重设 remote、force push，并通知协作端重新 clone。
- 确认备份目录 `E:\marine_target_stgnn_backup_20260706` 的保留期限。
- 500 epoch 完整复现尚未执行；当前 SCR 结果只是 2 epoch smoke，不能作为论文结论。
- 新 SCR 评估使用训练集杂波阈值口径，和旧测试集阈值历史输出不可直接对比；正式比较必须统一阈值来源。

## 未跟踪文件处置

本节对应 T8 修正裁决：先列出 `git status --short` 的全部 `??` 项，按 AGENTS.md 第 12 节 push 口径分类处理。以下清单不包含受保护训练产物、权重、数据或日志。

### 补交到 Git 的研究配置

- `paper_modules/configs/real_imag_tfe_replacement_diff_bicam.yaml`：研究配置，按文件名单独 `git add`。
- `paper_modules/configs/real_imag_tfe_replacement_diff_bicam_p16.yaml`：研究配置，按文件名单独 `git add`。
- `paper_modules/configs/real_imag_tfe_replacement_diff_bicam_p32.yaml`：研究配置，按文件名单独 `git add`。
- `paper_modules/configs/sfe_tfe_replacement_radar_diffbic.yaml`：研究配置，按文件名单独 `git add`。
- `paper_modules/configs/sfe_tfe_replacement_radar_diffbic_train_only_stats.yaml`：研究配置，按文件名单独 `git add`。

### 补交到 Git 的模块源码

- `paper_modules/models/modules/temporal.py`：模型模块源码，按文件名单独 `git add`。
- `paper_modules/models/modules/temporal_modules/__init__.py`：模型模块源码，按文件名单独 `git add`。
- `paper_modules/models/modules/temporal_modules/diff_bicam_tfe.py`：模型模块源码，按文件名单独 `git add`。
- `paper_modules/models/modules/temporal_modules/stgnn_tfe.py`：模型模块源码，按文件名单独 `git add`。

### 补交到 Git 的实验记录

- `reports/VideoFusion差分双向协同注意力迁移到雷达时间建模说明.md`：实验/文献记录，按文件名单独 `git add`。
- `reports/ipix_cellwise_stats_validation/feature_auc_by_file.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/ipix_cellwise_stats_validation/feature_auc_by_split.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/ipix_cellwise_stats_validation/summary.md`：实验记录 Markdown，按文件名单独 `git add`。
- `reports/ipix_eval_overall.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/ipix_eval_paper_fig7_style_pfa_0.001.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/ipix_eval_pfa_0.001_by_polarization.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/ipix_eval_pfa_0.001_worst_files.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/ipix_experiment_log.md`：实验记录 Markdown，按文件名单独 `git add`。
- `reports/ipix_fig7_current_audit_plan.md`：实验记录 Markdown，按文件名单独 `git add`。
- `reports/radar_paper_prior_stage_review.md`：实验记录 Markdown，按文件名单独 `git add`。
- `reports/radar_prior_preprocessing_plan.md`：实验记录 Markdown，按文件名单独 `git add`。
- `reports/差分增强双向协同注意力时间建模观察时间对比实验报告.md`：实验记录 Markdown，按文件名单独 `git add`。
- `reports/差分增强双向时序协同注意力时间建模文献依据整理.md`：文献记录 Markdown，按文件名单独 `git add`。
- `reports/新空间新时间_Fig7风格_train_clutter_pfa_0.001.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/新空间新时间_与论文STGNN_Fig7对比_train_clutter_pfa_0.001.csv`：实验记录 CSV，按文件名单独 `git add`。
- `reports/新空间新时间模块IPIX论文对比实验报告.md`：实验记录 Markdown，按文件名单独 `git add`。
- `reports/雷达先验动态图空间特征提取模块替换实验报告.md`：实验记录 Markdown，按文件名单独 `git add`。

### T8 明确删除清单中的未跟踪 quick 配置

- `paper_modules/configs/quick_real_imag_sfe_replacement_radar_prior_dynamic_sfe.yaml`：T8 明确删除项，已按裁决授权直接删除，当前不存在。
- `paper_modules/configs/quick_real_imag_sfe_tfe_replacement_radar_prior_dynamic_sfe_diff_bicam.yaml`：T8 明确删除项，已按裁决授权直接删除，当前不存在。
- `paper_modules/configs/quick_real_imag_tfe_replacement_diff_bicam.yaml`：T8 明确删除项，已按裁决授权直接删除，当前不存在。

### 不提交、不删除项

- 当前 `??` 清单中没有 `*.pth`、`*.npz`、`*.cdf`、`*.log` 或临时文件。

### 无法归类项

- 无。
