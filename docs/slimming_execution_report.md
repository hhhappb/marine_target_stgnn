# 仓库瘦身执行报告

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
