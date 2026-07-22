# TFE-DIFFBICAM-RETIRE-001

## 目标与边界

退役旧 `DiffBiCAMTFE`，将 `CorrectedDiffOnlyTFE` 机械迁移为 `DiffTFE`。
本任务不改变前向计算，不加入尺度归一化、trend、curvature 或其他新算法。
`logs/training/`、`reports/`、`share/` 仅作历史证据保留，不修改、不删除。

## Git 跟踪文件删除清单

### D1：失效 suite

- `paper_modules/configs/suites/per_file_fig7_hardpoints_three_arm.yaml`
- `paper_modules/configs/suites/per_file_fig7_hardpoints_b.yaml`
- `paper_modules/configs/suites/per_file_fig7_hardpoints_shift_aug_b.yaml`

### D2：旧 DiffBiCAM 配置

- `paper_modules/configs/real_imag_tfe_replacement_diff_bicam.yaml`
- `paper_modules/configs/real_imag_tfe_replacement_diff_bicam_p16.yaml`
- `paper_modules/configs/real_imag_tfe_replacement_diff_bicam_p32.yaml`
- `paper_modules/configs/sfe_tfe_replacement_radar_diffbic.yaml`
- `paper_modules/configs/sfe_tfe_replacement_radar_diffbic_train_only_stats.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints/sfe_tfe_radar_diffbic_label01_vv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints/sfe_tfe_radar_diffbic_label03_vv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints/sfe_tfe_radar_diffbic_label06_hh.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints/sfe_tfe_radar_diffbic_label06_hv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints/sfe_tfe_radar_diffbic_label06_vh.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints/sfe_tfe_radar_diffbic_label06_vv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints/sfe_tfe_radar_diffbic_label12_vv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints_shift_aug/sfe_tfe_radar_diffbic_label01_vv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints_shift_aug/sfe_tfe_radar_diffbic_label03_vv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints_shift_aug/sfe_tfe_radar_diffbic_label06_hh.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints_shift_aug/sfe_tfe_radar_diffbic_label06_hv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints_shift_aug/sfe_tfe_radar_diffbic_label06_vh.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints_shift_aug/sfe_tfe_radar_diffbic_label06_vv.yaml`
- `paper_modules/configs/per_file_fig7_hardpoints_shift_aug/sfe_tfe_radar_diffbic_label12_vv.yaml`

### D3：旧模块源码

- `paper_modules/models/modules/temporal_modules/diff_bicam_tfe.py`

## 明确授权的未跟踪文件删除清单

- `paper_modules/configs/per_file_fig7_fullstats/sfe_tfe_radar_diffbic_label01_vv.yaml`
- `paper_modules/configs/per_file_fig7_fullstats/sfe_tfe_radar_diffbic_label03_vv.yaml`
- `paper_modules/configs/per_file_fig7_fullstats/sfe_tfe_radar_diffbic_label06_hh.yaml`
- `paper_modules/configs/per_file_fig7_fullstats/sfe_tfe_radar_diffbic_label06_hv.yaml`
- `paper_modules/configs/per_file_fig7_fullstats/sfe_tfe_radar_diffbic_label06_vh.yaml`
- `paper_modules/configs/per_file_fig7_fullstats/sfe_tfe_radar_diffbic_label06_vv.yaml`
- `paper_modules/configs/per_file_fig7_fullstats/sfe_tfe_radar_diffbic_label12_vv.yaml`
- `paper_modules/configs/suites/per_file_fig7_fullstats_b.yaml`
- `work/ipix_hardpoint_five_arm/spatial_temporal.yaml`
- `work/ipix_hardpoint_five_arm/temporal_only.yaml`
- `work/n_scale_sdrdsp/n128_spatial_temporal.yaml`
- `work/n_scale_sdrdsp/n128_temporal_only.yaml`
- `work/n_scale_sdrdsp/n512_spatial_temporal.yaml`
- `work/n_scale_sdrdsp/n512_temporal_only.yaml`
- `work/stage1_screen_sdrdsp_p16/spatial_temporal.yaml`
- `work/stage1_screen_sdrdsp_p16/temporal_only.yaml`
- `work/stage1_screen_sdrdsp/spatial_temporal.yaml`
- `work/stage1_screen_sdrdsp/temporal_only.yaml`
- `work/stage1_smoke_sdrdsp/spatial_temporal.yaml`
- `work/stage1_smoke_sdrdsp/temporal_only.yaml`

## 机械迁移清单

- `paper_modules/models/modules/temporal_modules/corrected_diff_tfe.py`
  重命名为 `paper_modules/models/modules/temporal_modules/diff_tfe.py`。
- 类名 `CorrectedDiffOnlyTFE` 重命名为 `DiffTFE`，前向计算保持不变。
- registry 类型 `corrected_diff_only_tfe` 重命名为 `diff_tfe`。
- 删除 registry 类型 `diff_bicam_tfe`，不提供兼容别名。
- `paper_modules/configs/sdrdsp_corrected_diff_only_n256_seed42.yaml`
  重命名为 `paper_modules/configs/sdrdsp_diff_tfe_p4_n256_seed42.yaml`。
- 更新 `tests/test_corrected_temporal_modules.py` 的类名和 registry 断言。
- 从 `README.md`、`paper_modules/README.md` 删除已失效 suite 的命令和入口说明。

## 删除与验证顺序

1. 完成机械迁移并验证新旧前向计算一致。
2. 逐文件检查 suite 引用为 0，使用 `git rm` 删除 D1，并独立提交。
3. 逐文件检查旧配置引用为 0，使用 `git rm` 删除 D2，并独立提交。
4. 删除明确授权的未跟踪文件，并记录路径与不可通过 Git 恢复的事实。
5. 除本计划和受保护历史资产外，检查旧模块文件名/导入路径引用为 0。
6. 使用 `git rm` 删除 D3，并独立提交。
7. 运行时间模块单测、整模型 `[B,4,14] complex -> [B,2,14]` 接口检查和 P=4 smoke。

## 执行记录

- 机械迁移提交：`721eaeb refactor(tfe): rename corrected diff module`。
- 时间模块单测：`6 passed`。
- 整模型接口：`complex [2,4,14] -> float [2,2,14]`。
- 已删除以下 6 个明确授权、未被 Git 跟踪且无调用方的配置；它们无法通过 Git 恢复：
  - `work/stage1_screen_sdrdsp_p16/spatial_temporal.yaml`
  - `work/stage1_screen_sdrdsp_p16/temporal_only.yaml`
  - `work/stage1_screen_sdrdsp/spatial_temporal.yaml`
  - `work/stage1_screen_sdrdsp/temporal_only.yaml`
  - `work/stage1_smoke_sdrdsp/spatial_temporal.yaml`
  - `work/stage1_smoke_sdrdsp/temporal_only.yaml`
- 已从 `work/ipix_hardpoint_five_arm/suite.yaml` 移除两个退役时间分支，suite 名称同步改为 three-arm；随后删除以下 2 个已归零引用的未跟踪配置：
  - `work/ipix_hardpoint_five_arm/spatial_temporal.yaml`
  - `work/ipix_hardpoint_five_arm/temporal_only.yaml`
- 其余 12 个未跟踪删除目标及全部 tracked 删除目标因 `reports/` 或 `share/` 中仍存在逐路径历史引用，按删除 SOP 暂停。
