# IPIX Fig.7 当前结果可信度审计整理

本文整理当前 IPIX Fig.7 结果、预处理代码、模型代码和需要验证的风险假设。目的不是解释高结果，而是证明高 `PD/accuracy` 是否来自实验设置错误、代码错误、预处理泄漏或真实模型改进。

## 1. 当前最新结果

最新可见报告：

```text
reports/新空间新时间模块IPIX论文对比实验报告.md
```

报告核心结果：

| 项目 | 数值 |
|---|---:|
| 任务 | IPIX 14 数据集 × 4 极化 Fig.7-style |
| 输入 | P=4, N=14 |
| 阈值来源 | train_clutter |
| target Pfa | 0.001 |
| threshold | 0.99826056 |
| PD | 0.999796 |
| actual PF | 0.002243 |
| TP/FN/FP/TN | 2,778,118 / 566 / 16,816 / 7,480,388 |
| 56 个文件-极化平均 PD | 0.999800 |
| 最低 PD | 0.997807 |

当前边界：

- 报告引用的 checkpoint 不在当前工作区中；
- 当前代码与报告配置存在不一致，报告结果需要重新跑可追溯版本；
- 该结果目前只能作为诊断强信号，不能直接作为严格论文结论。

## 2. 原论文 IPIX Fig.7 关键协议

从原论文 Section IV 可抽取的 IPIX Fig.7 关键点：

| 协议项 | 论文要求 |
|---|---|
| 数据 | IPIX Dartmouth 14 个数据集 |
| 极化 | HH / HV / VV / VH |
| 每个数据集长度 | 131072 range profiles |
| 输入样本 | 4 个连续 range profiles |
| Range cells | N=14 |
| 切分 | 每个 dataset/polarization 前 60% train，后 40% test |
| 滑窗 | ST-GNN 使用非重叠 P=4 窗口 |
| loss | Cross entropy |
| optimizer | Adam, lr=0.001 |
| IPIX batch size | 512 |
| 判决分数 | clutter probability `o0` |
| FAR/Pfa 阈值 | 训练集 clutter 样本的 `o0` 升序分位 |
| Fig.7 Pfa | 0.001 |

待澄清的协议点：

- Fig.7 是每个 dataset×polarization 单独训练一个 detector，还是把所有 IPIX train samples pooled 后训练一个 detector；
- 论文中 official auto-processing 是否等价于全文件统计处理，还是官方已处理后的数据直接使用；
- target label 是否包含 PRC+SRC，还是只评价 PRC。

## 3. 当前预处理代码路径

核心文件：

```text
scripts/preprocess_ipix.py
datasets/ipix_dartmouth/labels.json
datasets/ipix_dartmouth/processed/window4_stride4_related/
```

当前流程：

1. 读取原始 `.cdf`；
2. 对每个文件/极化提取 I/Q；
3. 在切分 train/test 之前执行 `auto_process_iq()`：
   - 按 range cell 计算全文件 `mean_i/mean_q/std_i/std_q`；
   - 计算 I/Q phase imbalance；
   - 输出 complex echo；
4. 按 `split_at = int(nsweep * 0.6)` 切分；
5. 分别用 `window=4, stride=4` 做非重叠窗口；
6. 将固定 range label broadcast 成每个窗口的 `y_range`。

已确认的数据事实：

| 检查项 | 结果 |
|---|---|
| train 文件数 | 56 |
| test 文件数 | 56 |
| train windows | 1,100,960 |
| test windows | 733,992 |
| train target/clutter bins | 4,167,920 / 11,245,520 |
| test target/clutter bins | 2,778,684 / 7,497,204 |
| 每个 `.npz` 的 `y_range` | 所有窗口完全相同 |
| train/test exact window overlap | 未发现 exact duplicate window |
| train/test 预处理统计 | `mean/std/inbalance` 完全相同 |

预处理风险：

- 全文件统计先于 60/40 切分，可能引入测试段分布统计；
- 固定 `range_labels` 在同一文件所有窗口中重复，模型可能学到文件/极化/距离单元稳定指纹；
- 当前 `target_policy=related` 使用 PRC+SRC，需确认与论文评价口径完全一致。

## 4. 当前训练与评估代码路径

核心文件：

```text
train_ipix.py
paper_modules/experiments/train.py
paper_modules/experiments/auto_experiment.py
```

当前训练组织：

- `list_split_files(data_dir, "train", pols)` 会收集所有极化、所有 source 的 train `.npz`；
- `IpixWindowDataset(train_files)` 将这些文件拼成一个训练集；
- 因此当前主线是 pooled training，而不是明确的 per dataset×polarization training。

当前评估逻辑风险：

- `paper_modules/experiments/train.py::evaluate_files()` 只接收 eval/test files；
- 它从这些 eval/test files 的 clutter scores 中计算 threshold；
- 这更接近 `test_diagnostic_current_eval`，不是论文要求的 train clutter threshold；
- 最新报告中虽然有 `train_clutter` 表，但当前代码路径不能直接复现该表。

## 5. 当前模型代码路径

Baseline 路径：

```text
models/st_gnn.py
```

论文模块替换路径：

```text
paper_modules/models/sfe_replacement_stgnn.py
paper_modules/models/modules/spatial_graphs/radar_prior_dynamic_sfe.py
paper_modules/models/modules/temporal_modules/diff_bicam_tfe.py
paper_modules/configs/sfe_tfe_replacement_radar_diffbic.yaml
```

当前模型侧风险：

- 最新报告配置使用 `temporal.type=diff_bicam_tfe`；
- 当前 `paper_modules/models/sfe_replacement_stgnn.py` 中存在只允许 `temporal.type=stgnn_tfe` 的检查；
- 因此最新报告配置与当前模型入口不一致，需要先恢复可构建、可复现状态；
- `models/st_gnn.py` baseline 应保持冻结，不能把新想法直接加入 baseline。

## 6. 需要证明或排除的风险假设

### H1：结果高是因为 pooled training，而不是模型真实泛化

风险等级：High

现象：

- 当前模型使用 56 个 source/pol 的 train windows 一起训练；
- test 中的 56 个 source/pol 与 train 完全重合，只是时间段不同；
- 如果模型学到 source/pol 的稳定特征和固定目标 range cells，PD 可能异常高。

验证：

1. 跑 per dataset×polarization 独立训练评估；
2. 跑 leave-one-file-out；
3. 对比 pooled vs per-file 的 PD/PF。

判定：

- 如果 per-file 仍高，pooled 不是主要原因；
- 如果 leave-one-file-out 明显下降，说明存在文件指纹依赖。

### H2：结果高是因为全文件 auto-processing 泄漏测试段统计

风险等级：High

现象：

- 当前 `auto_process_iq()` 在 60/40 split 前执行；
- train/test `.npz` 保存的 `mean/std/inbalance` 完全相同；
- 这使用了测试段分布信息。

验证：

1. 生成 `train_only_auto_stats` 数据；
2. 用 train 前 60% 拟合 I/Q 统计；
3. 同一统计应用到 train/test；
4. 对比当前 full-file auto 数据的 PD/PF。

判定：

- 如果差异很小，则预处理泄漏不是主要原因；
- 如果 PD 明显下降，则当前结果需降级为诊断。

### H3：结果高是因为阈值使用了测试集 clutter

风险等级：Medium-High

现象：

- 当前代码默认 eval/test clutter 定阈值；
- 报告中 `test_diagnostic` 的 actual PF 更贴近 target Pfa；
- 报告中 `train_clutter` actual PF=0.002243，高于 0.001。

验证：

1. 在代码中显式实现 `threshold_source=train_clutter`；
2. 重新输出 threshold、actual PF、TP/FN/FP/TN；
3. 保存 threshold clutter bins 数。

判定：

- 如果 train_clutter 复现报告数值，则阈值不是异常高的唯一来源；
- 如果无法复现，则报告结果不可用。

### H4：结果高是因为标签口径过宽

风险等级：Medium

现象：

- 当前 `target_policy=related` 标记 PRC+SRC；
- 每个文件通常有 3-5 个 target cells，target ratio 约 27%；
- 如果论文只评价 PRC，则当前任务更容易。

验证：

1. 生成 `target_policy=primary` 数据；
2. 跑同一训练和评估；
3. 同时报告 primary-only 与 related 的结果。

判定：

- 如果 primary-only 大幅下降，需要在论文表述中明确 related 口径；
- 如果变化不大，标签口径不是主要原因。

### H5：结果高是因为 optimizer/loss 不同

风险等级：Medium-Low

现象：

- 论文使用 Adam + CE；
- 当前使用 AdamW + weight decay + class weights；
- 这会影响训练分数校准，但通常不足以单独解释 PD 接近 1。

验证：

1. 跑 Adam + unweighted CE；
2. 跑 AdamW + weighted CE；
3. 比较 PD/PF 和 score distribution。

判定：

- 若差异小，则不是主要原因；
- 若差异大，正式复现应回到论文 optimizer/loss。

### H6：最新报告不是当前代码产物

风险等级：Critical

现象：

- 报告 checkpoint 当前不存在；
- 配置中的 `diff_bicam_tfe` 与当前模型入口不兼容；
- 无完整 run 目录。

验证：

1. 恢复或重新运行同名配置；
2. 确保模型能构建；
3. 保存完整 `logs/training/<run_id>/`；
4. 用同一 checkpoint 重新生成报告表。

判定：

- 不能复现前，最新结果不能作为正式结论。

## 7. 最小验证顺序

建议按以下顺序执行，避免一次改太多变量：

1. **可复现性修复**
   - 让 `sfe_tfe_replacement_radar_diffbic.yaml` 可构建；
   - 重新跑当前数据、当前配置；
   - 保存完整 run 产物。

2. **train_clutter 阈值复核**
   - 不改预处理；
   - 不改模型；
   - 只修评估阈值来源；
   - 验证是否复现报告的 `PD=0.999796, actual PF=0.002243`。

3. **per dataset×polarization 复核**
   - 不改预处理；
   - 不改模型；
   - 改训练组织；
   - 判断 pooled training 是否导致偏高。

4. **train-only auto-processing 对照**
   - 只改预处理统计边界；
   - 跑 pooled 与 per-file 两种最小对照；
   - 判断全文件统计是否导致偏高。

5. **negative controls**
   - primary-only；
   - target-cell shift；
   - leave-one-file-out。

## 8. 当前可写入论文/报告的谨慎表述

当前结果可以表述为：

> 在当前 pooled IPIX Fig.7-style 设置、related target label、当前 auto-processing 和报告中的 train_clutter 阈值口径下，新空间+新时间模型取得极高 PD。但由于报告 checkpoint 与当前代码不可追溯、训练组织和阈值实现仍需严格复核，该结果目前应视为诊断结果，不能直接作为严格复现并超过论文 ST-GNN 的正式结论。

暂时不建议表述为：

> 严格按照论文 Fig.7 设置超过 ST-GNN。

## 9. 最终判断

当前最需要证明的不是模型是否强，而是：

1. 高结果是否可以由当前代码、配置、checkpoint 完整复现；
2. 高结果是否仍能在 per dataset×polarization 协议下保持；
3. 高结果是否不依赖全文件 auto-processing 统计；
4. 高结果是否不依赖固定 range label 或 source/pol 指纹。

只有这些验证通过后，才能把当前结果从“诊断强信号”升级为“可写入论文的正式性能结论”。
