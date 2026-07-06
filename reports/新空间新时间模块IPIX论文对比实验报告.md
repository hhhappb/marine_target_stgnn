# 新空间新时间模块 IPIX 论文对比实验报告

> **紧急更正（2026-07-06）**：本报告原先把 pooled 训练结果与论文 Fig.7 数字化曲线直接对比，并把差值解释为新空间/新时间模块优化，这是不成立的。论文 Fig.7 应按每个 `dataset × polarization` 训练独立 detector；本报告结果来自一个 pooled detector，训练组织不一致。因此下文所有“相同设置下超过论文”“提升来自模块优化”的表述降级为历史诊断记录，不能作为正式论文结论。后续必须使用 per-file 独立 detector 的 baseline vs 新模块单变量对照重新归因。

## 实验目的

本报告整理新空间模块 + 新时间模块在 IPIX Fig.7-style 实验中的结果，并与参考论文 `Marine Target Detection via Spatial-Temporal Graph Neural Network` 的 Fig.7 中 ST-GNN detector 曲线进行对比。

需要特别说明：参考论文 Fig.7 没有提供逐点原始表格数值，因此本文中的 `Paper ST-GNN detector` 曲线来自论文图片的近似数字化结果，只用于趋势和量级对比，不等价于作者公开的精确数据。

## 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | IPIX Dartmouth 14 个数据集 |
| 极化 | HH / HV / VV / VH |
| 输入窗口 | P=4，N=14 |
| 滑窗 | window length=4，sliding step=4 |
| 训练/测试划分 | 每个极化前 60% range profiles 训练，后 40% range profiles 测试 |
| 模型 | New SFE+TFE detector |
| 空间模块 | radar_prior_dynamic_sfe |
| 时间模块 | diff_bicam_tfe |
| checkpoint | `logs/training/ipix_fig7_full_sfe_tfe_radar_diffbic/runs/sfe_tfe_replacement_radar_diffbic_seed42/best_model.pth` |
| 配置 | `paper_modules/configs/sfe_tfe_replacement_radar_diffbic.yaml` |
| 主要对比指标 | PD @ target Pfa=0.001 |
| 当前正式阈值来源 | train_clutter |

## Fig.7 风格对比图

![新空间新时间与论文 ST-GNN Fig7 对比](figures/新空间新时间_与论文STGNN_Fig7对比_train_clutter_pfa_0.001.png)

该图中红色虚线为论文 Fig.7 中 ST-GNN detector 的近似数字化曲线，蓝色实线为本实验的新空间 + 新时间 pooled detector 结果。由于训练组织不同，该图只能说明 pooled 设置下分数很高，不能说明模型模块在论文协议下优于 ST-GNN。

## 本模型 Fig.7 风格图

![新空间新时间 Fig7 风格图](figures/新空间新时间_Fig7风格_train_clutter_pfa_0.001.png)

从完整坐标轴看，新模型在 14 个 IPIX 数据集、四个极化上的 PD 基本贴近 1.0。由于整体检测概率非常高，完整坐标轴下不同数据集之间的细微差异不明显。

## 本模型放大图

![新空间新时间 Fig7 风格放大图](figures/新空间新时间_Fig7风格_train_clutter_pfa_0.001_放大.png)

放大图显示，当前模型最弱点仍然出现在 `VV` 极化的个别数据集，但最低 PD 仍为 `0.997807`，说明检测稳定性明显增强。

## 与论文 ST-GNN 的指标对比

基于论文 Fig.7 的近似数字化曲线，统计得到：

| 指标 | Paper ST-GNN digitized approx. | New SFE+TFE | 增长 |
|---|---:|---:|---:|
| 56 个文件/极化组合平均 PD | 0.942821 | 0.999800 | +0.056979 |
| 最低 PD | 0.649000 | 0.997807 | +0.348807 |
| HH 平均 PD | 0.947286 | 0.999733 | +0.052447 |
| HV 平均 PD | 0.964500 | 0.999940 | +0.035440 |
| VV 平均 PD | 0.903286 | 0.999575 | +0.096289 |
| VH 平均 PD | 0.956214 | 0.999954 | +0.043740 |

提升最大的点集中在论文 ST-GNN 曲线下探明显的位置：

| Label | 数据源 | 极化 | Paper ST-GNN PD | New SFE+TFE PD | 增长 | New PF |
|---:|---|---|---:|---:|---:|---:|
| 12 | 19931118_162155_stareC0000 | VV | 0.649 | 0.999981 | +0.350981 | 0.000504 |
| 6 | 19931109_191449_starea | HH | 0.692 | 0.999059 | +0.307059 | 0.012561 |
| 1 | 19931107_135603_starea | VV | 0.746 | 0.997807 | +0.251807 | 0.003265 |
| 6 | 19931109_191449_starea | VV | 0.769 | 0.999568 | +0.230568 | 0.015953 |
| 3 | 19931107_145028_starea | VV | 0.776 | 0.999975 | +0.223975 | 0.000201 |
| 6 | 19931109_191449_starea | HV | 0.782 | 0.999873 | +0.217873 | 0.005805 |
| 6 | 19931109_191449_starea | VH | 0.791 | 0.999924 | +0.208924 | 0.006138 |
| 12 | 19931118_162155_stareC0000 | HH | 0.865 | 1.000000 | +0.135000 | 0.000038 |

整体来看，`VV` 极化的数字差值最大。但该差值混入 pooled 训练、固定标签位置、预处理统计边界、标签口径等因素，不能归因为新模块补强了原 ST-GNN。

## 两种阈值分割方式对比

ST-GNN 的最终判决基于杂波概率 `o0`：

```text
o0 <= threshold -> target
o0 >  threshold -> clutter
```

`threshold` 越大，模型越容易判为 target，因此 PD 通常会上升，但 PF 也更容易上升。

### train_clutter 阈值

`train_clutter` 是论文 FAR controller 更接近的口径：先用训练集 clutter cells 的 `o0` 排序，根据目标 Pfa 取分位数阈值，再把这个固定阈值用于测试集。

```text
threshold_source = train_clutter
threshold clutter bins = 11,245,520
eval clutter bins = 7,497,204
```

| Target Pfa | Threshold | PD | Actual PF | TP | FN | FP | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.62174684 | 0.999245 | 0.000679 | 2,776,587 | 2,097 | 5,091 | 7,492,113 |
| 0.001 | 0.99826056 | 0.999796 | 0.002243 | 2,778,118 | 566 | 16,816 | 7,480,388 |
| 0.01 | 0.99999988 | 0.999963 | 0.013043 | 2,778,582 | 102 | 97,786 | 7,399,418 |

### test_diagnostic 阈值

`test_diagnostic_current_eval` 是诊断口径：用测试集 clutter cells 的 `o0` 直接定阈值，再在同一测试集上计算 PD/PF。因此 actual PF 会非常接近目标 Pfa，适合看模型分数可分性，但不适合作为最终正式结论。

| Target Pfa | Threshold | PD | Actual PF | TP | FN | FP | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.00143427 | 0.995645 | 0.000100 | 2,766,583 | 12,101 | 750 | 7,496,454 |
| 0.001 | 0.93300003 | 0.999506 | 0.001000 | 2,777,310 | 1,374 | 7,498 | 7,489,706 |
| 0.01 | 0.99999940 | 0.999954 | 0.010120 | 2,778,557 | 127 | 75,875 | 7,421,329 |

### 阈值结果解读

在 `target Pfa=0.001` 下：

| 阈值来源 | Threshold | PD | Actual PF | 说明 |
|---|---:|---:|---:|---|
| train_clutter | 0.99826056 | 0.999796 | 0.002243 | 更接近论文 FAR controller；PD 极高，但测试集实际 PF 高于目标 |
| test_diagnostic | 0.93300003 | 0.999506 | 0.001000 | 测试集诊断阈值；actual PF 几乎贴合目标 |

两者差异说明：训练集与测试集 clutter 的 `o0` 分布存在一定偏移。train_clutter 阈值更高，使模型在测试集上更容易判为 target，因此 PD 略高，但 actual PF 也升高到 `0.002243`。报告时应优先使用 train_clutter 口径，并同时报告 actual PF。

## 原“为什么会取得明显提升”解释已撤回

以下机制解释只可作为候选假设，不能由本报告的 pooled 对比数据支持。要证明这些机制有效，必须在 per `dataset × polarization` 独立 detector 协议下比较冻结 baseline 与新模块，且保持数据、预处理、标签、阈值和 seed 一致。

1. 空间模块引入雷达先验动态图  
   原 ST-GNN 的空间建模更依赖学习到的图注意力关系；新空间模块引入静态距离先验、局部动态相似边和少量 top-k 非局部补边，使 range cell 之间的信息传播更符合海杂波小目标的空间延展特性。对于 `VV`、`HH` 中容易受海杂波和海尖峰干扰的数据集，雷达先验图能减少无约束动态图被杂波相似性误导的问题。

2. 时间模块引入差分增强与双向时序协同注意力  
   原 TFE 主要是局部卷积门控，能够压缩短时间上下文，但对前后脉冲间的目标扰动建模较弱。Diff-BiCAM TFE 显式使用前后脉冲差分、forward/backward temporal context 和 bi-temporal co-attention，更容易捕获小目标破坏海杂波慢时间连续性的模式。

3. 空间与时间模块形成互补  
   空间动态图负责约束 range cell 之间的传播范围，时间差分协同注意力负责提取慢时间扰动。二者配合后，模型对困难 label 的漏检显著减少，尤其是论文 ST-GNN 曲线低谷处提升最明显。

4. 困难极化提升最大  
   `VV` 极化的平均 PD 从近似 `0.903286` 提升到 `0.999575`，是四个极化中提升最大的部分。这说明新模块主要提升了模型在非平稳海杂波、低可分性极化和局部异常扰动场景下的鲁棒性。

## 结论与边界

在当前 pooled IPIX Fig.7-style 诊断设置下，新空间 + 新时间模型表现出接近 1.0 的检测概率：

```text
Paper ST-GNN digitized mean PD ≈ 0.942821
New SFE+TFE mean PD ≈ 0.999800
平均提升 ≈ +0.056979
```

这些数字不能和论文 Fig.7 直接做优化归因。最主要的下一步是按 per-file 独立 detector 协议复跑 hard points，并与冻结 baseline 做单变量对照。

同时需要保留两个结论边界：

1. 训练组织与论文 Fig.7 不一致：本报告是 pooled detector，论文协议应为 per `dataset × polarization` 独立 detector。
2. 论文 ST-GNN 数值来自 Fig.7 图片近似数字化，不是论文作者提供的精确原始表格。
3. train_clutter 口径下 `target Pfa=0.001` 对应的测试集实际 `PF=0.002243`，高于目标虚警率。正式论文表述应同时报告 `target Pfa`、`actual PF`、`threshold_source` 和阈值。

因此，本报告推荐表述修正为：

> 在 pooled IPIX Fig.7-style 诊断设置、related label、train_clutter 阈值口径下，新空间 + 新时间模型取得 `PD=0.999796`、`PF=0.002243`。由于训练组织与论文 Fig.7 的 per dataset×polarization 独立 detector 协议不一致，该结果不能作为严格超过论文 ST-GNN 的结论，也不能把与论文数字化曲线的差值归因为模块优化。
