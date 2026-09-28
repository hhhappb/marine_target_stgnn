# IPIX 当前实验协议

核对日期为2026年9月28日。本文件记录最近本地P=8 Original/E4-selected完整配对实验，不与原论文Fig.7复现或P=16 PAX-L1混称。见 [协议索引](experiment_protocols.md)。

## 数据与标签

- Dartmouth IPIX 14文件×HH/HV/VV/VH四极化，共56条件。每条件独立训练两个模型，共112次训练。
- 复数输入 `[B,8,14]`，步长4，窗口重叠4脉冲。
- 官方ipixload auto处理，预处理保留double精度。Original在模型输入边界转换为complex64。预处理统计基于完整记录，不应描述为仅训练段拟合。
- 从已有P=8、步长4、50/50存储分段重建131072个连续脉冲，以 `int(131072*0.6)` 为界重新划分前60%训练、后40%测试，两段分别分窗，无窗口跨界。存储manifest的50/50不是本次有效划分。
- 每条件训练19659窗、测试13106窗。
- 训练与评价均使用 `primary_strict`。仅 `range_roles == 2` 为目标，SRC与其他单元为杂波。目录名中的 `related` 不是本次有效标签。
- 使用真实目标回波，不按SCR注入合成目标，不使用SDRDSP的SCR公式。

## 训练与模型

- seed42，70 epochs，batch64，每批直接更新，无梯度累积，保留末尾不足64的批次。
- Adam，学习率0.001，普通无权重交叉熵，选择最后第70轮模型。
- 每轮对复回波执行非循环 `clutter_fill` 距离平移，最大3格，包含零位移。合法位移保证目标相关区域不越界，腾空单元由真实杂波填充并将标签清零。测试无增强。
- Original使用原始ST-GNN，P=8下两级TFE后剩余时间位置取均值。
- E4-selected使用E4特征与 `gdcm_a0` 检测头，E4预处理器仅在训练回波上拟合。

| 槽位 | Original | E4-selected |
|---|---|---|
| SFE1 | 原始STFE | radar_prior_dynamic_sfe |
| TFE1 | stgnn_tfe | pulse_attention_only_tfe |
| SFE2 | 原始STFE | radar_prior_dynamic_sfe |
| TFE2 | stgnn_tfe | stgnn_tfe |

E4空间参数为 `static_gamma=0.5`、`static_delta=5`、`static_weight=0.7`、`dynamic_temperature=0.2`、`dropout=0.1`。TFE1参数为 `attention_dim=64`、`num_heads=4`、`residual_scale=0.1`、`use_attention=true`。比较改变了输入特征和检测头等位置，只能归因于整体方案。

## 评价

两模型使用相同规则，各自从训练输出计算温度和阈值。

```text
T = clip(mean(abs(train_clutter_logits)), 2, 16)
o0 = softmax(logits / T)[:, 0, :]
Nc = 19659 * 13 = 255567
threshold = sort(train_clutter_o0)[ceil(0.001 * Nc) - 1]
o0 <= threshold -> target
```

校准使用训练标签为0的单元，包括SRC。温度缩放是项目扩展，不能称为未修改的论文评分实现。主指标为名义Pfa=0.001下逐条件测试PRC的PD及56条件等权均值。测试实际PF不参与排序，不使用测试杂波重定阈值。单seed不代表跨seed稳定性。

## 批次边界

| 批次 | 输入 | 预算与模型选择 |
|---|---|---|
| 当前本地完整配对 | P=8 | 70轮，batch64，最后一轮 |
| 外部 run_se_v21_full56(1).log | P=8 | 60轮，日志写batch64×8，模型选择未披露 |
| 本地2026年8月17日Fig.7基线 | P=4 | 60轮，batch512，最低训练损失 |
| 外部Fig.7复现包 | P=4 | 100轮，batch512，最后一轮 |
| PAX-L1原协议 | P=16 | 以各次运行快照为准 |

[PAX-L1协议](ipix_pax_l1_protocol.md)限定P=16，本次P=8不属于该定义。

## 本地证据

- `logs/training/20260926_ipix_p8_original_vs_e4_selected_a1_full56_seed42_split60/` 内的 `config.json`、`process.json`、`source_manifest.json`、`stdout.log`、`result.json`、`progress.json`。完成记录为112/112。
- 运行工作树 `E:/ip8-e4`，基准提交 `10e2906`，入口 `e4d2_gdcm/experiments/run_p8_selected_comparison.py`。
- 数据目录 `datasets/ipix_dartmouth/processed/window8_stride4_related_official_ipixload_auto_double`。

这些本地路径用于审计，不表示GitHub包含完整E4运行代码、数据、权重或日志。
