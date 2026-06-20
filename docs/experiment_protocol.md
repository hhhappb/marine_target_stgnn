# Experiment Protocol

本文档固定当前项目的实验口径，避免后续结果因为阈值、标签、数据切分或命名不同而不可比较。

## 1. 主线问题

当前研究目标不是继续寻找最优数据预处理，而是回答：

1. 雷达先验如何进入空间图建模；
2. 雷达先验如何进入时间建模；
3. 雷达先验如何作为模块输入参与融合；
4. Pfa/FAR 约束如何进入评估和训练目标。

## 2. 默认 baseline

默认 baseline 是当前 IPIX 预处理后的 Raw I/Q + ST-GNN：

```text
data_dir = datasets/ipix_dartmouth/processed/window4_stride4_related
input = complex E [B, P=4, N=14]
feature = [I, Q]
label = y_range
target_policy = related
polarizations = hh, hv, vv, vh
target_pfa = 0.001
```

该 baseline 对应原论文“直接使用 range profiles，不额外做 TF/RD 预处理”的主张。后续所有先验模块都必须和这个 baseline 比较。

## 3. 实验命名

推荐用以下组别命名配置和结果目录：

```text
baseline/                 Raw I/Q + original ST-GNN-style modules
radar_feature/            PL/RPH, amplitude/phase, Doppler, phase diff
statistical_prior/        cell-wise stats, phase stability, local stats
spatial_prior/            local graph, CFAR/reference-window graph, similarity graph
temporal_prior/           ConvGRU, multiscale TCN, range migration temporal module
pfa_aware/                Pfa-aware loss or decision constraints
ablation/                 controlled removals or negative controls
```

每个实验配置必须能说明：

- 相比 baseline 改了哪个模块；
- 是否引入额外雷达先验；
- 参数量或输入通道是否明显变化；
- 结果应与哪个 baseline 或消融项比较。

## 4. 阈值协议

ST-GNN 判决使用杂波概率 `o0`：

```text
o0 <= threshold -> target
o0 >  threshold -> clutter
```

正式结果的阈值来源优先级：

1. 训练集杂波样本；
2. 独立校准集杂波样本；
3. 仅用于诊断的测试集杂波样本。

第三种不能写成正式性能结论。任何报告都必须写明：

```text
threshold_source = train | calibration | test_diagnostic
target_pfa = ...
actual_pf = ...
num_clutter_bins_for_threshold = ...
```

## 5. 报告最低要求

每个正式实验至少报告：

- 训练日志目录 `logs/training/<run_id>/`；
- 配置文件路径；
- checkpoint 路径或 commit/运行标识；
- 数据目录、极化列表、训练/测试文件数；
- seed、epoch、batch size、learning rate；
- `PD`、实际 `PF`、threshold、TP/FN/FP/TN；
- 按极化分组结果；
- 最差文件/极化组合；
- 是否与论文 Fig. 7 口径一致，以及不一致项。

## 6. 训练日志目录

所有训练和评估脚本后续应统一把日志写入：

```text
logs/training/<run_id>/
```

`run_id` 推荐格式：

```text
YYYYMMDD-HHMMSS_<experiment_name>
```

每个 run 目录至少包含：

```text
config.yaml              # 本次运行配置快照
stdout.log               # 训练/评估标准输出
metrics.csv/json         # epoch 指标或最终评估指标
summary.md               # 面向论文记录的结果摘要
artifacts.txt            # checkpoint、图表、报告文件路径
```

`checkpoints/` 可以继续保存模型权重，但正式报告必须能从 `logs/training/<run_id>/artifacts.txt` 追溯到对应 checkpoint。

## 7. 可复用结论与禁区

可以优先复用：

- PL/RPH；
- cell-wise stats；
- amplitude/phase/delta phase；
- Doppler/FFT 统计；
- 距离邻接、距离衰减、相似性空间图；
- Pfa-aware loss 或训练集杂波阈值。

当前不作为主线：

- 单纯把 STFT/WVD 图送入普通 CNN；
- 大型 PLM 或复杂预训练；
- 仅使用 global stats 作为主创新；
- 未重新设计的小 N Local RMS/CFAR 大窗口；
- 使用测试集统计量做归一化或正式阈值。

## 8. 修改前检查

新增实验前先回答：

1. 这是雷达特征、空间图、时间模块、杂波门控、loss，还是评估逻辑？
2. 是否保持 `model(E_complex) -> [B, 2, N]`？
3. 阈值是否来自训练/校准杂波，而不是测试集？
4. 是否需要一个 negative control，例如 stats-only、global-only 或 shuffle stats？
5. 是否能用小样本命令先验证 shape 和 loss 不崩？
