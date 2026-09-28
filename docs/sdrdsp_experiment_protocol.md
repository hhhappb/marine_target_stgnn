# SDRDSP 实验协议

本文件记录自2026年9月22日起使用的SDRDSP v1.1本地协议，2026年9月28日补充SCR口径审计及P=8运行设置。此前的2400窗快筛、10 epochs、P=16全量训练和logit margin结果仅保留为历史诊断，不进入新实验排序或结论。下文P=4、20轮是快筛设置，不代表全部后续预算。见 [协议索引](experiment_protocols.md)。

## SCR定义及论文差异

当前 `scripts/preprocess_sdrdsp_v11.py` 对P个脉冲与M=20个随机非目标参考单元一起取平均。

```text
cp_mean = sum_p sum_m |clutter[p,m]|^2 / (P * M)
A = sqrt(cp_mean * 10^(SCR_mean / 10))
target[p] = A * exp(j * (phi0 + 4*pi*v*PRT*p/lambda))
```

P=4和P=8数据manifest均标记 `scr_reference_power=per_window_random_20_cell_mean_power`。

ST-GNN论文式(17)对M个参考单元功率求和，没有除以M。在相同脉冲平均方式下，旧E4脚本 `e4d2_gdcm/experiments/train_sdrdsp.py` 的sum/P与当前定义关系为

```text
cp_sum = M * cp_mean
SCR_sum = SCR_mean - 10*log10(20) = SCR_mean - 13.0103 dB
```

同一个目标的当前SCR比求和口径高13.0103 dB。相同SCR标签下，当前目标功率为求和口径的1/20，幅度为1/sqrt(20)。不能把当前横轴直接称为论文式(17)口径，也不能将跨协议曲线差值归因于模型。代数换算不代表两套完整生成流程或实验结果可互相替代。

来源为2026年9月22日提供的《SDRDDSP数据集实验协议(1).docx》第3.2节cp说明与第3.3节SCR定义。材料明确写了 `/(P*M)`，但同时标注为Eq.17，与论文存在上述差异。当日按材料及用户指定的100轮、seed42、最低训练损失checkpoint生成本Markdown和代码，9月26日由提交 `de3a93b` 入库。此次仅记录实际口径，不改变历史数据或结果。

## P=8扩展及运行预算

| 版本 | 每SCR训练窗 | 每SCR测试窗 | 总训练窗 | 用途 |
|---|---:|---:|---:|---|
| P=4、步长4 | 1734 | 1629 | 24276 | v1.1基础协议 |
| P=8、步长8 | 867 | 814 | 12138 | 更长观察窗扩展 |

两版本均舍弃最后一个可完整取出的窗口。数据分别位于 `data/sdrdsp_v11_p4_n256_seed42` 和 `data/sdrdsp_v11_p8_n256_seed42`，协议ID分别为 `sdrdsp_fig9_v11_p4_n256_seed42` 和 `sdrdsp_fig9_v11_p8_n256_seed42`。

20轮为候选快筛，100轮为历史确认预算，最近2026年9月28日Original/E4-selected配对配置为P=8、120轮、seed42、batch24、学习率0.001、SCR均衡采样、最低训练损失checkpoint。必须按相同预算比较，不把120轮与20轮结果混排。

当前配对的本地证据目录为

- `logs/training/20260928_sdrdsp_v11_p8_original_120ep_seed42_pair/`，配置为 `config.yaml`。
- `logs/training/20260928_sdrdsp_v11_p8_e4_selected_120ep_seed42_pair_fixed_sampler/`，配置为 `config.json`，来源为 `provenance.json`。使用修复采样器版本，不与同名未修复运行混用。

Original使用I/Q及原始两级SFE/TFE与检测头。E4-selected采用E4特征、gdcm_a0检测头，SFE1/SFE2为radar_prior_dynamic_sfe，TFE1为pulse_attention_only_tfe，TFE2为stgnn_tfe。空间和时间参数与 [IPIX当前协议](ipix_experiment_protocol.md)列出的selected配置一致。比较属于整体模型方案，不是单模块归因。

预生成NPZ使用训练P99归一化。E4分支另有训练拟合的E4预处理，来源记录 `raw_reconstruction_p99=3121.6865234375`、`e4_train_preprocessor_p99=3475.56396484375`，不能称为两模型输入特征完全相同。此处记录运行配置，不据此宣称训练完成或完整E4代码已经进入main。

## 数据和目标构造

- 协议ID：sdrdsp_fig9_v11_p4_n256_seed42。
- 训练背景：20210106155330_01_staring.mat，裁剪为6940脉冲乘256距离单元。
- 测试背景：20210106155432_01_staring.mat，裁剪为6520脉冲乘256距离单元。
- 裁剪中心对应论文第2083距离单元，窗内下标为128。
- 每个样本包含4个连续脉冲，步长为4。
- 按协议文档舍弃最后一个仍可完整取出的窗口，训练每SCR为1734窗，测试每SCR为1629窗。
- 训练SCR为-12至14 dB，间隔2 dB；测试SCR为-24至14 dB，间隔2 dB。
- 每个训练窗注入5个目标，距离间隔至少10格；测试目标固定在下标128。
- 训练速度从0.1至0.5 m/s确定性抽取；测试速度为0.4 m/s。
- 每个目标使用20个随机非目标参考单元，以窗口内脉冲和参考单元的平均功率定义杂波功率。
- 每个目标每个窗口独立抽取初相，初相随机流固定为RandomState(777)。
- 训练背景复幅度的99分位数记为P99；所有深度模型输入统一使用I/P99和Q/P99，测试沿用训练P99。

## 训练

- 当前正式运行只使用seed42。
- 时间建模候选统一使用20 epochs快速筛选，batch size为24。
- 优化器为Adam，固定学习率0.001，不使用学习率调度或权重衰减。
- 使用普通交叉熵，不使用类别权重。
- 批次按14个训练SCR近似均衡构造；由于24不能被14整除，每批各SCR样本数相差不超过1，额外名额按批次轮换。
- 启用确定性训练。
- checkpoint按最低训练损失选择，测试集不参与选择。
- 20 epochs结果用于判断候选方向是否具有正向信号；候选与Original必须在同一20 epochs预算下重新配对，不得与历史100 epochs结果直接比较。

## 阈值和评价

- 正式评分使用softmax通道0的杂波概率o0。
- 阈值只使用训练集全部杂波距离单元确定。
- 将训练杂波o0升序排列，索引为ceil(Pfa乘Nc)-1。
- o0不大于threshold时判为目标，否则判为杂波。
- 正式Pfa为0.0001、0.001和0.01。
- 主结果报告逐SCR测试PD；测试实际PF仅作为诊断，不参与模型排序。
- 当前单seed结果只描述seed42下的配对差异，不作随机性显著性声明。

## 入口

生成数据：

    .\.venv\Scripts\python.exe scripts\preprocess_sdrdsp_v11.py

原始MAT默认从只读参考目录E:/stgnn/data读取；生成器只向新的data/sdrdsp_v11_p4_n256_seed42目录写入。

Original基线：

    .\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\sdrdsp_v11_original_seed42.yaml

SNDD双级时间模块：

    .\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\sdrdsp_v11_sndd_both_seed42.yaml
