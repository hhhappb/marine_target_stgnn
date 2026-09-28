# marine_target_stgnn

本仓库研究 **ST-GNN 的空间建模和时间建模改进**。`main` 保存原始 ST-GNN
基线、可配置的模块实现及实验协议；不能把仓库中存在某个模块理解为默认模型
已经启用它，或理解为该模块的独立效果已经得到证明。

## `main` 中选定的建模改进

[PR #1](https://github.com/hhhappb/marine_target_stgnn/pull/1)
（合并提交 `10e2906`）集成的是以下**组件与协议支持**，并非完整 E4 模型：

| 位置 | 选定实现 | 含义 |
|---|---|---|
| 空间 SFE1/SFE2 | [`RadarPriorDynamicSFE`](paper_modules/models/modules/spatial_graphs/radar_prior_dynamic_sfe.py) | 动态相似度连接限制在半径 5 的局部距离窗口；**静态图仍保留全局距离衰减先验**。固定参数 `static_gamma=0.5`、`static_weight=0.7`、`dynamic_temperature=0.2`、`dropout=0.1`；已移除 `dynamic_topk` 非局部补边。 |
| 时间 TFE1 | [`PulseAttentionOnlyTFE`](paper_modules/models/modules/temporal_modules/pulse_attention_tfe.py) | 选定 A1：每个距离单元内部沿脉冲维做注意力，`attention_dim=64`、`num_heads=4`、固定 `residual_scale=0.1`、`use_attention=true`。 |
| 时间 TFE2 | 原始 ST-GNN 门控卷积 | A1 方案**只替换 TFE1**，并未把 TFE2 也替换成注意力模块。 |

模块由配置和模型入口显式选择；[`OriginalSTGNN`](paper_modules/models/original_stgnn.py)
的默认行为仍是原始基线。IPIX 的
[`ipix_pax_l1_a1_tfe1`](paper_modules/models/ipix_pax_l1_a1_tfe1.py)
是 **P=16、N=14、原始空间模块＋A1 TFE1** 的专用适配器，不是 E4，也不代表
“空间＋时间”联合模型。SDRDSP 的
[`sdrdsp_v11_p8_a1_attention_tfe1_seed42.yaml`](paper_modules/configs/sdrdsp_v11_p8_a1_attention_tfe1_seed42.yaml)
同样是 TFE1 替换配置；使用前应核对其中四个槽位和数据协议。

`main` 中还可能保留历史实验模块、配置与报告，例如 SNDD。它们不是本节
选定 A1 的别名；请根据实际配置检查 SFE1/TFE1/SFE2/TFE2，不能仅凭
`optimized`、`latest` 或文件名判断模型结构。只替换 TFE1 的结果也不能写成
双级时间模块的独立归因证据。更多来源与边界见
[集成说明](docs/selected_spatial_a1_integration.md)。

## E4 与本仓库的关系

E4-D2/GDCM 是可使用上述 ST-GNN 模块的**下游整体系统**，包含额外的幅相、
全局特征分支、融合与检测头。PR #1 **没有将完整 E4、其模型权重或整体实验结果
提交到 `main`**。本地针对 P=8 的“去补边空间＋A1 TFE1”E4 联合实验，不能
反推为 GitHub `main` 已提供这一完整模型或已验证模块的独立贡献。其他工作树
和未提交文件也不会随 `git clone` 出现。

若要在 E4 分支接入，应从当前 `main` 核对上述两个组件的实际实现与参数，
在 E4 自身的构建入口显式设置 SFE1/SFE2、TFE1/TFE2，并保持其余分支及检测头
符合该实验的配置。不得沿用历史 E4 的“optimized SFE/TFE”命名来推断时间模块：
此前整体 E4 实验曾使用**双级 SNDD**，不是此处的 A1。

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Structure

- `paper_modules/experiments/train.py`: unified training and evaluation entry point
- `paper_modules/experiments/auto_experiment.py`: suite runner for comparable experiments
- `paper_modules/experiments/leakage_probe.py`: diagnostic probe for range-position leakage
- `models/`: frozen ST-GNN baseline model implementations
- `paper_modules/models/`: configurable experimental model modules
- `paper_modules/datasets/`: dataset registry and dataset adapters
- `data/`, `datasets/`: prepared and raw datasets
- `checkpoints/`: saved model weights
- `logs/training/`: training and evaluation run logs

## Commands

IPIX 预处理默认使用 McMaster 官方 `ipixload.m` 的 `auto` 分支，逐距离单元处理并保留 double 精度：

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_ipix.py
```

输出固定写入 `datasets/ipix_dartmouth/processed/window4_stride4_related_official_ipixload_auto_double`。
脚本拒绝覆盖已存在的输出目录。官方源码地址为
`https://soma.ece.mcmaster.ca/ipix/dartmouth/mfiles/ipixload.m`，固化 SHA-256 为
`40f5499eeb0d1b1d7e158658bdcfddb31a18ef01eea4023e7da8ca0ba21bd517`。
训练接口通过 `dataset.expected_processing_mode: official_ipixload_auto` 校验数据来源，
并仅在模型输入边界将 complex128 转成现有模型契约所需的 complex64/float32。

IPIX module smoke:

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\real_imag_sfe_replacement_original_sfe.yaml --epochs 1 --max-train-windows 64 --max-test-windows-per-file 5 --no-progress --log-interval 1
```

SDRDSP v1.1 的生成参数和 P=8 对照配置见
[实验协议](docs/sdrdsp_experiment_protocol.md)、
[`sdrdsp_v11_p8_a0_original_seed42.yaml`](paper_modules/configs/sdrdsp_v11_p8_a0_original_seed42.yaml)
和上文 A1 配置。配置中的数据与输出目录应在运行前核对；不得把仓库以外的
本地数据、权重或日志当作 `main` 自带内容。

