# SDRDSP 修正 Pulse-attention 的 P=8/16/32 对比

## 实验口径

- 数据协议：`sdrdsp_fig9_local_crop_v2`
- N：256
- P：8、16、32
- seed：42
- 输入：`real_imag`
- SFE：`original_stfe`
- 候选：只替换 TFE1 为修正后的 `PulseAttentionOnlyTFE`，TFE2 保持原始 TFE
- 对照：`original SFE + original TFE`
- optimizer：Adam，learning rate=0.001，batch size=24
- epochs：10
- 训练窗口：每个 P 固定抽取 2400 个训练窗口
- optimizer steps：每个 run 每 epoch 100 steps，共 1000 steps
- 阈值：各模型自己的 2400 窗口训练杂波
- score：`logit_margin`

P=8 和 P=32 数据由同一 SDRDSP Fig. 9 预处理脚本、同一 seed=42 生成；P=16 使用已有同协议数据。全部数据通过严格 manifest 和 SCR 精度审计。

## Pfa=0.001 结果

| P | 模型 | PD | 实际 PF | PD-SCR AUC | 低 SCR 平均 PD |
|---:|---|---:|---:|---:|---:|
| 8 | Original TFE | 0.901963 | 0.001213 | 0.919858 | 0.385276 |
| 8 | Pulse-attention | 0.910736 | 0.000736 | 0.926154 | 0.460532 |
| 16 | Original TFE | 0.944472 | 0.000749 | 0.958684 | 0.631450 |
| 16 | Pulse-attention | 0.880590 | 0.000434 | 0.899716 | 0.257166 |
| 32 | Original TFE | 0.872660 | 0.000487 | 0.892274 | 0.175698 |
| 32 | Pulse-attention | 0.929803 | 0.000491 | 0.949313 | 0.535304 |

相对同 P baseline：

| P | PD 差值 | AUC 差值 | 低 SCR PD 差值 |
|---:|---:|---:|---:|
| 8 | +0.008773 | +0.006296 | +0.075256 |
| 16 | -0.063882 | -0.058968 | -0.374284 |
| 32 | +0.057143 | +0.057039 | +0.359606 |

## Attention 诊断

| P | entropy | max weight | diagonal fraction | residual/input |
|---:|---:|---:|---:|---:|
| 8 | 0.997169 | 0.140918 | 0.124574 | 0.247728 |
| 16 | 0.997873 | 0.072344 | 0.061915 | 0.145728 |
| 32 | 0.997739 | 0.040236 | 0.031158 | 0.139823 |

理论均匀注意力在 P=8/16/32 下的 max weight 和 diagonal fraction 分别约为 1/P：0.125、0.0625、0.03125。实际统计非常接近这些值，因此 P 增大没有让注意力变成明显的选择性 Q/K 注意力。

## 结论

1. P 增大没有稳定地让修正时间模块变得可用：P=8 只有小幅提升，P=16 明显失败，P=32 相对同 P baseline 有较大提升。
2. P=32 的正向结果是真实的候选信号：PD 提升 `+0.0571`，实际 PF 几乎不变，低 SCR 平均 PD 提升 `+0.3596`。但它仍低于 P=16 原始 TFE 的 PD=`0.9445`，不能解释为 P 越大越好。
3. P=32 baseline 自身发生了明显退化，因此 P=32 候选的提升部分可能是“修复长窗口下原始 TFE 的退化”，而不是选择性 attention 本身带来的普遍增益。
4. P=8/16/32 的注意力仍近似均匀，不能声称 P 增大验证了选择性时间注意力。
5. 本轮是单 seed 筛选；P=32 候选值得保留为后续候选，但正式主线前至少需要同协议补 seed=123、2026，并同时保留同 P 原始 TFE baseline。

## 与 P=4 的关系

此前 P=4 修正 pulse-attention 结果使用完整训练集 24,290 个窗口、10,130 steps；本轮 P=8/16/32 为了公平比较统一使用 2400 个窗口、1000 steps。因此 P=4 旧结果不能直接与本表构成严格的 P 曲线。若要形成完整 P=4/8/16/32 消融，应重新按本轮 2400 窗口协议补跑 P=4 baseline 和候选。

## 产物

- 自动化运行目录：`logs/training/20260721-115930_corrected_tfe_observation_p8_p16_p32_seed42/`
- 自动化摘要：`logs/training/20260721-115930_corrected_tfe_observation_p8_p16_p32_seed42/summary.md`
- P=8/P=16/P=32 数据目录：`data/sdrdsp_strict_256_p8_v1`、`data/sdrdsp_strict_256_p16_v1`、`data/sdrdsp_strict_256_p32_v1`
