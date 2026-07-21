# SDRDSP N=256 修正时间模块筛选

## 协议

- 数据：`data/sdrdsp_strict_256_v2`
- 协议：`sdrdsp_fig9_local_crop_v2`
- 输入：`P=4, N=256, real_imag`
- SFE：`original_stfe`
- TFE：只替换 TFE1，TFE2 保持 `stgnn_tfe`
- seed：42
- 训练：Adam，学习率 `0.001`，batch `24`，10 epochs，1000 optimizer steps
- 阈值：训练集杂波
- 评分：`logit_margin = target_logit - clutter_logit`
- 目标 Pfa：`0.0001 / 0.001 / 0.01`

已有 N=256 原始 ST-GNN 对照（同一协议、同一 seed、同一训练预算）：Pfa=0.001 时 `PD=0.930552`、`PF=0.000809`、归一化 PD-SCR AUC `0.944624`。

低 SCR 的 PD 对照（SCR=-24/-22/-20 dB）：原始 baseline 为 `0.3264/0.5810/0.7933`；Corrected Diff-only 为 `0.1957/0.3853/0.6031`；Pulse-attention-only 为 `0.4147/0.6945/0.8810`。

## 结果

| 模型 | Pfa | PD | 实际 PF | PD-SCR AUC |
|---|---:|---:|---:|---:|
| Corrected Diff-only | 0.0001 | 0.852055 | 0.000050 | 0.868712 |
| Corrected Diff-only | 0.001 | 0.896380 | 0.000619 | 0.912092 |
| Corrected Diff-only | 0.01 | 0.935552 | 0.009256 | 0.947966 |
| Pulse-attention-only | 0.0001 | 0.905153 | 0.000079 | 0.922683 |
| Pulse-attention-only | 0.001 | 0.947730 | 0.000623 | 0.960381 |
| Pulse-attention-only | 0.01 | 0.973988 | 0.006556 | 0.981240 |

## 诊断

### Corrected Diff-only

- `diff_residual_ratio = 0.000102`
- `diff_gate_mean = 0.5782`
- `tfe_input_rms = 599.9133`
- `tfe_enhanced_rms = 599.9140`

差分门控虽然有非零均值，但最终残差相对输入约为 `1e-4`，当前 `diff_scale=0.1` 下基本没有改变 TFE1 输入；该方向本轮没有正向信号。

### Pulse-attention-only

- 注意力 shape：`[B×N, H, P, P] = [B×256, 4, 4, 4]`
- `attention_entropy = 0.999930`
- `attention_max_weight = 0.253613`
- `attention_diagonal_fraction = 0.249613`
- `attention_residual_ratio = 1.808794`
- `tfe_input_rms = 1826.0244`
- `tfe_enhanced_rms = 4655.2217`

注意力权重接近 P=4 的均匀分布，尚不能称为“选择性 pulse attention”；但该模块在本轮单 seed 筛选中超过原始 baseline：Pfa=0.001 的 PD 提升 `+0.017178`，实际 PF 低于 baseline。

## 参数与耗时

| 模型 | 可训练参数 | batch=24 单批前向耗时 |
|---|---:|---:|
| Original ST-GNN | 5,066,274 | 0.00594 s |
| Corrected Diff-only | 5,132,578 | 0.01106 s |
| Pulse-attention-only | 5,099,170 | 0.00803 s |

训练 run 总耗时约为：Diff-only `231.369 s`，Pulse-attention-only `231.648 s`。

## 结论与下一步闸门

1. 修正后的 Diff-only 在当前设置下仍失败，且差分残差几乎关闭；不继续扩展 N=128/N=512 或 P=16。
2. Pulse-attention-only 产生了 seed=42 的正向筛选信号，但注意力近似均匀、残差幅度过大，不能直接声称“有效时间注意力”。
3. 按预注册条件，下一步只允许比较 pulse attention 只替换 TFE1 与同时替换 TFE1/TFE2；仍保持 N=256、seed=42 和相同阈值口径。
4. 以上仍是单 seed 候选证据，不是三 seed 正式结论。

## 追溯

- 训练 run：`logs/training/20260721-095151_corrected_tfe_n256_seed42/`
- 正确 `logit_margin` 评估结果：`logs/training/sdrdsp_corrected_diff_only_n256_seed42/`、`logs/training/sdrdsp_pulse_attention_only_n256_seed42/`
- 最终诊断字段位于各自 `metrics.json` 的 `final_temporal_diagnostics`，范围标记为 `last_evaluation_batch`。
- 评估器后来补写了 `score_space` 和 `PD_SCR_AUC`；未来 eval-only 已修正为保留既有 `training_history`。
