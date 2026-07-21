# SDRDSP fixed-uniform temporal mixer 筛选结果

## 协议

- 数据：`data/sdrdsp_strict_256_v2`
- 协议：`sdrdsp_fig9_local_crop_v2`
- 输入：`P=4, N=256, real_imag`
- SFE：`original_stfe`
- TFE：只替换 TFE1；TFE2 保持 `stgnn_tfe`
- seed：`42`
- 训练：Adam，学习率 `0.001`，batch `24`，10 epochs，每 epoch 1013 个 optimizer steps
- 阈值：训练集杂波
- 评分：`logit_margin = target_logit - clutter_logit`
- 目标 Pfa：`0.0001 / 0.001 / 0.01`

fixed-uniform TFE1 在每个距离单元内对 P 个 pulse 做固定均值混合，保留 value/output 投影、`residual_scale=0.1` 和原始 TFE 门控，但删除 Q/K 与 softmax。

## 结果

| 模型 | Pfa | PD | 实际 PF | PD-SCR AUC |
|---|---:|---:|---:|---:|
| Original ST-GNN | 0.0001 | 0.886932 | 0.000118 | 0.904133 |
| Fixed-uniform TFE1 | 0.0001 | 0.862914 | 0.000089 | 0.879916 |
| Pulse-attention learned | 0.0001 | 0.905153 | 0.000079 | 0.922683 |
| Original ST-GNN | 0.001 | 0.930552 | 0.000809 | 0.944624 |
| Fixed-uniform TFE1 | 0.001 | 0.918221 | 0.000772 | 0.933516 |
| Pulse-attention learned | 0.001 | 0.947730 | 0.000623 | 0.960381 |
| Original ST-GNN | 0.01 | 0.960337 | 0.008945 | 0.969761 |
| Fixed-uniform TFE1 | 0.01 | 0.954387 | 0.008737 | 0.965370 |
| Pulse-attention learned | 0.01 | 0.973988 | 0.006556 | 0.981240 |

Pfa=0.001 时低 SCR 三点：

| 模型 | -24 dB | -22 dB | -20 dB | 平均 |
|---|---:|---:|---:|---:|
| Original ST-GNN | 0.3264 | 0.5810 | 0.7933 | 0.5669 |
| Fixed-uniform TFE1 | 0.2552 | 0.4939 | 0.7344 | 0.4945 |
| Pulse-attention learned | 0.4147 | 0.6945 | 0.8810 | 0.6634 |

## 计算量与诊断

| 模型 | 可训练参数 | batch=24 单批前向耗时 |
|---|---:|---:|
| Original ST-GNN | 5,066,274 | 0.00594 s |
| Fixed-uniform TFE1 | 5,082,786 | 0.00742 s |
| Pulse-attention learned | 5,099,170 | 0.00803 s |

Fixed-uniform 的理论注意力统计为 entropy `1.0`、max weight `0.25`、diagonal fraction `0.25`。训练过程中 residual/input ratio 从约 `0.17` 增长到约 `2.15`，但最终测试最后一个 batch 的诊断为：

- `tfe1_attention_residual_ratio = 0.6778`
- `tfe1_tfe_input_rms = 537.41`
- `tfe1_tfe_enhanced_rms = 666.95`

## 结论

1. fixed-uniform TFE1 在当前 N=256、seed42、10 epoch 协议下没有超过原始 ST-GNN：Pfa=0.001 的 PD 为 `0.918221`，比 baseline 低 `0.012331`。
2. learned checkpoint 的 uniform 反事实几乎不损失性能，但从头训练 fixed-uniform 后性能下降，说明收益不能简单归因于固定 pulse 均值混合；更可能与 learned 模块的联合参数适配、残差尺度或训练动力学有关。
3. 当前没有足够证据证明 Q/K 选择性注意力有效，也没有足够证据支持 fixed-uniform 作为改进模块。
4. 按 seed42 筛选闸门，暂不继续训练 TFE1+TFE2、N=128/512 或 P=16；该时间模块方向应先暂停，避免继续消耗训练成本。

## 追溯

- 配置：`paper_modules/configs/sdrdsp_fixed_uniform_tfe_n256_seed42.yaml`
- 训练目录：`logs/training/sdrdsp_fixed_uniform_tfe_n256_seed42/`
- checkpoint：`logs/training/sdrdsp_fixed_uniform_tfe_n256_seed42/best_model.pth`
- 评估结果：`logs/training/sdrdsp_fixed_uniform_tfe_n256_seed42/eval_results.json`
- 诊断结果：`logs/training/20260721-105325_sdrdsp_temporal_attention_audit/diagnostics.json`
- 本结果为单 seed 筛选证据，不构成三 seed 正式模型结论。
