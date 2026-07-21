# SDRDSP attention logits scale scan（冻结 checkpoint）

## 协议

- 数据协议：`sdrdsp_fig9_local_crop_v2`
- 输入：`P=4, N=256, real_imag`
- checkpoint：`logs/training/20260721-095151_corrected_tfe_n256_seed42/runs/sdrdsp_pulse_attention_only_n256_seed42_seed42/best_model.pth`
- 评分：`logit_margin = target_logit - clutter_logit`
- 阈值：每个倍率分别使用训练集杂波校准
- 测试：20 个 SCR 点，每点 1,630 个窗口；未截断
- 训练：未执行；所有结果来自同一冻结 checkpoint

## Pfa=0.001 结果

| 模式 | PD | 实际 PF | 归一化 PD-SCR AUC | 低 SCR 平均 PD | target entropy | clutter entropy |
|---|---:|---:|---:|---:|---:|---:|
| learned ×1 | 0.947730 | 0.000621 | 0.960381 | 0.663395 | 0.999967 | 0.999928 |
| learned ×2 | 0.947791 | 0.000654 | 0.960462 | 0.663395 | 0.999870 | 0.999712 |
| learned ×4 | 0.947822 | 0.000643 | 0.960462 | 0.663395 | 0.999481 | 0.998853 |
| learned ×8 | 0.948589 | 0.000659 | 0.961011 | 0.668303 | 0.997953 | 0.995468 |
| learned ×16 | 0.948988 | 0.000662 | 0.961350 | 0.670757 | 0.992130 | 0.982555 |
| forced-uniform | 0.947423 | 0.000628 | 0.960107 | 0.661759 | 1.000000 | 1.000000 |

## 结论

提高 logits scaling 会降低 attention entropy，并带来小幅冻结性能提升；但目标与杂波 entropy 同时下降，尚未形成清晰的目标选择性。相对 forced-uniform，learned ×16 的 PD 提升 `+0.001565`、AUC 提升 `+0.001243`，不足以称为明确的 Q/K 选择收益。

因此暂不启动新的 learnable-temperature 训练；Q/K attention 保留为已验证但尚未证明选择价值的诊断方向。
