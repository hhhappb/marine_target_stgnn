# 论文模块实验包说明

这个目录专门放新论文实验模块，原则是：

```text
一个研究思路 -> 一个独立代码文件 -> 一个可配置实验
```

原始 `models/st_gnn.py` 保持不动，作为已经复现成功的 ST-GNN baseline。新思路都放在 `paper_modules/` 中，通过 YAML 配置切换。

## 思路与代码对应关系

| 研究思路 | 代码文件 | 典型配置 |
|---|---|---|
| 原 ST-GNN 输入：只用实部/虚部 | `models/modules/radar_features/real_imag.py` | `configs/baseline.yaml` |
| 幅相输入：加入幅度、相位正余弦 | `models/modules/radar_features/amplitude_phase.py` | `configs/radar_feature.yaml` |
| 幅相稳定性：加入相邻脉冲幅度差、相位差 | `models/modules/radar_features/phase_diffs.py` | `configs/radar_feature.yaml` |
| Doppler 先验：加入慢时间 FFT 统计 | `models/modules/radar_features/doppler_stats.py` | `configs/full_model.yaml` |
| 原 ST-GNN 空间图：固定相邻距离单元 | `models/modules/spatial_graphs/local_range.py` | `configs/baseline.yaml` |
| 纯动态空间图：全距离单元注意力 | `models/modules/spatial_graphs/dynamic_attention.py` | `spatial_graph.type: pure_dynamic` |
| 距离衰减空间图：远距离边受惩罚 | `models/modules/spatial_graphs/distance_decay.py` | `spatial_graph.type: distance_dynamic` |
| 雷达先验动态图：局部邻接 + 距离衰减 + 特征相似 | `models/modules/spatial_graphs/prior_dynamic.py` | `configs/dynamic_spatial_graph.yaml` |
| 原 ST-GNN 时间处理：卷积门控压缩 | `models/modules/temporal_modules/convgru_baseline.py` | `configs/baseline.yaml` |
| 多尺度时间卷积：短/中/长时间感受野 | `models/modules/temporal_modules/multiscale_tcn.py` | `configs/temporal_graph.yaml` |
| Range migration 时间混合：允许跨邻近距离单元传播 | `models/modules/temporal_modules/range_migration.py` | `configs/range_migration_temporal.yaml` |
| 时间消融：只做脉冲维平均池化 | `models/modules/temporal_modules/temporal_pool.py` | `temporal.type: mean_pool` |
| 无杂波门控基线 | `models/modules/clutter_gates/identity.py` | `clutter_gate.enabled: false` |
| 局部海杂波统计门控：抑制疑似 sea spike | `models/modules/clutter_gates/local_statistics.py` | `configs/clutter_gate.yaml` |
| 标准交叉熵训练目标 | `losses/cross_entropy.py` | `configs/baseline.yaml` |
| Pfa-aware loss：惩罚杂波高分尾部 | `losses/pfa_aware.py` | `configs/pfa_aware_loss.yaml` |

## 统一接口

所有模型变体都保持同一个接口：

```python
logits = model(E_complex)
```

其中：

```text
E_complex: [B, P, N]    # 复数雷达回波
logits:    [B, 2, N]    # 每个距离单元的杂波/目标二分类 logits
```

## 常用命令

运行单个实验：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\baseline.yaml
```

快速小样本试跑：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\full_model.yaml --epochs 1 --max-train-windows 64 --max-test-windows-per-file 5 --no-progress --log-interval 1
```

预览批量消融命令：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\run_ablation.py --configs paper_modules\configs\baseline.yaml paper_modules\configs\radar_feature.yaml --dry-run
```
