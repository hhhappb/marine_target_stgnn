# IPIX Cell-Wise 统计预处理可分性验证

- data_dir: `datasets\ipix_dartmouth\processed\window4_stride4_related`
- splits: `train, test`
- polarizations: `hh, hv, vv, vh`
- max_windows_per_file: `None`
- 方法：不训练模型，只计算每个距离单元的统计特征，并用目标/杂波标签计算单特征 AUC。
- `auc_target_high > 0.5` 表示目标单元该特征更大；`< 0.5` 表示目标单元该特征更小。
- `separability_auc = max(AUC, 1-AUC)`，越接近 1 表示单特征越能区分目标/杂波。

## 总体单特征结果

| split | feature | direction | AUC(target high) | separability AUC | target median | clutter median |
|---|---|---|---:|---:|---:|---:|
| test | phase_std | target_low | 0.3556 | 0.6444 | 0.2984 | 0.6198 |
| test | phase_stability | target_high | 0.6444 | 0.6444 | 0.9050 | 0.8027 |
| test | amp_std | target_low | 0.3826 | 0.6174 | 0.1051 | 0.1575 |
| test | amp_contrast | target_high | 0.5981 | 0.5981 | 0.9886 | 0.9777 |
| test | amp_p95 | target_low | 0.4664 | 0.5336 | 0.9345 | 1.0280 |
| test | coherent_mean | target_high | 0.5260 | 0.5260 | 0.6605 | 0.6082 |
| test | amp_rms | target_low | 0.4807 | 0.5193 | 0.7863 | 0.8418 |
| test | amp_mean | target_low | 0.4867 | 0.5133 | 0.7608 | 0.8047 |
| train | phase_std | target_low | 0.3436 | 0.6564 | 0.2831 | 0.6307 |
| train | phase_stability | target_high | 0.6564 | 0.6564 | 0.9099 | 0.7992 |
| train | amp_std | target_low | 0.3846 | 0.6154 | 0.1056 | 0.1589 |
| train | amp_contrast | target_high | 0.5995 | 0.5995 | 0.9888 | 0.9776 |
| train | coherent_mean | target_high | 0.5367 | 0.5367 | 0.6802 | 0.6038 |
| train | amp_p95 | target_low | 0.4702 | 0.5298 | 0.9495 | 1.0314 |
| train | amp_rms | target_low | 0.4854 | 0.5146 | 0.8001 | 0.8434 |
| train | amp_mean | target_low | 0.4915 | 0.5085 | 0.7753 | 0.8054 |

## 最稳定的候选特征

- train: phase_std(0.656, target_low), phase_stability(0.656, target_high), amp_std(0.615, target_low)
- test: phase_std(0.644, target_low), phase_stability(0.644, target_high), amp_std(0.617, target_low)

## 分文件表现提示

- train: 198/448 个 文件-极化-特征 组合达到 separability_auc >= 0.6。
- test: 208/448 个 文件-极化-特征 组合达到 separability_auc >= 0.6。
