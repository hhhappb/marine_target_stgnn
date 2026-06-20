# marine_target_stgnn Project Rules

这些规则用于约束后续代码修改、实验扩展和结果表述。默认先遵守本文件，再参考局部 README 或实验说明。

## 1. 修改边界

- `models/st_gnn.py` 是论文 ST-GNN 复现 baseline，默认冻结。除非明确修 bug，不在这里加入新研究想法。
- 新论文想法、消融模块和可配置实验都放在 `paper_modules/`。
- `paper_model/` 视为早期/备用实现，不作为新实验主线扩展位置。
- `train_ipix.py` 是 IPIX baseline 训练评估入口；新增实验优先使用 `paper_modules/experiments/`。
- 不做顺手重构。每次修改必须能对应到当前研究问题、实验协议或明确 bug。

## 2. 敏感模型代码审核

模型代码是高敏感区域，默认必须先说明修改意图和影响范围，再实施修改。

敏感模型代码包括：

- `models/`
- `paper_model/`
- `paper_modules/models/`
- `paper_modules/losses/`

修改这些路径时必须满足：

- 先写清楚：改哪个模块、为什么改、是否影响 baseline、预期输入输出 shape。
- 不允许在一次修改里同时改多个研究想法。一个实验想法对应一组最小代码改动。
- 不允许绕过 registry、配置或统一接口直接硬编码实验分支。
- 修改 `models/st_gnn.py`、`paper_model/` 或已有 baseline 行为前，必须经过人工审核确认；除非是明显 bugfix，也要在最终说明中单独列出。
- 模型代码修改完成后，至少做一次 `[B, 4, 14] complex -> [B, 2, 14]` 的接口检查。

## 3. 稳定接口

所有 ST-GNN 变体必须保持统一模型接口：

```python
logits = model(E_complex)
```

契约如下：

```text
E_complex: complex tensor [B, P, N]
logits:    float tensor   [B, 2, N]
```

当前 IPIX 主线默认：

```text
P = 4
N = 14
polarizations = hh, hv, vv, vh
label = y_range
target_policy = related
```

如果模块改变输入通道，只能在雷达特征编码器内部完成，不能改变外部 `E_complex` 契约。

## 4. 实验模块规则

- 一个研究思路对应一个独立实现文件、一个 registry 条目、一个 YAML 配置。
- 新模块必须 fail loud：未知 `type`、不支持形状、缺少必要配置时直接抛错，不静默回退。
- 雷达先验模块必须写清物理含义，例如相位线性度、Doppler 峰值、cell-wise stats、CFAR 参考窗或距离衰减。
- 普通优化器、scheduler、dropout、batch size、梯度裁剪、通用 MLP/Conv 堆叠不能包装成“雷达先验”。
- IPIX `N=14` 与 SDRDSP `N=256` 不能混用参数假设。依赖大范围距离参考窗的方法必须重新设计并说明原因。

## 5. 评估与结果口径

- Pfa/FAR 阈值必须声明来源：训练集、校准集、测试集或临时诊断集。
- 正式结果不得使用测试集杂波分布估计阈值；如果为了诊断临时这样做，报告必须明确标注。
- 评价至少报告 `PD`、实际 `PF`、阈值、TP/FN/FP/TN、极化维度结果和困难文件。
- 与论文 Fig. 7 或其他论文对比时，必须说明标签口径、阈值来源、数据切分和目标 Pfa。
- 不允许通过改变标签定义、泄漏测试统计量或筛掉困难极化来提升指标。

## 6. 数据与产物

- 原始 `.cdf`、处理后 `.npz`、模型权重 `.pth`、checkpoint、实验结果图和日志默认视为可再生产物，不应随普通代码修改一起提交。
- 小型配置、实验协议、结果摘要 Markdown/CSV 可以保留，但必须能说明生成方式。
- 新增数据处理逻辑必须写明输出数组 key、shape、标签含义和是否使用训练/测试统计量。
- 训练日志统一输出到 `logs/training/<run_id>/`。`run_id` 建议使用 `YYYYMMDD-HHMMSS_<experiment_name>`。
- 每个训练日志目录至少保留配置快照、stdout/stderr 日志、指标 CSV/JSON、checkpoint 路径和评估摘要。

## 7. 验证要求

- 改模型模块时，至少做 shape/接口检查：输入 `[B, 4, 14] complex`，输出 `[B, 2, 14]`。
- 改 registry 时，检查已知类型可构建、未知类型会抛出清晰错误。
- 改评估逻辑时，检查 Pfa 阈值排序、判决方向 `o0 <= threshold -> target`、以及实际 `PF`。
- 无法运行完整训练时，要报告跳过原因，并尽量运行小样本或单 batch 验证。
