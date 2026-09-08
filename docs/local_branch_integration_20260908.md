# 本地分支集成清单（2026-09-08）

## 目标

以 `main@18a77f2` 为稳定基线，在 `codex/integration-mainline-20260908` 中保留当前仍有效的
通用能力。只迁移可复用源码、必要测试和最小复现入口，不整体合并历史实验分支。

## 已集成

### `codex/ipix-position-augmentation`

已迁移的通用能力：

- `clutter_fill` 非循环位置增强；
- `train_sources` / `test_sources` 跨文件路由；
- 训练与测试 source 不相交校验；
- `train.py` 运行元数据记录；
- `auto_experiment.py` 跨文件实验单元识别与可比性校验；
- 精简的 PAX 核心单元测试。

集成提交：`7bdc1b6 feat(ipix): integrate PAX cross-file protocol`。

明确未迁移：

- `stage_aware_*`、`pairwise_content_fusion_tfe` 等历史时间候选；
- PAX-L1 的批量生成配置和历史绘图脚本；
- 分支内的 `AGENTS.md` 版本；
- 数据、checkpoint 和训练日志。

## 无需同步

### `codex/sdrdsp-strict-preprocess`

相对当前集成分支没有独有提交。其严格预处理、lag-aware 时间模块、DiffBiCAM 退役和
scale-normalized 时间模块等有效历史均已包含在 main 的提交祖先中。

## 暂不纳入

### `codex/shortcut-diagnostics`

仅有独有提交 `1f19051 Add IPIX shortcut diagnostic logging`。该提交同时修改
`auto_experiment.py`、`train.py` 和 `leakage_probe.py`，产生于现行 train-only 阈值规则之前。
除非重新按当前协议审核，否则不进入集成分支。

### `origin/e4d2-gdcm-v2-ipix`

这是独立的远端 `e4d2_gdcm/` 包，不是本地实验分支。当前工作区存在未跟踪副本，但尚未验证
其与远端提交是否一致，也未验证依赖和测试，因此留作独立审核工作单元。

## 验证状态

- PAX 核心专项测试：7 passed。
- PAX、阈值策略、训练批处理和 ModularSTGNN 相关回归：20 passed，1 deselected。
- 被排除测试是 main 基线已存在的不一致：`tests/test_threshold_source_policy.py` 要求
  `scripts/extract_ipix_report.py` 已删除，但该删除尚未进入 main 提交。
- 原 main 工作区的未提交内容未被修改、暂存或提交。

## 后续分支

时间建模研发从本集成版本派生为 `codex/temporal-multilag-p16`，主协议固定为 P=16。
在设计冻结前不加入新的时间模块实现。
