# 当前实验协议索引

核对日期为2026年9月28日。文档记录实际实验设置，不表示严格恢复了原论文所有未公开参数。

| 数据集 | 文档 | 当前运行概况 |
|---|---|---|
| IPIX | [IPIX当前实验协议](ipix_experiment_protocol.md) | P=8、70轮、batch64、56条件Original/E4配对，真实目标，无SCR注入 |
| SDRDSP | [SDRDSP实验协议](sdrdsp_experiment_protocol.md) | v1.1采用Σ/(P×20)参考功率，区分P=4/P=8及20/100/120轮预算 |

[IPIX PAX-L1](ipix_pax_l1_protocol.md)保留P=16定义，不替代P=8运行说明。SDRDSP现用SCR与论文式(17)参考单元求和定义存在约13.01 dB差异。

比较模型必须统一数据、标签、预算、评分与阈值来源。文档中的本地证据路径仅供追溯，不表示数据、权重、日志或完整E4代码已经上传。本次发布不启动新实验或授权新增seed。
