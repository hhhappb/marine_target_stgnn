# SDRDSP 实验协议

本文件定义项目自2026年9月22日起唯一有效的SDRDSP正式实验协议。此前的2400窗快筛、10 epochs、P=16全量训练和logit margin结果仅保留为历史诊断，不进入新实验的模型排序或结论。

## 数据和目标构造

- 协议ID：sdrdsp_fig9_v11_p4_n256_seed42。
- 训练背景：20210106155330_01_staring.mat，裁剪为6940脉冲乘256距离单元。
- 测试背景：20210106155432_01_staring.mat，裁剪为6520脉冲乘256距离单元。
- 裁剪中心对应论文第2083距离单元，窗内下标为128。
- 每个样本包含4个连续脉冲，步长为4。
- 按协议文档舍弃最后一个仍可完整取出的窗口，训练每SCR为1734窗，测试每SCR为1629窗。
- 训练SCR为-12至14 dB，间隔2 dB；测试SCR为-24至14 dB，间隔2 dB。
- 每个训练窗注入5个目标，距离间隔至少10格；测试目标固定在下标128。
- 训练速度从0.1至0.5 m/s确定性抽取；测试速度为0.4 m/s。
- 每个目标使用20个随机非目标参考单元，以窗口内脉冲和参考单元的平均功率定义杂波功率。
- 每个目标每个窗口独立抽取初相，初相随机流固定为RandomState(777)。
- 训练背景复幅度的99分位数记为P99；所有深度模型输入统一使用I/P99和Q/P99，测试沿用训练P99。

## 训练

- 当前正式运行只使用seed42。
- 时间建模候选统一使用20 epochs快速筛选，batch size为24。
- 优化器为Adam，固定学习率0.001，不使用学习率调度或权重衰减。
- 使用普通交叉熵，不使用类别权重。
- 批次按14个训练SCR近似均衡构造；由于24不能被14整除，每批各SCR样本数相差不超过1，额外名额按批次轮换。
- 启用确定性训练。
- checkpoint按最低训练损失选择，测试集不参与选择。
- 20 epochs结果用于判断候选方向是否具有正向信号；候选与Original必须在同一20 epochs预算下重新配对，不得与历史100 epochs结果直接比较。

## 阈值和评价

- 正式评分使用softmax通道0的杂波概率o0。
- 阈值只使用训练集全部杂波距离单元确定。
- 将训练杂波o0升序排列，索引为ceil(Pfa乘Nc)-1。
- o0不大于threshold时判为目标，否则判为杂波。
- 正式Pfa为0.0001、0.001和0.01。
- 主结果报告逐SCR测试PD；测试实际PF仅作为诊断，不参与模型排序。
- 当前单seed结果只描述seed42下的配对差异，不作随机性显著性声明。

## 入口

生成数据：

    .\.venv\Scripts\python.exe scripts\preprocess_sdrdsp_v11.py

原始MAT默认从只读参考目录E:/stgnn/data读取；生成器只向新的data/sdrdsp_v11_p4_n256_seed42目录写入。

Original基线：

    .\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\sdrdsp_v11_original_seed42.yaml

SNDD双级时间模块：

    .\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\sdrdsp_v11_sndd_both_seed42.yaml
