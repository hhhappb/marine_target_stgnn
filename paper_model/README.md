# Paper Model Code

这个目录只维护论文 ST-GNN 模型主体，方便后续替换模块做对比实验。

当前文件：

- `st_gnn.py`: 从原项目 `models/st_gnn.py` 单独抽出的论文模型主干。
- `__init__.py`: 暴露模型类，便于 `from paper_model import STGNNDetector`。

主要可替换模块：

- `STFE`: 空间特征提取模块，内部使用 `GraphAttentionLayer`。
- `TFE`: 时间特征提取模块，当前是门控时序卷积。
- `Detector`: 检测头，当前是两层 `Conv1d`。
- `FARController`: 测试阶段虚警率控制模块。

建议先只替换一个模块，保持输入/输出 shape 不变：

- `STFE.forward(x_list)` 输入若干个 `[B, C, N]`，输出 `[B, C_out, P, N]`。
- `TFE.forward(x)` 输入 `[B, C, P, N]`，输出 `[B, C_out, P_next, N]`。
- `Detector.forward(x)` 输入 `[B, 1024, N]`，输出 `[B, 2, N]`。
