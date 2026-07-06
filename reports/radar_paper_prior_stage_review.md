# new paper 文件夹雷达论文先验使用阶段整理

## 1. 目的

根据 `C:\Users\Administrator\Desktop\new paper` 文件夹中的论文，筛选出与雷达海上小目标检测、海杂波抑制、HFSWR、雷达时空建模相关的论文，并判断其中使用的雷达领域知识属于：

1. 预处理/特征构造阶段；
2. 模型结构/训练阶段；
3. 两者都有。

这里的“雷达先验”指来自雷达信号机理或传统检测经验的知识，例如 Doppler、相位、距离单元、距离-多普勒图、时频谱、CFAR/Pfa、海杂波统计、极化、时空相关、参考窗/保护窗等。

## 2. 结论概览

从当前文件夹看，雷达领域论文大致分成三类：

1. **预处理/特征构造型**：先把雷达回波转换成时频图、距离-多普勒图、序列特征、人工统计特征，再送入 CNN/Bi-LSTM/分类器。
2. **模型结构型**：尽量少做人工特征，在模型内部用 GNN、注意力、ConvGRU、自学习、复杂值网络等结构学习雷达时空关系。
3. **混合型**：先做雷达物理变换或特征构造，再用专门模型融合，例如 RD 图 + GNN、距离像 + Doppler 双分支、序列特征 + PLM。

对当前课题最有参考价值的是第 2 和第 3 类，因为老师的任务是研究“雷达先验如何进入时空图建模”，不只是做独立预处理。

## 3. 论文逐篇分类

| 论文 | 是否雷达相关 | 使用的雷达知识 | 进入阶段 | 对当前工作的启发 |
|---|---:|---|---|---|
| `Marine_Target_Detection_via_SpatialTemporal_Graph_Neural_Network.pdf` | 是 | 距离单元、少脉冲 range profile、时空图、FAR/Pfa 阈值 | 主要在模型阶段 | 论文明确强调无数据预处理，直接用 I/Q，经 CNN、空间图、GAT、ConvGRU 提取时空特征，是当前 baseline 对照。 |
| `Radar_Maritime_Target_Detection_via_SpatialTemporal_Feature_Attention_Graph_Convolutional_Network.pdf` | 是 | 空间-时间特征、图卷积、注意力、检测阈值/Pfa | 主要在模型阶段 | 可参考其“图 + 注意力”如何表达时空特征，适合支持空间图改进方向。 |
| `A_Sea_Clutter_Suppression_Method_With_Graph_Neural_Network_for_Maritime_Target_Detection.pdf` | 是 | range-Doppler 谱、时空图、图神经网络、CA-CFAR/Pfa | 预处理 + 模型阶段 | 先把序列 RD 图切成 patches 构图，再用 GNN 抑制海杂波；适合参考“RD 先验如何变成图结构”。 |
| `DFF Sequential Dual-Branch Feature Fusion for Maritime Radar Object Detection and Tracking via Video Processing.pdf` | 是 | 复数雷达回波、距离像、距离-Doppler谱、相位-幅度关系、连续脉冲 | 预处理 + 模型阶段 | 双分支复杂值 U-Net 分别处理距离 profile 和 Doppler spectrogram，再做序列融合；适合参考多模态输入。 |
| `RadarPLM.pdf` | 是 | instantaneous phase、Doppler spectrum entropy、STFT marginal spectrum、amplitude、Doppler peak 等序列特征 | 主要在预处理/特征构造阶段，模型阶段用 PLM 选择性微调 | 典型“雷达手工序列特征 -> 模型 token”的路线，可为 PL/RPH、cell stats 输入提供依据。 |
| `Sequence-Feature_Detection_of_Small_Targets_in_Sea_Clutter_Based_on_Bi-LSTM.pdf` | 是 | 瞬时相位 IP、Doppler spectrum entropy、STFT marginal spectrum | 主要在预处理/特征构造阶段 | 先提取三种序列特征，再用 Bi-LSTM 分类；证明相位、Doppler、STFT 边缘谱是有效雷达先验。 |
| `Small-floating Target Detection in Sea Clutter via.pdf` | 是 | Time-Doppler spectra、Doppler shift、I/Q 幅相、纹理特征 | 主要在预处理/特征构造阶段 | 将回波转成 time-Doppler 图，再提视觉/纹理特征；属于雷达变换后图像分类路线。 |
| `A_False_Alarm_Controllable_Detection_Method_Based_on_CNN_for_Sea-Surface_Small_Targets.pdf` | 是 | STFT time-frequency features、PFA control、IPIX 多极化数据 | 预处理 + 模型阶段 | 先用 STFT 转时频特征图，再 CNN + PFA 控制单元；适合参考“时频先验 + 可控虚警”。 |
| `Enhanced_CNN-Based_Small_Target_Detection_in_Sea_Clutter_With_Controllable_False_Alarm.pdf` | 是 | Wigner-Ville distribution 时频图、PFA control、海杂波统计困难 | 预处理 + 模型阶段 | WVD 是明确的预处理阶段雷达先验，PFA 控制单元是模型/决策阶段雷达约束。 |
| `Maritime_Radar_Target_Detection_in_Sea_Clutter_Based_on_CNN_With_Dual-Perspective_Attention.pdf` | 是 | 海杂波、CFAR/Pfa、global/local correlation of radar echo | 主要在模型阶段 | 未强调人工时频预处理，核心是 CNN + 双视角注意力提取全局/局部相关特征。 |
| `Overcoming_Data_Scarcity_in_Maritime_Radar_Target_Detection_via_a_Complex-Valued_Hybrid_Spatiotemporal_Network.pdf` | 是 | IPIX、14 range cells、STFT/TF representation、复数相位、微多普勒、时空特征 | 预处理 + 模型阶段 | 用 TF 表示作为输入，同时用复杂值 CNN/BiLSTM 建模相位和时空结构；与当前 IPIX `N=14` 很相关。 |
| `Sea_Clutter_Suppression_Method_Based_on_a_Self-Attention_Spatio-Temporal_Prediction_Network.pdf` | 是 | 海杂波预测、相邻距离单元时空相关、I/Q 相位、海杂波非线性非平稳 | 预处理 + 模型阶段 | 有 sea clutter preprocessing module，也用 self-attention 时空预测；适合参考“先预测/抑制杂波再检测”。 |
| `Intelligent_Maritime_Radar_Target_Detection_in_Unknown_Environment_via_Self-Learning.pdf` | 是 | 未知环境、Pfa、domain/environment shift、range-Doppler/time-frequency/polarization 特征背景 | 主要在模型/训练阶段 | 雷达知识主要作为环境自学习和检测器适配约束，不是新增预处理特征。 |
| `Adaptive_Intelligent_Radar_Target_Detection_in_Time-Varying_Sea_Clutter_via_Activate_Self-Learning.pdf` | 是 | 时变海杂波、Pfa、环境漂移、主动自学习 | 主要在模型/训练阶段 | 重点是时间变化环境下的自学习适配，可作为后续“跨环境鲁棒性”参考。 |
| `A_Joint_Detection_and_Tracking_Paradigm_Based_on_Reinforcement_Learning_for_Compact_HFSWR.pdf` | 是 | compact HFSWR、检测阈值、跟踪预测、密集杂波、多目标 | 主要在模型/决策阶段 | 跟踪器反向指导检测阈值，属于检测-跟踪闭环先验，不是纯预处理。 |

## 4. 按阶段归纳

### 4.1 先验主要在预处理/特征构造阶段使用

这些论文的共同特点是：先根据雷达机理把原始回波转换成更有判别力的表达，再交给通用模型。

代表论文：

- `Sequence-Feature_Detection_of_Small_Targets_in_Sea_Clutter_Based_on_Bi-LSTM.pdf`
- `Small-floating Target Detection in Sea Clutter via.pdf`
- `RadarPLM.pdf`
- `A_False_Alarm_Controllable_Detection_Method_Based_on_CNN_for_Sea-Surface_Small_Targets.pdf`
- `Enhanced_CNN-Based_Small_Target_Detection_in_Sea_Clutter_With_Controllable_False_Alarm.pdf`

典型雷达先验：

- STFT 时频图；
- WVD 时频图；
- Time-Doppler spectra；
- instantaneous phase；
- Doppler spectrum entropy；
- STFT marginal spectrum；
- Doppler peak；
- amplitude sequence；
- texture/visual features on time-Doppler images。

对当前工作的意义：

这些方法证明了相位、Doppler、时频能量、幅度序列确实有判别力。当前不需要重复证明这些先验是否存在价值，而应考虑如何把它们轻量化地接入 ST-GNN，例如 PL/RPH、RD/FFT、delta phase、cell-wise stats。

### 4.2 先验主要在模型阶段使用

这些论文不一定做复杂人工预处理，而是在模型结构中体现雷达时空规律、图结构、注意力或环境适配。

代表论文：

- `Marine_Target_Detection_via_SpatialTemporal_Graph_Neural_Network.pdf`
- `Radar_Maritime_Target_Detection_via_SpatialTemporal_Feature_Attention_Graph_Convolutional_Network.pdf`
- `Maritime_Radar_Target_Detection_in_Sea_Clutter_Based_on_CNN_With_Dual-Perspective_Attention.pdf`
- `Intelligent_Maritime_Radar_Target_Detection_in_Unknown_Environment_via_Self-Learning.pdf`
- `Adaptive_Intelligent_Radar_Target_Detection_in_Time-Varying_Sea_Clutter_via_Activate_Self-Learning.pdf`
- `A_Joint_Detection_and_Tracking_Paradigm_Based_on_Reinforcement_Learning_for_Compact_HFSWR.pdf`

典型雷达先验：

- 距离单元构图；
- 时空相关建模；
- 全局/局部相关注意力；
- Pfa/FAR 阈值约束；
- 环境漂移/时变海杂波自适应；
- 跟踪预测反向指导检测。

对当前工作的意义：

这一类最贴合老师要求的“雷达先验如何进入时空图建模”。当前可以从固定相邻图升级为 CFAR参考窗图、相似度图、PL/RPH先验图，或者把 Pfa/FAR 作为训练和决策约束。

### 4.3 预处理和模型阶段同时使用

这些论文通常先做雷达物理变换，再在模型中做专门融合。

代表论文：

- `A_Sea_Clutter_Suppression_Method_With_Graph_Neural_Network_for_Maritime_Target_Detection.pdf`
- `DFF Sequential Dual-Branch Feature Fusion for Maritime Radar Object Detection and Tracking via Video Processing.pdf`
- `Overcoming_Data_Scarcity_in_Maritime_Radar_Target_Detection_via_a_Complex-Valued_Hybrid_Spatiotemporal_Network.pdf`
- `Sea_Clutter_Suppression_Method_Based_on_a_Self-Attention_Spatio-Temporal_Prediction_Network.pdf`

典型路线：

```text
原始复数回波
  -> 距离像 / RD图 / TF图 / patch graph / sea clutter preprocessing
  -> GNN / complex-valued network / dual branch / self-attention prediction
```

对当前工作的意义：

这类论文说明“雷达先验不一定只在预处理阶段，也可以转化为模型输入和图结构”。这正好支持当前主线：把前期有效预处理经验上升为模型内先验。

## 5. 与当前课题最相关的可复用结论

### 5.1 可以作为预处理/输入先验复用

1. 相位相关先验：instantaneous phase、phase continuity、phase linearity。
2. Doppler 相关先验：Doppler spectrum、Doppler entropy、RPH、RD/TF maps。
3. 幅度相关先验：amplitude sequence、relative amplitude、local statistics。
4. 时频相关先验：STFT、WVD、time-Doppler spectra。
5. 极化相关先验：IPIX 的 `HH/HV/VH/VV` 多极化响应差异。

### 5.2 可以作为模型结构先验复用

1. range cell 作为图节点；
2. 相邻距离单元或参考窗作为图边；
3. RD/TF patch 构图；
4. 空间-时间联合图；
5. global/local attention；
6. self-learning/domain adaptation；
7. FAR/Pfa 可控决策。

### 5.3 不建议当前作为主线的内容

1. 单纯把 STFT/WVD 图喂给 CNN：这是前处理型路线，离“时空图建模”主线较远。
2. 大型 PLM 或复杂预训练模型：可以参考特征 token 化思想，但实现成本和主题匹配度不如 PL/RPH + GNN。
3. 检测-跟踪强化学习闭环：适合 HFSWR 跟踪场景，不是当前 IPIX/ST-GNN 主线。
4. 只做 sea clutter suppression 前置模块：可以作为补充，但不如“先验图 + 时间先验输入”直接。

## 6. 对当前实验顺序的建议

建议按以下顺序开展：

```text
Baseline:
  当前 IPIX Raw I/Q + ST-GNN

Experiment 1:
  Raw I/Q + PL/RPH

Experiment 2:
  Raw I/Q + cell-wise statistics

Experiment 3:
  Raw I/Q + PL/RPH + cell-wise statistics

Experiment 4:
  在空间图中加入雷达先验边：
    - 距离邻接
    - CFAR参考窗
    - PL/RPH相似
    - 幅度/相位相似

Experiment 5:
  加入 FAR/Pfa 约束或 H0 Pfa 验证
```

这个顺序的好处是：

1. 从前期最稳定、最可解释的 PL/RPH 开始；
2. 再引入 cell-wise stats，验证统计先验；
3. 最后把先验从“输入特征”推进到“空间图结构”；
4. 能清楚回答老师的问题：雷达先验到底是在预处理阶段使用，还是进入了时空图模型内部。

## 7. 非雷达主线论文

以下文件与海上路径规划、AUV/USV、红外小目标或水下目标跟踪更相关，不作为当前“雷达先验进入 ST-GNN”主线分析对象：

- `A Multi-AUV Maritime Target Search Method for Moving.pdf`
- `Deep Reinforcement Learning-based Navigation of Unmanned Surface Vessel in .pdf`
- `High-Speed Underwater Object Tracking Using Deep .pdf`
- `Spatio-Temporal Reinforcement Learning-Driven Ship Path.pdf`
- `Spatio-Temporal Context Learning with Temporal Difference Convolution for.pdf`

它们可能对“时空建模”有启发，但不是雷达信号处理先验，不能直接作为雷达先验依据。
