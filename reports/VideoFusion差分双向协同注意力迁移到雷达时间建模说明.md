# VideoFusion差分双向协同注意力迁移到雷达时间建模说明

本文说明如何参考 VideoFusion 官方仓库，把多模态视频融合中的差分增强与双向时序协同注意力，迁移为本项目 ST-GNN 主干中的 TFE 替换模块。

## 1. 原任务为什么这样处理

VideoFusion 面向红外-可见光视频融合。官方仓库输入是：

```text
x_ir: [B, T, C, H, W]
x_vi: [B, T, C, H, W]
```

其中 `T` 是视频帧，`H/W` 是图像空间维度，红外和可见光是两个互补模态。

视频融合面对两个问题：

1. 单帧图像融合容易忽略跨帧时间一致性，输出视频会闪烁。
2. 红外和可见光存在互补信息，也存在互相干扰，需要突出跨模态差异中的有用部分。

因此官方实现中有两个关键代码思路。

### 1.1 跨模态差分增强

官方 `CrossModalityDiffEnhance` 的核心逻辑可以概括为：

```python
diff = supplementary_feature - main_feature
enhanced = cross_attention(main_feature, diff)
```

原任务含义：

```text
main_feature         当前模态特征，例如可见光
supplementary_feature补充模态特征，例如红外
diff                 红外与可见光之间的互补差异
enhanced             用差异信息增强当前模态
```

它解决的是多模态融合中的“互补信息如何注入”问题。

### 1.2 前后帧协同注意力

官方 `CoAttentionModule` 对每个视频帧取：

```text
prev_frame, curr_frame, next_frame
```

边界帧用自身补齐：

```text
i=0:     prev = curr, next = frame_1
i=T-1:   prev = frame_{T-2}, next = curr
middle:  prev = frame_{i-1}, next = frame_{i+1}
```

官方 `Co_Attention` 的核心结构可以概括为：

```python
attn_prev = attention(q_curr, k_prev)
attn_next = attention(q_curr, k_next)
attn_co = softmax(attn_prev * attn_next)
out = attn_co @ v_prev + attn_co @ v_next
```

原任务含义：

```text
当前帧的表示不能只看当前帧；
需要同时被前一帧和后一帧支持；
前后注意力相乘相当于协同确认，减少单侧帧噪声造成的不一致。
```

## 2. 为什么能迁移到雷达 TFE

本项目的 ST-GNN 时间模块输入不是视频，而是 SFE 后的雷达慢时间特征：

```text
H: [B, C, P, N]
```

其中：

```text
B batch
C 特征通道
P 慢时间 pulse / observation window
N range cell
```

论文原始 TFE 也是沿 `P` 维建模：

```text
Conv2d(kernel=(3,1), stride=(2,1), padding=(1,0))
```

这说明原论文本身已经把 `P` 维当作短时序。VideoFusion 的视频帧 `T` 可以对应到雷达慢时间 pulse `P`：

```text
视频帧 T       -> 雷达 pulse P
当前帧         -> 当前 pulse
前一帧/后一帧  -> 前一 pulse/后一 pulse
跨帧不一致     -> 慢时间目标扰动或海杂波尖峰
```

## 3. 维度如何修改

VideoFusion 官方 co-attention 处理的是：

```text
[B, T, C, H, W]
```

进入 attention 前通常会取单帧：

```text
curr_frame: [B, C, H, W]
prev_frame: [B, C, H, W]
next_frame: [B, C, H, W]
```

本项目 TFE 输入是：

```text
[B, C, P, N]
```

迁移时把每个 pulse 当作一帧，把 range cell 维 `N` 当成一维空间位置：

```text
curr_pulse: [B, C, N]
prev_pulse: [B, C, N]
next_pulse: [B, C, N]
```

为了保留 PyTorch `Conv2d` 和原 ST-GNN TFE 的习惯，实际代码不显式循环 pulse，而是在 `[B, C, P, N]` 上向前/向后平移：

```python
prev = cat([H[:, :, :1, :], H[:, :, :-1, :]], dim=2)
next = cat([H[:, :, 1:, :], H[:, :, -1:, :]], dim=2)
```

这样得到的 `prev/curr/next` 仍然是：

```text
[B, C, P, N]
```

后续 attention 内部再 reshape 为：

```text
[B*P, heads, C/heads, N]
```

含义是：每个 pulse 独立做前后上下文协同确认，attention 不直接跨 `N` 做全局 token 混合，避免 TFE 抢走 SFE 的空间建模职责。

## 4. 差分增强如何改写

原 VideoFusion：

```text
diff = infrared_feature - visible_feature
```

本项目改成慢时间差分：

```text
D_prev = H_t - H_{t-1}
D_next = H_{t+1} - H_t
```

代码对应：

```python
diff = cat([H - prev, next - H], dim=1)
diff_feature = tanh(Conv1x1(diff))
gate = sigmoid(Conv1x1(cat([H, diff_feature], dim=1)))
H_diff = H + gate * diff_feature
```

雷达含义：

```text
海杂波背景在短慢时间内通常具有一定连续性；
小目标会造成幅度、相位或高层特征的局部扰动；
差分增强用于突出这种慢时间扰动；
门控用于抑制差分对海杂波尖峰的过度放大。
```

## 5. 双向协同注意力如何改写

原 VideoFusion：

```text
q = curr_frame
k/v = prev_frame 和 next_frame
attn_co = softmax(attn_prev * attn_next)
```

本项目：

```text
q = 当前 pulse 特征
k/v_prev = 前一 pulse 特征
k/v_next = 后一 pulse 特征
```

代码思想：

```python
attn_prev = softmax(q @ k_prev.T)
attn_next = softmax(q @ k_next.T)
attn_co = softmax(attn_prev * attn_next)
co = attn_co @ v_prev + attn_co @ v_next
```

雷达含义：

```text
真实目标的慢时间变化应在前后 pulse 中有一定一致性；
单个 pulse 的海杂波尖峰可能只被一侧上下文支持；
前后注意力相乘相当于让前后上下文共同确认当前扰动。
```

## 6. 为什么仍保留 stride=2 时间压缩

原 ST-GNN TFE 的重要结构语义是：

```text
TFE1: P -> ceil(P/2)
TFE2: ceil(P/2) -> ceil(P/4)
```

默认 `P=4` 时：

```text
4 -> 2 -> 1
```

因此迁移模块最后仍使用：

```python
Conv2d(kernel=(3,1), stride=(2,1), padding=(1,0))
```

这保证新模块是 TFE 替换，而不是另起一条时间建模流水线。

## 7. 当前实现位置

新增代码：

```text
paper_modules/models/modules/temporal_modules/diff_bicam_tfe.py
paper_modules/models/modules/temporal_modules/stgnn_tfe.py
paper_modules/models/modules/temporal_modules/__init__.py
paper_modules/models/modules/temporal.py
```

配置：

```text
paper_modules/configs/real_imag_tfe_replacement_diff_bicam.yaml
```

配置中固定：

```text
radar_features.type = real_imag
spatial_graph.type = original_stfe
temporal.type = diff_bicam_tfe
```

这样实验只比较 TFE 替换，不混入输入特征替换或空间图替换。

## 8. 与官方代码的对应关系

| VideoFusion 官方思路 | 官方任务含义 | 本项目改写 | 雷达任务含义 |
|---|---|---|---|
| `supplementary - main` | 红外/可见光互补差异 | `H_t-H_{t-1}`, `H_{t+1}-H_t` | 慢时间扰动 |
| `curr/prev/next frame` | 视频前后帧上下文 | 当前/前一/后一 pulse | 前后慢时间上下文 |
| `attn_prev * attn_next` | 前后帧协同确认 | 前后 pulse 协同确认 | 抑制单点海杂波尖峰 |
| 2D frame `[C,H,W]` | 图像空间 | 1D range `[C,N]` | range cell 上的时序特征 |
| 视频融合输出 | 重建融合视频 | stride=2 TFE 输出 | 保持 ST-GNN 检测主干 |

## 9. 后续消融建议

为证明迁移不是简单堆模块，至少需要：

```text
original_tfe
diff_only
bicam_no_diff
forward_only
backward_only
no_coattention
shuffle_diff
same_P_original_tfe
```

其中 `shuffle_diff` 已在代码中保留开关，用于破坏差分与当前 pulse 的时间对应关系，检验差分收益是否来自真实慢时间结构。

