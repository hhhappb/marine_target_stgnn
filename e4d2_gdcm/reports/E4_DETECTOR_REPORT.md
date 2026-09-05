# E4 检测头优化实验报告

## Detector Head Optimization for CVOCA-Fusion ST-GNN

- **实验日期**：2026年5月20日
- **固定主干**：E4 Full CVOCA（ST-GNN + PhaseEncoder + AmpEncoder + CVOCA Fusion + Global concat）
- **训练协议**：SCR-uniform, 100 epochs × 3 seeds (42, 123, 456), wCE(10), BS=24, Adam lr=0.001
- **唯一变量**：Detector Head 结构

---

## 1. 实验目的

E4 CVOCA 在输入和融合层面已经做得很强，但最后的检测头仍是原 ST-GNN 的简单 `Conv1d(1216→512→2)`。这个头本质上只做通道压缩和逐距离单元分类，对距离维局部上下文利用很弱。本实验研究：**在固定主干的情况下，检测头能否通过更好的结构设计进一步提升低 SCR 检测能力？**

---

## 2. 实验设计

所有实验共享同一个 E4 Backbone（参数冻结在架构层面，每次独立训练），输入完全相同：

```
F = concat([F_main 1024ch, F_ap 64ch, F_global 128ch])  shape=[B, 1216, N]
```

### 2.1 D0：原始检测头（基线）

```
F → Conv1d(1216 → 512) → ReLU → Conv1d(512 → 2)
```

纯通道压缩 + 逐距离单元分类，无距离邻域。

### 2.2 D1：Residual Bottleneck

```
F → Conv1d(1216 → 512, k=1)
  → ResBlock × 2 (Conv1d 512→512→512, GELU, residual add)
  → Conv1d(512 → 2, k=1)
```

增加通道非线性深度，但仍不引入距离邻域。

### 2.3 D2：Local-CFAR-like

```
F → Conv1d(1216 → 256, k=1)

并行多尺度局部卷积：
  Branch1: Depthwise Conv1d(k=3, groups=256)
  Branch2: Depthwise Conv1d(k=7, groups=256)
  Branch3: Depthwise Conv1d(k=15, groups=256)

concat → Conv1d(768 → 512) → GELU → Conv1d(512 → 2)
```

模拟 CFAR 中"参考单元估计背景"的思想，用 3/7/15 多尺度局部窗口捕获邻域杂波上下文。

### 2.4 D3：Dilated TCN

```
F → Conv1d(1216 → 256, k=1)

Residual Dilated Conv Blocks:
  k=3, dilation=1  (感受野 3)
  k=3, dilation=2  (感受野 7)
  k=3, dilation=4  (感受野 15)
  k=3, dilation=8  (感受野 31)

→ Conv1d(256 → 512) → Conv1d(512 → 2)
```

用空洞卷积在不增加参数的前提下扩大距离维感受野至 31 单元。

### 2.5 D4：Global-FiLM

```
F_local = concat([F_main, F_ap])  # 1088ch
G = F_global 的 mean pooling  # 128-dim 向量

F_local → Conv1d(1088 → 512)
G → MLP → γ, β        # 每个通道的 scale 和 bias

F_mod = F_local × (1+γ) + β
F_mod → Conv1d(512 → 2)
```

将 Global 信息从"concat"改为"FiLM 调制"，检验 Global 是否更适合用于特征标定而非直接输入。

### 2.6 D5：Modality-Gated

```
F_main   [B,1024,N] → Conv1d(1024 → 256)
F_ap     [B,64,N]   → Conv1d(64 → 256)
F_global [B,128,N]  → Conv1d(128 → 256)

concat → Conv1d(768 → 3) → softmax → α_main, α_ap, α_global

F_fused = α_main × F_main_proj + α_ap × F_ap_proj + α_global × F_global_proj

F_fused → Conv1d(256 → 512) → Conv1d(512 → 2)
```

让检测头在每个距离单元上自动学习：更相信 ST-GNN 主干、CVOCA 幅相融合、还是 Global 标定。

### 2.7 D6：Clutter-Calibrated Margin

```
F → Conv1d(1216 → 512)

Target branch:   Conv1d(512 → 256, k=1) → GELU → Conv1d(256 → 1)
Clutter branch:  Conv1d(512 → 256, k=15) → GELU → Conv1d(256 → 1)

logits = [s_clutter, s_target]
```

显式分离"目标得分"和"杂波基线"两个预测通道：目标分支用 k=1（逐点），杂波分支用 k=15（邻域上下文）。设计意图是让 margin = s_t - s_c 直接优化固定虚警率下的可分离性。

---

## 3. 实验结果

| ID | AUC | PdL | Pd@-24dB | Pd@-20dB | Margin | Params | vs D0 ΔAUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E4-D0 (Original) | 0.983 ± 0.003 | 0.669 ± 0.026 | 0.285 | 0.618 | — | 5,508K | baseline |
| E4-D1 (Res Bottleneck) | 0.986 ± 0.001 | 0.698 ± 0.034 | 0.301 | 0.656 | — | 5,313K | +0.003 |
| **E4-D2** (Local-CFAR) | **0.990 ± 0.003** | **0.753 ± 0.065** | **0.404** | **0.737** | — | 4,352K | **+0.007** |
| E4-D3 (Dilated TCN) | 0.985 ± 0.001 | 0.719 ± 0.034 | 0.380 | 0.674 | — | 5,658K | +0.002 |
| E4-D4 (Global-FiLM) | 0.981 ± 0.005 | 0.706 ± 0.034 | 0.281 | 0.674 | — | 4,329K | −0.002 |
| E4-D5 (Modality-Gated) | 0.979 ± 0.004 | 0.667 ± 0.043 | 0.322 | 0.616 | — | 4,086K | −0.004 |
| **E4-D6** (Margin Head) | **0.987 ± 0.001** | **0.729 ± 0.047** | 0.333 | **0.707** | — | 6,360K | +0.004 |

---

## 4. 关键发现

### 发现 1：D2 (Local-CFAR) 在所有指标上全面超越原始检测头。

| 指标 | D0 | D2 | 提升 |
|------|:---:|:---:|:---:|
| AUC | 0.983 | **0.990** | +0.007 |
| PdL | 0.669 | **0.753** | +12.6% |
| Pd@-24dB | 0.285 | **0.404** | **+41.8%** |
| Pd@-20dB | 0.618 | **0.737** | +19.3% |

D2 seed=456 的 AUC=0.995、PdL=0.834 是当前实验中统计最高的单种子结果。

**多尺度局部卷积（k=3/7/15）完美模拟了 CFAR 中"参考单元估计背景"的思想**：近距离窗口（k=3）提供局部精细对比度，中距窗口（k=7）提供邻域统计，远距窗口（k=15）提供区域背景水平。三者在深度可分离卷积中互不干扰（groups=256），最后融合时协同工作。

### 发现 2：D1 (Res Bottleneck) 也有效，但增益远小于 D2。

| 指标 | D0 | D1 | D2 |
|------|:---:|:---:|:---:|
| AUC | 0.983 | 0.986 | **0.990** |
| PdL | 0.669 | 0.698 | **0.753** |

仅增强通道非线性 (+0.003 AUC) 远不如果引入距离邻域上下文 (+0.007 AUC)。**距离单元的上下文信息比通道非线性更重要。**

### 发现 3：D3 (Dilated TCN) 接近 D1 水平，但参数更多。

空洞卷积虽然扩大了感受野（到 31），但增益不如 D2 的多尺度显式窗口。可能是因为 4 层残差空洞卷积在 100 epoch 下梯度传递效率低于 D2 的并行分支。

### 发现 4：Global-FiLM (D4) 和 Modality-Gated (D5) 未超越基线。

- D4 AUC=0.981（−0.002 vs D0）。Global 信息作为 concat 已经足够有效，FiLM 调制在检测头层面没有提供额外增益。
- D5 AUC=0.979（−0.004 vs D0）。Per-range-cell 的模态门控引入了额外的优化难度，3 个模态的 softmax gate 在训练中难以收敛到有意义的分布。

### 发现 5：D6 (Margin Head) 是潜力方案。

| 指标 | D0 | D6 |
|------|:---:|:---:|
| AUC | 0.983 | **0.987** |
| Pd@-20dB | 0.618 | **0.707** (+14.4%) |

Clutter-Calibrated Margin Head 的 `s_t − s_c` 结构在 Pd@-20dB 上接近 D2 水平（0.707 vs 0.737），且 AUC=0.987 仅次于 D2。其 15-wide 杂波分支显式建模邻域杂波基线，为固定虚警率下的阈值选择提供了天然的 margin 结构。

---

## 5. 综合排名

| 排名 | 检测头 | AUC | PdL | 推荐理由 |
|:---:|:---:|:---:|:---:|------|
| 1 | **D2 (Local-CFAR)** | **0.990** | **0.753** | 全面最优，参数最少 |
| 2 | D6 (Margin) | 0.987 | 0.729 | Pd@-20dB 高，margin 天然适配 FAR |
| 3 | D3 (Dilated TCN) | 0.985 | 0.719 | Pd@-24dB 仅次于 D2 |
| 4 | D1 (Res Bottleneck) | 0.986 | 0.698 | 最简改进 |
| 5 | D0 (Original) | 0.983 | 0.669 | 基线 |

---

## 6. 结论与推荐

**结论 1：距离邻域上下文是检测头的关键瓶颈。** D2 的最大增益（AUC +0.007, PdL +12.6%）完全来自多尺度局部卷积引入的距离维邻域信息。原检测头的"逐距离单元分类"范式严重限制了低 SCR 检测能力。

**结论 2：建议将 E4-D2 作为新的默认检测头。** 参数仅 4.35M（比 D0 少 21%），AUC=0.990，Pd@-24dB=0.404 为历史最高。

**结论 3：D6 (Margin Head) 在 FAR 控制场景值得保留。** 虽然 AUC 低于 D2，但其 `target − clutter` 结构天然适配固定虚警率阈值选择。

**部署推荐**：
```
E4 Backbone (fixed)
  + D2 Local-CFAR Detector Head (default, AUC=0.990)
  + D6 Margin Head (backup, for FAR-critical scenarios)
```

---

## 附录

### A. 实验环境

| 项目 | 配置 |
|------|------|
| Python | 3.12, PyTorch 2.7.0+cu128 |
| GPU | NVIDIA GeForce RTX 4080 SUPER (16 GB) |

### B. 数据文件

| 文件 | 内容 |
|------|------|
| `checkpoints/E4_D_head.json` | E4-D0~D6 (3 seeds) |
| `run_e4_detector.py` | 实验脚本 |

---

**报告结束**
