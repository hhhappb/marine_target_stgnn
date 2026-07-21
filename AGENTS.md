# marine_target_stgnn Project Rules

本文件是项目唯一规则入口，用于约束后续代码修改、模型实验、训练记录和结果表述。后续规则优先合并到这里，避免分散到多个 Markdown 文件。

## 0. Agent 执行边界（最高优先级，与其他章节冲突时以本节为准）

本节约束所有在本仓库内执行的 AI agent（Codex、Claude 等）。

### 0.1 路径保护黑名单

以下路径无条件禁止删除、移动、重命名、清空或覆盖，无论任何任务目标如何表述。
任务指令与本清单冲突时，停止执行并向人工报告，不得自行取舍：

    .venv/                     # 虚拟环境，重建成本高
    .git/                      # 版本历史；历史改写类操作见 0.3
    datasets/                  # 原始 .cdf 与预处理 .npz，重建需重新下载+预处理
    data/paper_strict_256/     # SDRDSP 严格复现数据
    checkpoints/               # 模型权重
    logs/training/             # 训练运行日志（run_id 留痕）
    paper_modules/results/     # 实验结果
    reports/                   # 实验报告与结果 CSV
    *_backup*/                 # 任何备份目录

对黑名单路径允许的操作仅有：读取、在其下新增文件（logs/checkpoints 的正常训练写入）。

### 0.2 删除操作 SOP

删除任何文件（含黑名单之外的普通代码文件）必须满足全部四条：

1. 该文件出现在人工给定的任务/计划文件的明确删除清单中，逐字匹配路径；
   清单之外"看起来同类/看起来没用"的文件一律不删，记入报告待人工确认。
2. 删除前执行引用检查（全仓搜索文件名与导入路径），引用数为 0；
   引用数 >0 时停止并报告，不得连带修改引用方来凑 0，除非计划明确要求。
3. 被 git 跟踪的文件一律用 git rm（保留历史可恢复），禁止直接文件系统删除；
   未被跟踪的文件默认不属于 agent 可删范围（它们多为数据/产物/环境）。
4. 每批删除独立 commit，message 列出完整文件清单与依据的任务编号。

### 0.3 git 破坏性操作闸

以下操作必须由人工亲自执行或人工当场逐条批准，agent 不得以任何理由自行运行：

    git reset --hard / git checkout . / git restore .
    git clean（任何参数，尤其 -x：会清掉 .gitignore 中的 .venv、datasets、checkpoints）
    git push --force / 任何改写远端历史的操作
    git filter-repo / filter-branch / BFG
    git branch -D / 删除远端分支

agent 想"恢复干净工作区"时只允许：报告当前 git status，等人工决定。

### 0.4 环境边界

- 所有 Python 命令固定使用 .\.venv\Scripts\python.exe，不使用全局解释器。
- 禁止安装、卸载、升级、降级任何依赖（含 pip/conda/uv）。缺依赖 → 停止并报告。
- 禁止修改 .venv/ 内任何文件，禁止重建虚拟环境。
- 禁止修改本文件（AGENTS.md）第 0 节；其他章节的修改需在计划中明确列出改动点。

### 0.5 不确定即停止

出现以下任一情形，停止执行并输出「已完成事项 + 卡点 + 建议」，等待人工：

- 任务要求与第 0 节冲突；
- 验收/引用检查失败且两次修复无效；
- 需要触碰 0.1 黑名单或 0.3 操作闸才能达成目标；
- 发现计划未覆盖但影响执行的事实（如文件不存在、数据缺失、结果与预期不符）。

禁止用 mock 数据、跳过验证、放宽验收标准等方式"绕过卡点完成目标"。

## 1. 项目与模型代码目标

本项目当前第一目标是在**明确、冻结、可公开说明的自主实验协议**下，以遵循论文公开公式和 Fig. 2 架构实现的 `original_stgnn` 为内部 baseline，通过同数据、同划分、同训练预算、同阈值和多 seed 配对实验，证明新方法的相对性能提升。严格复现论文 Fig. 7/Fig. 9 的绝对曲线不再作为模型研究的前置闸门。

论文存在目标合成、距离范围、归一化、训练停止和 checkpoint 选择等未公开设置，无法据此恢复唯一实验真值。因此，后续不再通过猜测未公开参数追逐论文曲线。论文数字化曲线可以继续用于展示任务难度和结果量级；当本项目曲线高于论文曲线时，只能表述为“在本项目协议下取得更高的绝对 PD”，不得仅凭跨协议曲线差值把提升归因于模型。

本项目模型代码的目标不是任意堆叠深度学习模块，也不是先泛泛研究雷达先验。雷达先验、模块化实验、Pfa-aware loss、时空图改进等都只是超过论文 ST-GNN baseline 的候选手段。

当前研究主线是回答：

1. 是否已经把 IPIX、SDRDSP 的数据生成、划分、标签、阈值和评价协议完整冻结并可追溯；
2. 新方法是否在同一自主协议下稳定优于 `original_stgnn`，且提升能够跨训练 seed 复现；
3. 新方法是否在 SDRDSP 的理想、相位扰动、RCS 起伏、组合扰动和真实目标设置中保持更好的 matched-domain 性能与跨域稳健性；
4. 是否能通过消融、negative control、实时性和观测时间实验证明提升来自目标模块，而不是更多训练预算、协议偏移或牺牲 ST-GNN 核心优势。

有效模型修改必须满足：

- 保持 `model(E_complex) -> [B, 2, N]` 的统一接口；
- 能通过 YAML 配置开关控制；
- 有对应 baseline、ablation 或 negative control；
- 能解释其雷达物理含义或检测统计含义；
- 不通过改变标签口径、泄漏测试统计量或筛掉困难样本来提升指标。

## 2. 修改边界

- `models/st_gnn.py` 是论文 ST-GNN 复现 baseline，默认冻结。除非明确修 bug，不在这里加入新研究想法。
- 新论文想法、消融模块和可配置实验都放在 `paper_modules/`，但必须对齐论文 ST-GNN 主干做模块替换，不能另起一条自定义模型流水线。
- `paper_model/` 已删除，后续不得重新创建；旧论文模型主干保留在 `models/`，正式新实验主线以 `paper_modules/models/sfe_replacement_stgnn.py` 等论文主干对齐骨架为入口。
- 已删除旧的 `ExperimentalSTGNN` 自定义骨架及其配置；后续不得重新创建同类 `Feature -> Spatial -> Gate -> Temporal -> Head` 的非论文主干实验入口。
- 合法训练/评估入口只有 `paper_modules/experiments/` 下的 `train.py`、`auto_experiment.py`、`leakage_probe.py`；新增独立训练/评估脚本视为违规。新增数据集 = `paper_modules/datasets/` 一个 registry 条目 + yaml `dataset` 段；新增评估口径 = `eval.protocol` 一个分支；均不得以复制训练脚本方式实现。
- `E:\stgnn` 是前期预处理和实验代码的只读参考源；需要复用时只提取公式、结论或最小逻辑到当前仓库，不允许直接修改 `E:\stgnn`，也不能把旧脚本原样作为当前主线。
- 不做顺手重构。每次修改必须能对应到当前研究问题、实验协议或明确 bug。

## 3. 敏感模型代码审核

模型代码是高敏感区域，默认必须先说明修改意图和影响范围，再实施修改。

敏感模型代码包括：

- `models/`
- `paper_modules/models/`
- `paper_modules/losses/`

修改这些路径时必须满足：

- 先写清楚：改哪个模块、为什么改、是否影响 baseline、预期输入输出 shape。
- 不允许在一次修改里同时改多个研究想法。一个实验想法对应一组最小代码改动。
- 不允许绕过 registry、配置或统一接口直接硬编码实验分支。
- 不允许把新模块接入自定义骨架后，就声称完成了论文 ST-GNN 的 SFE/TFE/FT 替换实验。
- 修改 `models/st_gnn.py` 或已有 baseline 行为前，必须经过人工审核确认；除非是明显 bugfix，也要在最终说明中单独列出。
- 模型代码修改完成后，至少做一次 `[B, 4, 14] complex -> [B, 2, 14]` 的接口检查。

## 4. 稳定接口与默认 baseline

所有 ST-GNN 变体必须保持统一模型接口，并保留论文主干语义：

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
data_dir = datasets/ipix_dartmouth/processed/window4_stride4_related
input = complex E [B, P=4, N=14]
feature = [I, Q]
label = y_range
target_policy = related
polarizations = hh, hv, vv, vh
target_pfa = 0.001
```

该 baseline 对应原论文“直接使用 range profiles，不额外做 TF/RD 预处理”的主张。后续所有改进模块都必须和这个 baseline 比较。

如果模块改变输入通道，只能在雷达特征编码器内部完成，不能改变外部 `E_complex` 契约。

模块替换的正式含义如下：

```text
空间图改进：替换论文 ST-GNN 的 SFE1/SFE2 位置；
时间建模改进：替换论文 ST-GNN 的 TFE1/TFE2 位置；
输入先验改进：替换或扩展论文 ST-GNN 的 FT/输入特征位置；
检测约束改进：替换 loss、阈值校准或 DetectionHead，但必须保持判决口径。
```

如果某个实验没有落在上述论文结构位置，只能作为临时诊断代码，不能作为论文主线结论。

## 5. 内部基线实验与论文参考任务

论文 Section IV 继续作为任务设计参考，但不再要求先复现其绝对曲线。以下实验应根据当前研究问题选择覆盖；每项正式模型结论必须在冻结的本项目协议内同时运行 `original_stgnn` 和候选模型。论文图表只作为外部背景参照，不参与内部模型胜负判定。

1. IPIX Fig. 7 主实验  
   14 个 IPIX 数据集、四极化 `HH/HV/VV/VH`；每个数据集每个极化前 60% range profiles 训练，后 40% 测试；`Pfa=0.001`；报告每个 label、每个极化的 `PD`。`dataset × polarization` 独立 detector 和 pooled detector 是两个不同的合法内部协议，必须分别命名、冻结并让 baseline/候选模型使用同一协议。pooled 结果可以作为自主协议内的正式相对对照，但不得直接与论文 Fig. 7 数字化曲线归因为模型胜负。

2. SDRDSP 模拟目标 SCR 曲线 Fig. 9  
   训练背景 `20210106155330_01_staring`，测试背景 `20210106155432_01_staring`；训练 SCR `-12 dB` 到 `14 dB`，步长 `2 dB`；测试 SCR `-24 dB` 到 `14 dB`，步长 `2 dB`；报告 `Pfa=0.0001, 0.001, 0.01` 下的 `PD-SCR` 曲线。目标合成器、距离裁剪、随机流和 manifest 必须冻结，baseline 与候选模型必须使用逐值相同的数据；论文 Fig. 9 曲线仅作跨协议背景参照。

3. SDRDSP SCR=0 dB 可视化 Fig. 10  
   在 `Pfa=0.0001` 下比较检测图，要求候选模型的目标检出和虚警优于或不劣于同协议 `original_stgnn`。论文 Fig. 10 只作视觉背景参照。

4. 实时性对比 Table V  
   使用 IPIX `19931107_135603_starea.cdf` 的 `HH` 极化；前 60% 训练，后 40% 测试；记录 observation time、preprocessing time、detector running time，并与同机同协议 `original_stgnn` 比较。ST-GNN 的核心优势是 4 个 range profiles 一次检测所有 14 个 range cells，改进模型不能明显破坏这一点。

5. Pfa-PD 关系 Fig. 11  
   沿用 SDRDSP 模拟目标实验，报告不同 SCR 下 `PD` 随 `Pfa` 变化的曲线。

6. 快速运动目标 Fig. 12  
   在 `20210106155432_01_staring` 背景中加入速度 `100 m/s`、SCR `6 dB` 的模拟目标，报告 `Pfa=0.0001` 下的检测结果。

7. SDRDSP 真实目标 Fig. 13/14  
   使用 `20210106150614_01_staring`、`20210106150614_02_staring`、`20210106150614_03_staring` 训练，使用 `20210106160919_01_staring` 测试，报告 `Pfa=0.0001` 下 buoy、ship、island 的检测结果。

8. 消融实验 Fig. 15  
   在同一冻结内部协议下比较 `original_stgnn`、候选模型及其受控消融，证明空间、时间或输入先验模块的独立贡献。

9. 观测时间影响 Fig. 16  
   使用 IPIX `19931107_135603_starea.cdf` 的 `HH` 极化，报告不同 observation time 在 `Pfa=0.001, 0.005, 0.01` 下的 `PD` 曲线，证明默认 `P=4` 或改进设置的合理性。

## 6. 实验模块规则

- 一个研究思路对应一个独立实现文件、一个 registry 条目、一个 YAML 配置，配置名必须体现替换位置，例如 `sfe_replacement_*`、`tfe_replacement_*`、`feature_replacement_*`。
- 新模块必须 fail loud：未知 `type`、不支持形状、缺少必要配置时直接抛错，不静默回退。
- 雷达先验模块必须写清物理含义，例如相位线性度、Doppler 峰值、cell-wise stats、CFAR 参考窗或距离衰减。
- 模块代码中的注释、docstring 和面向开发者的说明必须使用中文；只保留必要的简短解释，优先说明雷达物理含义、shape 契约和非显然计算。
- 普通优化器、scheduler、dropout、batch size、梯度裁剪、通用 MLP/Conv 堆叠不能包装成“雷达先验”。
- IPIX `N=14` 与 SDRDSP `N=256` 不能混用参数假设。依赖大范围距离参考窗的方法必须重新设计并说明原因。
- 禁止保留指向错误骨架的配置文件；发现不再代表正式主线的实验入口或配置，应删除而不是降级归档。

推荐用以下组别命名配置和结果目录：

```text
sfe_replacement/          Raw I/Q + original SFE 或 SFE 替换模块
feature_replacement/      PL/RPH, amplitude/phase, Doppler, phase diff
statistical_prior/        cell-wise stats, phase stability, local stats
spatial_prior/            local graph, CFAR/reference-window graph, similarity graph
tfe_replacement/          ConvGRU, multiscale TCN, range migration temporal module
pfa_aware/                Pfa-aware loss or decision constraints
ablation/                 controlled removals or negative controls
```

每个实验配置必须能说明：

- 相比 baseline 改了哪个模块；
- 是否引入额外雷达先验或其他超越 ST-GNN 的机制；
- 参数量或输入通道是否明显变化；
- 结果应与哪个 baseline 或消融项比较。
- 删除不再代表主线的配置/入口时，同样必须走第 0.2 节删除 SOP，不得在其他任务中顺带执行。

## 7. 评估、阈值与结果口径

ST-GNN 判决使用杂波概率 `o0`：

```text
o0 <= threshold -> target
o0 >  threshold -> clutter
```

论文兼容评价默认使用 `o0`。如果 softmax 饱和并列导致序统计量退化，可以在实验前预注册 `logit_margin = logit_target - logit_clutter` 作为数值稳定的阈值分数；baseline 与候选模型必须使用同一分数空间、同一阈值规则和同一阈值来源，并在报告中明确标记其不是论文式(15)的原始 `o0` 实现。禁止在看到结果后为不同模型或测试域挑选更有利的分数空间。

正式结果的阈值来源优先级：

1. 训练集杂波样本；
2. 独立校准集杂波样本；
3. 仅用于诊断的测试集杂波样本。

第三种不能写成正式性能结论。任何报告都必须写明：

```text
threshold_source = train | calibration | test_diagnostic
target_pfa = ...
actual_pf = ...
num_clutter_bins_for_threshold = ...
```

评价至少报告：

- 配置文件路径；
- checkpoint 路径或 commit/运行标识；
- 数据目录、极化列表、训练/测试文件数；
- seed、epoch、batch size、learning rate；
- `PD`、实际 `PF`、threshold、TP/FN/FP/TN；
- 按极化分组结果；
- 最差文件/极化组合；
- 自主协议名称、版本、manifest/hash，以及相对论文公开设置的已知差异。

不允许通过改变标签定义、泄漏测试统计量、筛掉困难样本或给候选模型额外训练预算来提升指标。与论文 Fig. 7/Fig. 9 或其他论文对比时，必须说明标签口径、阈值来源、数据切分、目标 Pfa 和所有已知协议差异。

任何声称“在本项目协议下优于原论文 ST-GNN 架构 baseline”的正式结论，必须同时满足：

- 明确冻结自主协议名称、配置、数据 manifest/hash、主指标和验收线；
- 与 `original_stgnn` 在同一数据、同一预处理、同一标签、同一阈值规则、同一训练 seed 和同一训练预算下做单变量配对对照；
- 原则上至少完成 3 个配对训练 seed，报告逐 seed 结果、均值/标准差和配对置信区间；若只完成 1 个 seed，只能写作候选或初步证据；
- 同时报告 checkpoint、threshold_source、actual PF、PD/PD-SCR AUC、参数量和推理时间，且 baseline 与候选模型都通过预注册 PF 守门；
- 配置中除目标研究模块外不得混入额外特征、数据增强、loss、训练轮数或调参预算；必要时增加 negative control 或消融证明归因。

论文数字化曲线可以与本项目结果同图展示，并可陈述“本项目协议下的绝对曲线高于论文报告曲线”。该陈述必须紧邻注明这是 cross-protocol contextual comparison；仅凭该差值不得写成“严格超过论文原始实验”或“提升由新模型造成”。

## 8. 数据、日志与产物

- 原始 `.cdf`、处理后 `.npz`、模型权重 `.pth`、checkpoint、实验结果图和日志默认视为实验复现资产，不应随普通代码修改一起提交。
- 保护清单以第 0.1 节黑名单为准；“默认不 push” 只表示不随普通源码提交上传到 Git，不表示这些产物可以删除。
- 若需要释放磁盘空间，只能先列出待处理文件、说明对应实验 run/checkpoint、确认已有备份或可重新训练成本，再由人工明确批准后执行。删除训练产物前必须至少记录 `run_id`、配置文件、checkpoint 路径和指标摘要，并确认 `artifacts.txt` 或报告中仍能追溯。
- 小型配置、实验协议、结果摘要 Markdown/CSV 可以保留，但必须能说明生成方式。
- 新增数据处理逻辑必须写明输出数组 key、shape、标签含义和是否使用训练/测试统计量。

所有训练和评估脚本后续应统一把日志写入：

```text
logs/training/<run_id>/
```

`run_id` 推荐格式：

```text
YYYYMMDD-HHMMSS_<experiment_name>
```

每个 run 目录至少包含：

```text
config.yaml              # 本次运行配置快照
stdout.log               # 训练/评估标准输出
metrics.csv/json         # epoch 指标或最终评估指标
summary.md               # 面向论文记录的结果摘要
artifacts.txt            # checkpoint、图表、报告文件路径
```

`checkpoints/` 可以继续保存模型权重，但正式报告必须能从 `logs/training/<run_id>/artifacts.txt` 追溯到对应 checkpoint。

## 9. 可复用结论与禁区

可以优先复用：

- PL/RPH；
- cell-wise stats；
- amplitude/phase/delta phase；
- Doppler/FFT 统计；
- 距离邻接、距离衰减、相似性空间图；
- Pfa-aware loss 或训练集杂波阈值。

当前不作为主线：

- 单纯把 STFT/WVD 图送入普通 CNN；
- 大型 PLM 或复杂预训练；
- 仅使用 global stats 作为主创新；
- 未重新设计的小 N Local RMS/CFAR 大窗口；
- 使用测试集统计量做归一化或正式阈值。

## 10. 修改前检查

新增实验前先回答：

1. 这个修改服务于哪个冻结自主协议、研究问题和预注册主指标？
2. 替换的是论文 ST-GNN 的哪个位置：FT、SFE1/SFE2、TFE1/TFE2、DetectionHead、loss 还是阈值校准？
3. 是否保持 `model(E_complex) -> [B, 2, N]`？
4. 阈值是否来自训练/校准杂波，而不是测试集？
5. `original_stgnn` 是否会在相同数据、seed、训练预算和评价口径下作为配对 baseline？
6. 是否需要一个 negative control，例如 original_sfe、stats-only、global-only 或 shuffle stats？
7. 是否能用小样本命令先验证 shape 和 loss 不崩？

## 11. 验证要求

- 改模型模块时，至少做 shape/接口检查：输入 `[B, 4, 14] complex`，输出 `[B, 2, 14]`。
- 改 registry 时，检查已知类型可构建、未知类型会抛出清晰错误。
- 改评估逻辑时，检查 Pfa 阈值排序、判决方向 `o0 <= threshold -> target`、以及实际 `PF`。
- 无法运行完整训练时，要报告跳过原因，并尽量运行小样本或单 batch 验证。
- 改动训练/评估入口或数据集 registry 后，默认先跑 IPIX 线 smoke：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\real_imag_sfe_replacement_original_sfe.yaml --epochs 1 --max-train-windows 64 --max-test-windows-per-file 5 --no-progress --log-interval 1
```

- 改动 SCR 复现、`original_stgnn` 或 `pd_scr_curve` 后，默认先跑 SDRDSP SCR 线 smoke：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\repro_original_stgnn_scr256.yaml --epochs 2 --no-progress --log-interval 10
```

## 12. Git push 规则

本节管 push 内容取舍；破坏性 git 操作一律见第 0.3 节。

Git push 只推能复现实验主线、代码逻辑或论文记录的必要文件，不推本地运行产物。

每次 push 前必须先做清单审查：

```powershell
git status --short
git diff --stat
```

默认应 push：

- 源码修改：`paper_modules/experiments/`、`paper_modules/models/`、`paper_modules/losses/`、`utils/`、必要的 `scripts/`。
- 可复现实验配置：`paper_modules/configs/*.yaml`、`paper_modules/configs/suites/*.yaml`。
- 小型实验协议与结果记录：`reports/*.md`、必要的 `reports/*.csv`、项目 README 或 AGENTS 规则。
- 有意删除的旧错误入口、旧配置和旧模块，但必须能说明它们为何不再代表当前主线。

默认不 push：

- 原始或处理后数据：`*.cdf`、`*.npz`、`datasets/**/processed/` 新生成目录。
- 训练日志和运行产物：`logs/training/`、`paper_modules/results/`。这些目录默认不随普通提交上传，但必须保留本地，不得作为垃圾文件清理。
- 模型权重和 checkpoint：`*.pth`、`*.pt`、`*.ckpt`、`checkpoints/`。这些文件默认不随普通提交上传，但属于实验复现资产，不得未经确认删除。
- 临时网页、草稿和本地调试文件，例如 `wechat_article.html`、临时截图、缓存文件。
- 大型 PDF、图像或自动生成图表，除非明确用于论文记录且经过人工确认。

禁止使用 `git add .` 直接打包所有改动。必须按文件或按明确目录选择性 stage。若同一工作区同时存在代码、数据、日志和临时文件，只 stage 本次研究问题需要的源码、配置和小型报告。

推荐 push 前检查：

```powershell
git diff --cached --stat
git diff --cached --name-only
```

最终提交说明必须写清：

- 本次改的是哪个论文结构位置，例如 SFE、TFE、FT、loss 或阈值评估；
- 新增了哪些配置或消融实验；
- 做过哪些最小验证，例如 shape check、dry-run、smoke run 或完整 suite；
- 使用了哪个冻结自主协议和 `original_stgnn` 配对 baseline；
- 哪些结果是正式内部相对结论、哪些只是跨域诊断或与论文曲线的跨协议背景比较。
