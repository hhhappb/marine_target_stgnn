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
    share/                     # 对外审查包、代码快照与结果副本
    *_backup*/                 # 任何备份目录

对黑名单路径允许的操作仅有：读取、在其下新增文件（logs/checkpoints 的正常训练写入）。

### 0.2 删除操作 SOP

删除任何文件（含黑名单之外的普通代码文件）必须满足全部四条：

1. 该文件出现在人工给定的任务/计划文件的明确删除清单中，逐字匹配路径；
   清单之外"看起来同类/看起来没用"的文件一律不删，记入报告待人工确认。
2. 删除前执行引用检查（全仓搜索文件名与导入路径），当前运行时引用数为 0；
   当前运行时引用包括活动源码、registry、正式配置/suite、测试、README 命令和 `work/` 配置。
   对已经人工确认退役、且在任务/计划文件中逐字列出的目标，`logs/training/`、`reports/`
   和 `share/` 中的历史配置快照、文字说明、manifest、结果路径仅作为审计证据保留，
   不计入当前运行时引用，也不得为凑引用数而修改或删除。除此之外引用数 >0 时停止并报告，
   不得连带修改引用方来凑 0，除非计划明确要求。
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
- 默认禁止修改本文件（AGENTS.md）第 0 节；仅当人工明确授权修改第 0 节并限定改动范围时，
  agent 才能在该范围内修改。其他章节的修改需在计划中明确列出改动点。

### 0.5 不确定即停止

出现以下任一情形，停止执行并输出「已完成事项 + 卡点 + 建议」，等待人工：

- 任务要求与第 0 节冲突；
- 验收/引用检查失败且两次修复无效；
- 需要触碰 0.1 黑名单或 0.3 操作闸才能达成目标；
- 发现计划未覆盖但影响执行的事实（如文件不存在、数据缺失、结果与预期不符）。

禁止用 mock 数据、跳过验证、放宽验收标准等方式"绕过卡点完成目标"。

## 1. 项目与模型代码目标

本项目用于开展海上目标检测的数据处理、模型研究、论文复现和探索性实验。ST-GNN 是当前主要对比论文和默认对比对象，`original_stgnn` 是需要与 ST-GNN 比较时的 baseline，但项目不限定只能研究 ST-GNN 主干，也不要求每个实验都与 `original_stgnn` 配对。

论文存在目标合成、距离范围、归一化、训练停止和 checkpoint 选择等未公开设置，无法据此恢复唯一实验真值。因此，后续不再通过猜测未公开参数追逐论文曲线。论文数字化曲线可以继续用于展示任务难度和结果量级；当本项目曲线高于论文曲线时，只能表述为“在本项目协议下取得更高的绝对 PD”，不得仅凭跨协议曲线差值把提升归因于模型。

实验按目标分为以下几类，配置、日志或报告中应尽量标明类型：

- `stgnn_reproduction`：人工明确要求复现 ST-GNN 论文时，按论文公开设置对齐；
- `stgnn_comparison`：需要得出与 ST-GNN 的相对结论时，在同一协议下与 `original_stgnn` 公平比较；
- `exploratory`：允许尝试其他模型、特征、训练方法和数据协议，不要求使用 ST-GNN 主干；
- `diagnostic`：用于定位数据、训练或评估问题，不作为正式性能结论。

当前研究可以回答但不限于：

1. 是否已经把 IPIX、SDRDSP 的数据生成、划分、标签、阈值和评价协议完整冻结并可追溯；
2. 当目标是比较 ST-GNN 时，新方法是否在同一协议下稳定优于 `original_stgnn`；
3. 不同方法是否在 SDRDSP 的理想、相位扰动、RCS 起伏、组合扰动和真实目标设置中保持 matched-domain 性能与跨域稳健性；
4. 是否能通过消融、negative control、实时性或观测时间实验解释性能变化；
5. 其他经人工提出或批准的海上目标检测研究问题。

模型修改的一般要求：

- 优先复用现有 registry、配置和训练评估设施；
- 需要进入统一检测流水线的模型应保持任务所需的输入输出契约；实验确有不同任务定义时，可以使用不同接口，但必须在配置或说明中写清；
- 正式实验应有可追溯配置；探索性诊断可以从最小配置开始；
- 根据结论强度选择合适的 baseline、ablation 或 negative control，不要求所有早期探索一次性齐备；
- 说明模型或实验的研究动机；如果声称使用雷达先验或检测统计先验，应解释其对应含义；
- 不通过改变标签口径、泄漏测试统计量或筛掉困难样本来提升指标。

### 1.1 主要对比论文

本项目当前主要对比论文为：

    Marine_Target_Detection_via_SpatialTemporal_Graph_Neural_Network.pdf

未加限定的“ST-GNN 论文”或“主要对比论文”指上述文件。该论文用于确定主要对比对象和 `original_stgnn` baseline，但不自动约束所有实验的模型结构和研究方向。

只有当人工明确提出“复现 ST-GNN”“按论文设置执行”“与论文结构或实验协议对齐”等要求时，才启用论文对齐约束。其他模型、协议、诊断实验和探索性实验可以独立开展，不要求强制使用 ST-GNN 主干，也不要求落在 FT、SFE、TFE 等论文模块位置。

如果实验需要得出“优于 ST-GNN”或“改进 ST-GNN”的结论，应在相同数据、划分、训练预算、阈值和评价口径下与 `original_stgnn` 公平比较。如果只是探索其他方案，可以不运行 `original_stgnn`，但不得将结果表述为 ST-GNN 复现或对 ST-GNN 的公平超越。

其他论文可以作为补充文献、复现对象或新研究主线。更换主要对比论文或严格绑定某个论文版本时，由人工明确提出即可；不要求普通实验都更新为论文对齐模式。

## 2. 修改边界

- `models/st_gnn.py` 是论文 ST-GNN 复现 baseline，默认冻结。除非明确修 bug，不在这里加入新研究想法。
- ST-GNN 复现和改进可以继续放在 `paper_modules/` 并复用论文主干；其他研究允许新增清晰命名的模型或模块，不强制做成 ST-GNN 的模块替换。
- `paper_model/` 和旧 `ExperimentalSTGNN` 已退役，不应仅为恢复旧入口而重新创建；若新研究确实需要不同骨架，应使用新的明确命名、说明用途，并避免与历史退役实现混淆。
- 优先复用 `paper_modules/experiments/` 下的 `train.py`、`auto_experiment.py`、`leakage_probe.py` 以及现有 dataset/model/eval registry。现有入口无法合理支持新任务时，可以在说明必要性和复用边界后新增入口，不得简单复制整套训练脚本制造平行实现。
- `E:\stgnn` 是前期预处理和实验代码的只读参考源；需要复用时只提取公式、结论或最小逻辑到当前仓库，不允许直接修改 `E:\stgnn`，也不能把旧脚本原样作为当前主线。
- 避免与当前任务无关的大范围重构；为支持新实验所需的小范围公共化、接口整理或去重可以实施，并在最终说明中列出影响。

## 3. 敏感模型代码审核

模型代码是高敏感区域，默认必须先说明修改意图和影响范围，再实施修改。

敏感模型代码包括：

- `models/`
- `paper_modules/models/`
- `paper_modules/losses/`

修改这些路径时应满足：

- 先写清楚：改哪个模块、为什么改、是否影响 baseline、预期输入输出 shape。
- 优先让一次修改围绕一个清晰目标；人工明确要求组合实验时，可以同时实现多个模块，但应能通过配置或消融区分其作用。
- 不允许绕过 registry、配置或统一接口直接硬编码实验分支。
- 不允许把新模块接入自定义骨架后，就声称完成了论文 ST-GNN 的 SFE/TFE/FT 替换实验。
- 修改 `models/st_gnn.py` 或已有 baseline 行为前，必须经过人工审核确认；除非是明显 bugfix，也要在最终说明中单独列出。
- ST-GNN 兼容模型修改完成后，至少做一次 `[B, 4, 14] complex -> [B, 2, 14]` 的接口检查；其他模型按其预先声明的输入输出契约做最小检查。

## 4. ST-GNN 接口与默认 baseline

ST-GNN 复现、ST-GNN 改进和需要接入现有 ST-GNN 检测流水线的模型，应保持统一模型接口：

```python
logits = model(E_complex)
```

契约如下：

```text
E_complex: complex tensor [B, P, N]
logits:    float tensor   [B, 2, N]
```

当前 IPIX ST-GNN 对照实验的参考默认值为：

```text
data_dir = datasets/ipix_dartmouth/processed/window4_stride4_related
input = complex E [B, P=4, N=14]
feature = [I, Q]
label = y_range
target_policy = related
polarizations = hh, hv, vv, vh
target_pfa = 0.001
```

该 baseline 对应原论文“直接使用 range profiles，不额外做 TF/RD 预处理”的主张。只有需要声称“改进或优于 ST-GNN”时，才必须与这个 baseline 做同协议比较。

ST-GNN 兼容模块如果改变输入通道，应在雷达特征编码器内部完成，不改变外部 `E_complex` 契约。独立探索模型可以采用其他输入接口，但必须明确数据含义、shape 和对应训练评估入口。

当实验明确声称替换 ST-GNN 论文模块时，模块位置按以下含义解释：

```text
空间图改进：替换论文 ST-GNN 的 SFE1/SFE2 位置；
时间建模改进：替换论文 ST-GNN 的 TFE1/TFE2 位置；
输入先验改进：替换或扩展论文 ST-GNN 的 FT/输入特征位置；
检测约束改进：替换 loss、阈值校准或 DetectionHead，但必须保持判决口径。
```

没有落在上述论文结构位置的实验可以作为独立模型或探索性主线形成正式结论，但不得把它表述为 ST-GNN 的对应模块替换。

## 5. ST-GNN 论文参考任务

论文 Section IV 可以作为 ST-GNN 复现或比较任务的设计参考，但不构成其他实验的必做清单。只有人工明确要求论文复现、论文对齐或与 ST-GNN 比较时，才按任务需要选择以下项目；普通探索实验无需覆盖。论文图表只作为外部背景参照，不参与跨协议的模型胜负判定。

1. IPIX Fig. 7 主实验  
   14 个 IPIX 数据集、四极化 `HH/HV/VV/VH`；每个数据集每个极化前 60% range profiles 训练，后 40% 测试；`Pfa=0.001`；报告每个 label、每个极化的 `PD`。`dataset × polarization` 独立 detector 和 pooled detector 是两个不同的合法内部协议，必须分别命名、冻结并让 baseline/候选模型使用同一协议。pooled 结果可以作为自主协议内的正式相对对照，但不得直接与论文 Fig. 7 数字化曲线归因为模型胜负。

2. SDRDSP 模拟目标 SCR 曲线 Fig. 9  
   训练背景 `20210106155330_01_staring`，测试背景 `20210106155432_01_staring`；训练 SCR `-12 dB` 到 `14 dB`，步长 `2 dB`；测试 SCR `-24 dB` 到 `14 dB`，步长 `2 dB`；报告 `Pfa=0.0001, 0.001, 0.01` 下的 `PD-SCR` 曲线。目标合成器、距离裁剪、随机流和 manifest 必须冻结，baseline 与候选模型必须使用逐值相同的数据；论文 Fig. 9 曲线仅作跨协议背景参照。

3. SDRDSP SCR=0 dB 可视化 Fig. 10  
   在 `Pfa=0.0001` 下比较检测图；若任务目标是比较 ST-GNN，则与同协议 `original_stgnn` 对照。论文 Fig. 10 只作视觉背景参照。

4. 实时性对比 Table V  
   使用 IPIX `19931107_135603_starea.cdf` 的 `HH` 极化；前 60% 训练，后 40% 测试；记录 observation time、preprocessing time、detector running time。比较 ST-GNN 时应使用同机同协议 `original_stgnn`；其他模型按自身任务报告效率和性能权衡。

5. Pfa-PD 关系 Fig. 11  
   沿用 SDRDSP 模拟目标实验，报告不同 SCR 下 `PD` 随 `Pfa` 变化的曲线。

6. 快速运动目标 Fig. 12  
   在 `20210106155432_01_staring` 背景中加入速度 `100 m/s`、SCR `6 dB` 的模拟目标，报告 `Pfa=0.0001` 下的检测结果。

7. SDRDSP 真实目标 Fig. 13/14  
   使用 `20210106150614_01_staring`、`20210106150614_02_staring`、`20210106150614_03_staring` 训练，使用 `20210106160919_01_staring` 测试，报告 `Pfa=0.0001` 下 buoy、ship、island 的检测结果。

8. 消融实验 Fig. 15  
   在同一冻结内部协议下比较候选模型及其受控消融；需要归因于 ST-GNN 改进时，再加入 `original_stgnn` 配对对照。

9. 观测时间影响 Fig. 16  
   使用 IPIX `19931107_135603_starea.cdf` 的 `HH` 极化，报告不同 observation time 在 `Pfa=0.001, 0.005, 0.01` 下的 `PD` 曲线，证明默认 `P=4` 或改进设置的合理性。

## 6. 实验模块规则

- 不同研究思路应能通过配置、registry 类型或清晰命名加以区分；可以共享公共实现，不强制每个想法都新建一套文件。只有 ST-GNN 模块替换实验需要在名称中体现 `sfe_replacement_*`、`tfe_replacement_*`、`feature_replacement_*` 等替换位置。
- 新模块必须 fail loud：未知 `type`、不支持形状、缺少必要配置时直接抛错，不静默回退。
- 雷达先验模块必须写清物理含义，例如相位线性度、Doppler 峰值、cell-wise stats、CFAR 参考窗或距离衰减。
- 模块代码中的注释、docstring 和面向开发者的说明必须使用中文；只保留必要的简短解释，优先说明雷达物理含义、shape 契约和非显然计算。
- 普通优化器、scheduler、dropout、batch size、梯度裁剪、通用 MLP/Conv 堆叠不能包装成“雷达先验”。
- IPIX `N=14` 与 SDRDSP `N=256` 不能混用参数假设。依赖大范围距离参考窗的方法必须重新设计并说明原因。
- 不再推荐的实验入口或配置可以明确标记为历史、诊断或 deprecated；确需删除时必须走第 0.2 节删除 SOP，不要求在其他任务中顺带清理。

推荐用以下组别命名配置和结果目录：

```text
sfe_replacement/          Raw I/Q + original SFE 或 SFE 替换模块
feature_replacement/      PL/RPH, amplitude/phase, Doppler, phase diff
statistical_prior/        cell-wise stats, phase stability, local stats
spatial_prior/            local graph, CFAR/reference-window graph, similarity graph
tfe_replacement/          ConvGRU, multiscale TCN, range migration temporal module
pfa_aware/                Pfa-aware loss or decision constraints
ablation/                 controlled removals or negative controls
stgnn_reproduction/       明确按 ST-GNN 论文公开设置开展的复现
stgnn_comparison/         与 original_stgnn 同协议比较
exploratory/              其他模型、特征或训练方法探索
diagnostic/               数据、训练和评价问题诊断
```

正式实验配置应能按任务需要说明：

- 实验目标、类型和主要改动；
- 使用了哪些额外特征、先验或训练机制；
- 参数量或输入通道是否明显变化；
- 结果应与哪个 baseline、已有结果或消融项比较；若只是探索或诊断，可以明确暂不做正式 baseline 结论。
- 删除不再代表主线的配置/入口时，同样必须走第 0.2 节删除 SOP，不得在其他任务中顺带执行。

## 7. 评估、阈值与结果口径

ST-GNN 论文兼容评价使用杂波概率 `o0`：

```text
o0 <= threshold -> target
o0 >  threshold -> clutter
```

ST-GNN 论文兼容评价默认使用 `o0`。如果 softmax 饱和并列导致序统计量退化，可以在实验前预注册 `logit_margin = logit_target - logit_clutter` 作为数值稳定的阈值分数；进行模型比较时，各模型必须使用事先确定且公平的分数空间、阈值规则和阈值来源，并在报告中说明与论文式(15)的差异。其他任务可以定义不同检测分数和判决方向，但必须写清含义，禁止在看到结果后为不同模型或测试域挑选更有利的评价方式。

需要按 Pfa 标定的正式检测结果，其阈值来源优先级为：

1. 训练集杂波样本；
2. 独立校准集杂波样本；
3. 仅用于诊断的测试集杂波样本。

第三种不能写成正式性能结论。采用 Pfa 阈值的正式报告应写明：

```text
threshold_source = train | calibration | test_diagnostic
target_pfa = ...
actual_pf = ...
num_clutter_bins_for_threshold = ...
```

正式评价应根据任务和数据条件尽量报告：

- 配置文件路径；
- checkpoint 路径或 commit/运行标识；
- 数据目录、极化列表、训练/测试文件数；
- seed、epoch、batch size、learning rate；
- `PD`、实际 `PF`、threshold、TP/FN/FP/TN；
- 适用时报告按极化、文件或场景分组的结果；
- 自主协议名称、版本以及可获得的 manifest/hash；
- 只有论文复现或论文比较任务才要求列出相对论文公开设置的已知差异。

不允许通过改变标签定义、泄漏测试统计量、筛掉困难样本或给候选模型额外训练预算来提升指标。与论文 Fig. 7/Fig. 9 或其他论文对比时，必须说明标签口径、阈值来源、数据切分、目标 Pfa 和所有已知协议差异。

任何声称“在本项目协议下优于 ST-GNN”或把提升归因于某个 ST-GNN 改进模块的正式结论，应满足：

- 明确冻结自主协议名称、配置、数据 manifest/hash、主指标和验收线；
- 与 `original_stgnn` 在同一数据、同一预处理、同一标签、同一阈值规则、同一训练 seed 和同一训练预算下做单变量配对对照；
- 优先完成至少 3 个配对训练 seed，并报告逐 seed 结果和均值/标准差；只有 1 个 seed 时，应明确标记为初步结果，不禁止继续开展实验；
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

以下方向需要额外论证，但可以在明确实验目的后开展：

- 单纯把 STFT/WVD 图送入普通 CNN；
- 大型 PLM 或复杂预训练；
- 仅使用 global stats 作为主创新；
- 未重新设计的小 N Local RMS/CFAR 大窗口；
- 使用测试集统计量做归一化或正式阈值仍属于数据泄漏风险，只能用于明确标注的诊断，不能用于正式性能结论。

## 10. 修改前检查

新增实验前根据任务规模检查适用项，不要求对无关项逐条作答：

1. 这个修改服务于什么研究问题，属于复现、比较、探索还是诊断？
2. 如果声称改进 ST-GNN，替换的是 FT、SFE1/SFE2、TFE1/TFE2、DetectionHead、loss 还是阈值校准？独立模型无需对应论文位置。
3. 输入输出契约是什么；若接入 ST-GNN 流水线，是否保持 `model(E_complex) -> [B, 2, N]`？
4. 阈值是否来自训练/校准杂波，而不是测试集？
5. 只有需要与 ST-GNN 比较时，`original_stgnn` 是否会在相同数据、seed、训练预算和评价口径下作为配对 baseline？
6. 当前结论强度是否需要 baseline、消融或 negative control？早期探索可以后补，但不得提前作强结论。
7. 是否能用小样本命令先验证 shape 和 loss 不崩？

## 11. 验证要求

- 改 ST-GNN 兼容模型时，至少做 shape/接口检查：输入 `[B, 4, 14] complex`，输出 `[B, 2, 14]`；其他模型按其声明契约检查。
- 改 registry 时，检查已知类型可构建、未知类型会抛出清晰错误。
- 改 ST-GNN 或 Pfa 评价逻辑时，检查阈值排序、所用分数的判决方向以及实际 `PF`；其他评价协议检查自身预先声明的指标逻辑。
- 无法运行完整训练时，要报告跳过原因，并尽量运行小样本或单 batch 验证。
- 改动影响现有 IPIX 训练/评估入口、共享 registry 或 ST-GNN IPIX 路径时，运行以下 smoke；与该路径无关的独立实验无需强制运行：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\real_imag_sfe_replacement_original_sfe.yaml --epochs 1 --max-train-windows 64 --max-test-windows-per-file 5 --no-progress --log-interval 1
```

- 改动影响 SCR 复现、`original_stgnn` 或 `pd_scr_curve` 时，运行以下 SDRDSP SCR smoke：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\repro_original_stgnn_scr256.yaml --epochs 2 --no-progress --log-interval 10
```

## 12. Git push 规则

本节管 push 内容取舍；破坏性 git 操作一律见第 0.3 节。

### 12.1 主线优先与新分支审批

- `main` 是默认开发与交付分支。新增代码、配置、文档或实验实现本身不构成创建新分支的理由。
- 每项任务开始前，agent 必须先只读检查当前分支、`main`/`origin/main` 的关系、工作区状态和预期改动范围，优先判断改动能否安全地直接提交或并入 `main`。
- 若现有提交可无冲突地线性并入 `main`，应先完成必要审查与验证，再并入 `main`；不得为了普通代码修改默认保留长期功能分支，也不得在已有功能分支上继续无边界累积后续任务。
- 只有在无法安全并入 `main` 时，例如存在真实提交分叉、未解决冲突、必须隔离的高风险实验或外部 PR 流程要求，agent 才能建议创建新分支。建议时必须先说明阻塞原因、分支用途、预计改动范围和回并计划，并取得人工当次明确批准。
- 未取得人工明确批准前，禁止执行任何创建本地或远端分支的操作，包括 `git switch -c`、`git checkout -b`、`git branch <new>` 或等价命令。过去任务中的分支授权不得自动延续到新任务。
- 功能分支并入 `main` 后，agent 必须报告远端分支是否仍保留。删除本地或远端分支仍受第 0.3 节约束，必须由人工亲自执行或当场逐条批准。

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

最终提交说明应按本次任务适用范围写清：

- 本次改动的研究目标和实验类型；只有 ST-GNN 改进才需要注明 SFE、TFE、FT、loss 或阈值等论文结构位置；
- 新增了哪些配置、模型、实验或消融；
- 做过哪些最小验证，例如 shape check、dry-run、smoke run 或完整 suite；
- 使用了哪个实验协议和比较对象；不涉及 ST-GNN 比较时无需强制运行 `original_stgnn`；
- 哪些结果是正式结论，哪些只是探索、诊断或跨协议背景比较。
