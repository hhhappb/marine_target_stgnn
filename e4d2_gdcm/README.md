# E4-D2 (GDCM): CVOCA-Fusion ST-GNN with Direction-Attributed Magnitude Detector

Amplitude-Phase Coupled Detection whose detector head uses the **channel-wise
log magnitude of the CUT-vs-background difference** as its judgement input.

**Final adopted setting (A0 / `gdcm_a0`)**: the simplest and most stable
representation — the judgement network consumes ONLY

```
Δ = U_d − U_h
D_a = log(1 + |Δ|)      [B,128,R]
D_a → Conv1d(128→128) → GELU → Dropout → Conv1d(128→2)
```

(Group-direction variants g0..g4 remain available for ablation.)

## Self-Contained Package

This folder is a **complete, self-contained model package**. To train the
model, you only need:

1. This `e4d2_gdcm/` folder
2. The raw SDRDSP .mat files (or IPIX Dartmouth .cdf data)
3. Python dependencies listed in `requirements.txt`

No other project code is required.

## Variants (ablation reference)

| ID | D_s | D_a | Channels | Purpose |
|---|---|---|---|---|
| **A0 (DEFAULT, final)** | — | log(1+\|Δ\|) | 128 | **simplest & most stable** |
| **G0** | Δ | \|Δ\| | 256 | baseline (E4-D2-RDJ) |
| **G1** | Δ/ρ_g | \|Δ\| | 256 | true direction separation |
| **G2** | Δ/ρ_g | \|Δ\|/(\|U_d\|+\|U_h\|) | 256 | relative magnitude |
| **G3** | Δ/ρ_g | log(1+\|Δ\|) | 256 | suppress extreme clutter outliers |
| **G4** | Δ/ρ_g | ρ_g (8ch) | 136 | parameter-compressed control |

## Architecture (final A0)

```
E4Backbone (unchanged) → 1216ch → GdcmDetector(variant='a0'):
  F_d → DWConv(3/7/15) → H_ms
  U_d = GN(Proj_d(F_d)),  U_h = GN(Proj_h(H_ms))
  Δ = U_d − U_h
  D_a = log(1 + |Δ|)            [B,128,R]
  D_a → Conv1d(128→128) → GELU → Dropout(0.1) → Conv1d(128→2)
```

## Quick Start

```bash
pip install -r requirements.txt
python -m e4d2_gdcm.experiments.train_sdrdsp \
    --train data/20210106155330_01_staring.mat \
    --test  data/20210106155432_01_staring.mat \
    --detector gdcm_a0 --epochs 200 --seeds 42,123,456 --batch 24 \
    --output checkpoints/gdcm_best.pt
```

### Inference

```python
from e4d2_gdcm import E4D2

model = E4D2()                        # ← default gdcm_a0 (D_a only, log magnitude)
logits = model(x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global)
```

## Package Structure

```
e4d2_gdcm/
├── models/                  # Model definitions
│   ├── stgnn_backbone.py   # GAT, STFE, TFE, STGNNBackbone (shared)
│   └── e4d2.py             # Encoders, CVOCA, RdjDetector, GdcmDetector, E4D2
├── utils/                   # Data & preprocessing
│   ├── preprocess.py       # E4D2Preprocessor
│   ├── data.py             # SDRDSPBuilder, .mat loaders
│   └── inference.py        # load_model, run_inference, calibrate_threshold
├── losses/                  # Loss functions
│   └── losses.py           # CE, weighted CE, Focal Loss
├── configs/                 # Experiment configuration (YAML)
│   ├── default.yaml        # Shared defaults
│   └── sdrdsp.yaml         # SDRDSP dataset config
├── experiments/             # Runnable training scripts
│   ├── train_sdrdsp.py     # SDRDSP training + evaluation (--detector)
│   └── train_ipix.py       # IPIX training + evaluation
├── reports/                 # Documentation
│   └── GDCM_DETECTOR_REPORT.md  # Design & ablation report
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── REMOVED.md              # Old code cleanup record
```

## Key Results (200 epochs, 3 seeds)

**Protocol v1.1**: φ0~U(-π,π) per sample; unweighted CE (paper Eq.16); FAR via
sorting ALL training clutter cells, h = o_sorted[ceil(α_f·N_c)] (Eq.15).

**v1.1 实测结果（GDCM-A0，256-bin 裁剪，N_c=6,093,276）:**

| Model | AUC | PdL@1e-4 | PdL@1e-3 | PdL@1e-2 | Params |
|-------|:---:|:---:|:---:|:---:|:---:|
| **E4-D2-GDCM-A0 (final)** | **0.9666±0.005** | **0.484** | **0.558** | **0.627** | 4,425,954 |
| ST-GNN (Fig.9 复现) | 0.9852±0.010 | 0.228 | 0.370 | 0.616 | 5,066,274 |

*旧协议（weighted CE + 800 校准样本）历史结果见 `reports/GDCM_WHY_DS_REPORT.md`。*

**分析**：v1.1（普通 CE + 全量杂波 FAR）下，GDCM-A0 的 AUC 低于 ST-GNN，但在低 SCR 段的 Pd 显著更高（如 PFA=1e-4 时 Pd(-24dB)=0.194 vs 0.000、Pd(-20dB)=0.453 vs 0.011），且 seed 方差更小（±0.005 vs ±0.010）。ST-GNN 依靠 SCR≥-18dB 后更陡峭的 Pd 曲线取得更高 AUC。完整对比见 `reports/V11_GDCM_A0_VS_STGNN.md`。
