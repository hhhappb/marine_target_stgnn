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

## Key Results (200 epochs, 3 seeds, protocol v1.1, all weighted CE)

| Model | AUC | PdL@1e-3 | Params |
|-------|:---:|:---:|:---:|
| G0 E4-D2-RDJ (baseline) | 0.9812 | 0.540 | 4,442,338 |
| **A0 D_a only (final default)** | **0.9842±0.002** | **0.600** | 4,425,954 |
| G3 GroupDir + Log | 0.9830 | 0.600 | 4,442,338 |

**A0 (channel-wise log magnitude) is the final adopted setting**: it achieves
the highest AUC (0.9842) with the **smallest seed variance (±0.002)**, and
the best PdL at 1e-3/1e-2 (0.600 / 0.718, tied with G3) — the simplest and
most stable judgement representation. Full ablation in
`reports/GDCM_WHY_DS_REPORT.md` and `reports/GDCM_DETECTOR_REPORT.md`.
