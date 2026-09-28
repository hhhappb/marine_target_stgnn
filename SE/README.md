# SE: Selected P=8 E4 with Radar-Prior SFE + A1 Pulse-Attention TFE1

**SE** is the *selected P=8* model: the same E4 **parallel-representation
backbone** as `e4d2_gdcm` (ST-GNN 1024ch + PhaseEncoder 128 + AmpEncoder 64 +
CVOCAFusion 64 + global MLP 128 = **1216ch**) and the same **GDCM-A0** detector
head, but with the **selected spatiotemporal slots** in the main branch:

* `spatial1` / `spatial2` = **`radar_prior_dynamic_sfe`** — static distance
  decay prior (γ=0.5) blended 0.7/0.3 with a slow-time dynamic graph whose
  cosine edges are **restricted to a local radius δ=5** (temperature 0.2);
* `temporal1` = **`pulse_attention_only_tfe`** (A1) — a per-range-cell P×P
  slow-time attention residual, 4 heads, dim 64, fixed `residual_scale=0.1`;
* `temporal2` = **`stgnn_tfe`** — the original gated temporal extractor.

**Final adopted setting (`gdcm_a0`)**: the judgement head consumes ONLY

```
Δ  = U_d − U_h
D_a = log(1 + |Δ|)                       [B,128,14]
D_a → Conv1d(128→128) → GELU → Dropout(0.1) → Conv1d(128→2)
```

Fixed operating point: **P = 8 pulses, N = 14 range cells** (IPIX v2.x),
**4,461,538 parameters**. Class labels are `[B,2,14]` (channel 0 = clutter,
channel 1 = target).

## Self-Contained Package

This folder is a **complete, self-contained model package**:

1. This `SE/` folder (model core in `selected_e4/`, plus utils/losses/configs/experiments);
2. The IPIX `window8_stride4_related` `.npz` windows;
3. The Python dependencies in `requirements.txt`.

No other project code is required. The canonical model core is the untouched
`selected_e4/` package — it is re-exported, never copied or modified.

## Architecture (final A0)

```
[I/P99, Q/P99, PL, RPH/P99]  x_main        [B, 4, 8, 14]
[temporal_mean, std, stab]   x_phase_cell  [B, 3, 14]
[log1p(|X|/P99)]             x_amp_map     [B, 1, 8, 14]
[amp_mean, std, p95, pr]     x_amp_cell    [B, 4, 14]
[global rms/p95/p99/max…]    x_global      [B, 6]
        │
        ├─ main branch: ft1/ft2 → SFE(rp-dyn) → TFE1(A1 attn) → SFE → TFE2 → mean(P) → 1024ch
        ├─ AmpEncoder  → 64ch ┐
        ├─ PhaseEncoder→128ch ┴─ CVOCAFusion → 64ch
        └─ global MLP  → 128ch (broadcast)
        │
   concat = 1216ch → GdcmDetector(a0) → [B, 2, 14] logits
```

## Variants / Slot comparison vs `e4d2_gdcm`

| Slot | **SE (this package)** | `e4d2_gdcm` (classic) |
|---|---|---|
| spatial1 | `radar_prior_dynamic_sfe` | `STFE` (GAT) |
| temporal1 | `pulse_attention_only_tfe` (A1) | `TFE` (gated conv) |
| spatial2 | `radar_prior_dynamic_sfe` | `STFE` (GAT) |
| temporal2 | `stgnn_tfe` | `TFE` (gated conv) |
| detector | `gdcm_a0` | `gdcm_a0` |
| encoders / CVOCA / global | identical | identical |

Everything **except the four spatiotemporal slots is shared** with
`e4d2_gdcm`; SE changes no encoder, fusion, global MLP or detector code.
See `reports/ARCHITECTURE_AND_DIFF.md` for the full module-correspondence table.

## Quick Start

```bash
cd /tmp/ST-GNN/SE
pip install -r requirements.txt

# forward-shape / interface smoke check (prints [4, 2, 14])
python example.py

# IPIX v2.x single-condition training + evaluation (SE P=8)
PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.train_ipix \
    --conditions 19931109_191449_starea:hh --epochs 60 \
    --output checkpoints/se_ipix.json

# v2.1 / PAX-L1 quick-8 screening, or the full 56-point cohort
PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.run_v21_ipix --epochs 60
PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.run_v21_ipix --all-56 \
    --output checkpoints/v21_se_selected_e4_p8.json
```

### Inference

```python
from SE import E4Model, E4D2Preprocessor          # or: from selected_e4 import ...
# from utils.inference import load_model, run_inference, calibrate_threshold, detect

# 1) shape-only example
import numpy as np, torch
rng = np.random.default_rng(42)
echoes = (rng.normal(size=(4,8,14)) + 1j*rng.normal(size=(4,8,14))).astype(np.complex64)
prep  = E4D2Preprocessor().fit(echoes)            # fit on TRAINING echoes only
inputs = {k: torch.from_numpy(v) for k, v in prep.transform(echoes).items()}
model = E4Model().eval()
with torch.no_grad():
    logits = model(**inputs)                      # [4, 2, 14]

# 2) with a trained checkpoint
# model, prep = load_model('checkpoints/19931109_191449_starea__hh.pt')
# probs = run_inference(model, prep, X_complex)            # [B, 14] target prob
# h     = calibrate_threshold(model, prep, clutter)        # Eq.15 (train clutter)
# det   = detect(model, prep, X_complex, h)                # [B, 14] bool
```

## Package Structure

```
SE/
├── README.md                 # this file
├── requirements.txt          # Python dependencies
├── __init__.py               # E4Model, E4D2Preprocessor, count_params
├── example.py                # forward-shape smoke check
├── MANIFEST.json             # sha256 manifest of package files
├── VERIFICATION.json         # recorded verification results
├── selected_e4/              # CANONICAL model core (untouched)
│   ├── model.py              # E4Model
│   ├── e4_backbone.py        # E4Backbone (1216ch)
│   ├── encoders.py           # PhaseEncoder / AmpEncoder / CVOCAFusion
│   ├── stgnn_backbone.py     # ConfiguredSTGNNBackbone (4 slots)
│   ├── registry.py           # slot builders
│   ├── spatial.py            # RadarPriorDynamicSFE        (SE spatial)
│   ├── temporal_attention.py # PulseAttentionOnlyTFE       (SE TFE1 / A1)
│   ├── temporal_gate.py      # STGNNTemporalGate           (SE TFE2)
│   ├── detector.py           # GdcmDetector (A0)
│   ├── preprocess.py         # E4D2Preprocessor
│   ├── common.py             # RangeAttentionBase
│   └── config.json           # fixed selected slots
├── models/                   # re-exports of selected_e4 + count_params
├── utils/                    # preprocess re-export, IPIX data, inference
│   ├── data.py               # windowed npz reader, primary_strict labels
│   └── inference.py          # load_model / run_inference / calibrate / detect
├── losses/                   # CE, weighted CE, Focal, get_loss
├── configs/                  # default.yaml / ipix.yaml / sdrdsp.yaml
├── experiments/              # train_ipix.py, run_v21_ipix.py
├── reports/                  # ARCHITECTURE_AND_DIFF.md, VERIFICATION.md
└── checkpoints/              # .gitkeep (weights are not shipped)
```

## Verification

`selected_e4/` is the state-dict-verified core. As recorded in
`VERIFICATION.json` (unchanged by this package):

* strictly `state_dict`-compatible with **56** E4 checkpoints;
* preprocessing numerically **equal** to the experiment pipeline;
* evaluation logits **exactly equal** to the experiment model;
* forward shape `[4,2,14]`, finite backward.

See `reports/VERIFICATION.md` for the reproduction commands.

## Key Results (historical references)

SE is the selected slot configuration of the E4 + GDCM-A0 family. The
**same-family baseline** (`e4d2_gdcm`, classic STFE/TFE slots) was measured
under the v2.1 / PAX-L1 IPIX protocol (seed 42, P_F = 1e-3, primary_strict,
train-clutter Eq.15 threshold + temperature scaling):

| Model (v2.1 PAX-L1, IPIX) | 56-point PD | 56-point PF_all | quick-8 PD | quick-8 PF_all |
|---|:---:|:---:|:---:|:---:|
| E4-D2-GDCM-A0 (`e4d2_gdcm`, classic slots) | 0.9565 | 0.00492 | 0.9055 | 0.00632 |
| ST-GNN (windowed) | 0.9368 | 0.00348 | 0.8175 | 0.00467 |

*Source: `/tmp/ST-GNN/V21_FIVE_MODELS_DETAILED_REPORT.md`; per-point results in
`e4d2_gdcm/checkpoints/v21_e4d2_gdcm_a0_p8*.json`. These belong to the
classic-slot E4-D2-GDCM-A0, **not** to SE.*

SE itself swaps the four spatiotemporal slots for the selected ones above.
Its own end-to-end IPIX numbers are reproducible with the runner in this
package but are **not** shipped here as a result file:

```bash
PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.run_v21_ipix --all-56 \
    --output checkpoints/v21_se_selected_e4_p8.json
```

The slot provenance (radar-prior dynamic SFE and the A1 pulse-attention TFE1)
and their SDRDSP P8/N256 diagnostics are documented in
`marine_target_stgnn-main/.../docs/selected_spatial_a1_integration.md` and
`.../reports/` (SDRDSP is a different operating point; see `configs/sdrdsp.yaml`).

## Notes and limits

* `E4Model` is compiled for **P = 8**; other pulse counts require a separate
  model build (see `configs/sdrdsp.yaml`).
* Preprocessor statistics must be fit on **training** echoes only; test-time
  thresholds/temperature come from training-side calibration.
* This package ships no weights; `checkpoints/` holds only a `.gitkeep`.
