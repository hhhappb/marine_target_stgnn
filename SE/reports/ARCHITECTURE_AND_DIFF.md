# SE — Architecture and Differences vs e4d2_gdcm

SE is the *selected P=8 E4* model: the same parallel-representation E4 backbone
as `e4d2_gdcm`, but with the **selected spatiotemporal slots** in the main
branch, followed by the same **GDCM-A0** detector head.

## 1. Shared part (identical to e4d2_gdcm)

| Component | Channels | Source |
|---|---|---|
| Input stem `ft1/ft2` (Conv2d (1,3)) | 4 → 32 → 64 | both |
| Spatial stage 1 | 64 → 128 | replaces STFE |
| Temporal stage 1 | 128 → 256 | replaces TFE |
| Spatial stage 2 | 256 → 512 | replaces STFE |
| Temporal stage 2 | 512 → 1024 | replaces TFE |
| `PhaseEncoder` | 3 → 128 | both (identical) |
| `AmpEncoder` | (1,4) → 64 | both (identical) |
| `CVOCAFusion` | 64 + 128 → 64 | both (identical) |
| Global MLP | 6 → 32 → 128 | both (identical) |
| Detector `GdcmDetector(variant='a0')` | 1216 → 256 → 512 → 128 → 2 | both (identical) |

Concatenated backbone output: `1024 (ST-GNN) + 64 (CVOCA) + 128 (global) = 1216`.
Detector A0 judgement input: `D_a = log(1 + |U_d − U_h|)`, 128ch →
`Conv1d(128→128) → GELU → Dropout(0.1) → Conv1d(128→2)`.

## 2. The only difference: spatiotemporal slots

`selected_e4/config.json` fixes the four slots:

| Slot | SE (this package) | e4d2_gdcm (classic) |
|---|---|---|
| spatial1 | `radar_prior_dynamic_sfe` (static γ=0.5, δ=5, w=0.7, T=0.2, drop 0.1) | `STFE` (GAT) |
| temporal1 | `pulse_attention_only_tfe` (A1: dim 64, heads 4, residual 0.1) | `TFE` (gated conv) |
| spatial2 | `radar_prior_dynamic_sfe` (same hyper-params) | `STFE` |
| temporal2 | `stgnn_tfe` (original gated TFE) | `TFE` |

Both models compress the slow-time axis `P=8 → 4 → 2 → mean` to `[B, 1024, N]`.

### Module mapping (SE file ↔ e4d2_gdcm file)

| Role | SE | e4d2_gdcm |
|---|---|---|
| Model entry | `selected_e4/model.py` (`E4Model`) | `models/e4d2.py` (`E4D2`) |
| Backbone stitch | `selected_e4/e4_backbone.py` | `models/e4d2.py` (`E4Backbone`) |
| Encoders / CVOCA | `selected_e4/encoders.py` | `models/e4d2.py` |
| Detector head | `selected_e4/detector.py` (`GdcmDetector`) | `models/e4d2.py` (`GdcmDetector`) |
| Main branch slots | `selected_e4/stgnn_backbone.py` + `registry.py` | `models/stgnn_backbone.py` (`STGNNBackbone`) |
| Spatial (SE) | `selected_e4/spatial.py` (`RadarPriorDynamicSFE`) | — |
| Temporal1 (SE, A1) | `selected_e4/temporal_attention.py` (`PulseAttentionOnlyTFE`) | — |
| Temporal2 (SE) | `selected_e4/temporal_gate.py` (`STGNNTemporalGate`) | `models/stgnn_backbone.py` (`TFE`) |
| Shared attention base | `selected_e4/common.py` (`RangeAttentionBase`) | — |
| Preprocessing | `selected_e4/preprocess.py` (`E4D2Preprocessor`) | `utils/preprocess.py` |
| Fixed config | `selected_e4/config.json` | `configs/default.yaml` |

## 3. Fixed operating point

| Item | Value |
|---|---|
| Pulses `P` | 8 (`E4Model` is compiled for P=8) |
| Range cells `N` | 14 (IPIX) |
| Input | pre-processed complex `[B,8,14]` (official ipixload auto) |
| Feature tensors | `x_main [B,4,8,14]`, `x_phase_cell [B,3,14]`, `x_amp_map [B,1,8,14]`, `x_amp_cell [B,4,14]`, `x_global [B,6]` |
| Output | logits `[B,2,14]` (channel 0 = clutter, channel 1 = target) |
| Parameters | **4,461,538** |

## 4. Verification conclusion

`selected_e4/` is the canonical, untouched core. It has been verified
(see `VERIFICATION.json` / `reports/VERIFICATION.md`) as:

* strictly `state_dict`-compatible with **56** E4 checkpoints;
* preprocessing numerically equal to the experiment pipeline;
* logits **exactly equal** to the experiment model for the same weights/input;
* forward shape `[4,2,14]` and finite backward.

This package only re-exports and scaffolds around `selected_e4/`; it changes
no network computation.
