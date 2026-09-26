# Selected spatial and temporal experiment code

## Scope and authorization

Prepared on 2026-09-26 from local main ee9ab6edb156f625a0d1f01796b72801751c47ec.
The user selected removal of dynamic nonlocal edges and fixed-scale A1, and authorized organizing their experiment code. No new training, commit, or push is authorized by this preparation.

## Selected implementations

- Spatial: RadarPriorDynamicSFE, already committed in 85033e0. Static distance decay remains global; dynamic cosine edges are restricted to radius 5. Static/dynamic weights 0.7/0.3, gamma 0.5, temperature 0.2. Retired dynamic_topk configuration is rejected.
- Temporal: PulseAttentionOnlyTFE, already tracked at the base revision. Only TFE1 is enhanced, attention_dim=64, heads=4, fixed residual_scale=0.1; TFE2 remains Original. No tanh cap, ratio cap, relative bias, or learnable residual scalar is added.
- IPIX A1 registry adapter: IpixPaxL1A1TFE1, extracted from the authorized IPIX experiment task. It retains the OriginalSTGNN backbone and requires P=16/N=14 and explicit fixed A1 settings.

## Reproduction support

The selected SDRDSP A0/A1 configs use v1.1 P8 seed42, 20 epochs, batch24, train-clutter o0 thresholds and minimum-training-loss checkpoint selection. The SCR dataset contract, SCR-balanced sampler, preprocessing script, protocol documents, and their existing tests are included because these configs depend on them. Default shuffle behavior remains unchanged unless scr_uniform is explicitly selected.

Config data paths are relative to the repository. This isolated checkout intentionally contains no data or historical outputs. Supply the existing dataset location explicitly when preparing a future authorized run, and use a fresh run_id; do not reuse historical save directories. No training was launched during cleanup.

## Evidence boundaries

The SDRDSP and IPIX A1 results use Original spatial modules. RadarPriorDynamicSFE plus A1 has not been validated as a combined model, and no trained-combination claim or new combination experiment is introduced here. TFE1-only results are configuration ablations, not formal two-stage temporal-only attribution.

The IPIX historical screen runner depends on the untracked E4D2 experiment framework and machine-specific historical runs. It is not copied into this integration as a standalone reproducible runner. The selected adapter is available via registry; the IPIX protocol document states the frozen data and scoring rules. Historical training results remain in their original workspaces.

## Git disposition

Branch: codex/selected-spatial-a1-clean, checkout E:/stgnn-clean-a1.
This branch starts from local main, retaining both committed changes: spatial edge simplification and retirement of cross-file IPIX splits. It does not merge the older integration branch or restore retired protocols. Existing tracked historical modules remain as repository history; rejected uncommitted candidates are not imported.

Verification: 13 existing protocol/spatial tests passed. See selected_spatial_a1_sources.json for exact copied source hashes. Model interface smoke checks cover SDRDSP P8/N256 and IPIX P16/N14. No data, weights, training logs, historical reports, or virtual environment files are included in the intended commit.
