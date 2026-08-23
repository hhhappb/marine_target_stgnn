"""E4-D2 (GDCM): CVOCA-Fusion ST-GNN with Group-Directional + Channel-wise Magnitude detector.

Self-contained package. All upstream modules (E4Backbone: STGNN 1024ch +
PhaseEncoder/AmpEncoder/CVOCA 64ch + Global 128ch → 1216ch) are UNCHANGED;
only the detector head is replaced with the GDCM decomposition:

  128 channels split into G=8 semantic groups of C_g=16 (matches GroupNorm):
    ρ_g = RMS(Δ_g) = sqrt((1/16)Σ_{c∈g}Δ²_c + ε)            [B,8,R]
    D_s,c = Δ_c / (ρ_{g(c)} + ε)   group-normalised direction [B,128,R]
    D_a (variant):
      g1: |Δ|;  g2: |Δ|/(|U_d|+|U_h|+ε);  g3: log(1+|Δ|);  g4: ρ_g [B,8,R]

Classifier is IDENTICAL to RDJ: concat[D_s,D_a] → Conv1d→128 → GELU →
Dropout(0.1) → Conv1d(128→2).

Detector variants:
  E4D2(detector='rdj')    baseline [Δ, |Δ|]
  E4D2(detector='gdcm_a0')  A0 final — D_a only log(1+|Δ|)   ← DEFAULT (final)
  E4D2(detector='gdcm_g0')  GDCM baseline (same as rdj)
  E4D2(detector='gdcm_g1')  group-dir + |Δ|
  E4D2(detector='gdcm_g2')  group-dir + relative |Δ|
  E4D2(detector='gdcm_g3')  group-dir + log(1+|Δ|)
  E4D2(detector='gdcm_g4')  group-dir + group magnitude (136ch)

Experiment scripts:
    python -m e4d2_gdcm.experiments.train_sdrdsp --train data/train.mat --test data/test.mat --detector gdcm_a0

Inference:
    from e4d2_gdcm.utils.inference import load_model, run_inference
    model, prep = load_model('checkpoints/e4d2_best.pt')
    probs = run_inference(model, prep, X_complex)
"""

from e4d2_gdcm.models.e4d2 import E4D2, E4D2RawPolar, GdcmDetector, count_params
from e4d2_gdcm.utils.preprocess import E4D2Preprocessor
from e4d2_gdcm.utils.data import SDRDSPBuilder

__version__ = '1.0'
__all__ = ['E4D2', 'E4D2RawPolar', 'GdcmDetector', 'E4D2Preprocessor', 'SDRDSPBuilder', 'count_params']
