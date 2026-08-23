"""E4-D2: CVOCA-Fusion ST-GNN with Local-CFAR Detector Head.

Architecture:
  Input:  [I/P99, Q/P99, PL_raw, RPH/P99]           → 4ch main
          [temporal_mean, temporal_std, phase_stab]   → 3ch phase cell
          [amp_map log1p(|X|/P99)]                    → 1ch amp map
          [amp_mean, amp_std, amp_p95, amp_peak_ratio] → 4ch amp cell
          [global_rms, p95, p99, max, mean, std]      → 6ch global

  Backbone: STGNN(4→1024) + PhaseEncoder(3→128) + AmpEncoder((1,4)→64)
            + CVOCAFusion(A, phase) → 64ch
            + Global MLP(6→128) broadcast
            → concat [1024+64+128] = 1216ch

  Detector (D2 Local-CFAR):
    Conv1d(1216→256) → 3× DepthwiseConv(k=3/7/15, groups=256)
    → concat(768) → Conv1d(768→512) → GELU → Conv1d(512→2)

Parameters: ~4.31M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from e4d2_gdcm.models.stgnn_backbone import GraphAttentionLayer, STFE, TFE, STGNNBackbone


# ────────────────────────────────────────────────────────────
#  Phase Encoder
# ────────────────────────────────────────────────────────────

class PhaseEncoder(nn.Module):
    """Encode 3ch phase cell stats: [temporal_mean, temporal_std, phase_stability]."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 64, 3, padding=1)
        self.c2 = nn.Conv1d(64, 128, 3, padding=1)
        self.r = nn.ReLU()

    def forward(self, x):
        return self.r(self.c2(self.r(self.c1(x))))  # [B, 128, N]


# ────────────────────────────────────────────────────────────
#  Amplitude Encoder
# ────────────────────────────────────────────────────────────

class AmpEncoder(nn.Module):
    """Encode amplitude map [B,1,4,N] + amp cell stats [B,4,N] → [B,64,N]."""

    def __init__(self, P=4):
        super().__init__()
        self.map_c1 = nn.Conv2d(1, 16, (1, 3), padding=(0, 1))
        self.map_c2 = nn.Conv2d(16, 32, (P, 1), padding=0)
        self.map_r = nn.ReLU()
        self.cell_c1 = nn.Conv1d(4, 32, 3, padding=1)
        self.cell_c2 = nn.Conv1d(32, 32, 3, padding=1)
        self.cell_r = nn.ReLU()
        self.fuse = nn.Conv1d(64, 64, 3, padding=1)
        self.fuse_r = nn.ReLU()

    def forward(self, amp_map, amp_cell):
        a = self.map_r(self.map_c1(amp_map))
        a = self.map_r(self.map_c2(a))
        a = a.squeeze(2)           # [B, 32, N]
        c = self.cell_r(self.cell_c1(amp_cell))
        c = self.cell_r(self.cell_c2(c))  # [B, 32, N]
        return self.fuse_r(self.fuse(torch.cat([a, c], 1)))  # [B, 64, N]


# ────────────────────────────────────────────────────────────
#  Raw Polar Encoders
# ────────────────────────────────────────────────────────────

class RawPhaseEncoder(nn.Module):
    """Encode raw polar phase direction [B,2,P,R] → [B,128,R].

    [cosθ, sinθ] ∈ R^{B×2×4×R}
      → Conv2d(2→32, k=(3,1))      → [B,32,4,R]
      → Conv2d(32→64, k=(3,1), s=2) → [B,64,2,R]
      → Conv2d(64→128, k=(3,1), s=2) → [B,128,1,R]
      → squeeze temporal → F_phase [B,128,R]

    The network learns pulse-to-pulse phase evolution itself (4→2→1)
    instead of using hand-crafted temporal_mean/std/stability statistics.
    """

    def __init__(self, P=4):
        super().__init__()
        self.c1 = nn.Conv2d(2, 32, (3, 1), padding=(1, 0))
        self.c2 = nn.Conv2d(32, 64, (3, 1), stride=(2, 1), padding=(1, 0))
        self.c3 = nn.Conv2d(64, 128, (3, 1), stride=(2, 1), padding=(1, 0))
        self.g = nn.GELU()

    def forward(self, x):
        """x: [B, 2, P, R]  (cosθ, sinθ per pulse per range bin)"""
        x = self.g(self.c1(x))      # [B, 32, P, R]
        x = self.g(self.c2(x))      # [B, 64, P//2, R]
        x = self.c3(x)              # [B, 128, P//4, R]
        return x.squeeze(2)         # [B, 128, R]


class RawAmpEncoder(nn.Module):
    """Encode raw polar amplitude [B,1,P,R] → [B,64,R].

    A ∈ R^{B×1×4×R}
      → Conv2d(1→16, k=(3,1))       → [B,16,4,R]
      → Conv2d(16→32, k=(3,1), s=2) → [B,32,2,R]
      → Conv2d(32→64, k=(3,1), s=2) → [B,64,1,R]
      → squeeze temporal → F_amp [B,64,R]

    The network learns amplitude evolution across the 4 pulses itself.
    """

    def __init__(self, P=4):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, (3, 1), padding=(1, 0))
        self.c2 = nn.Conv2d(16, 32, (3, 1), stride=(2, 1), padding=(1, 0))
        self.c3 = nn.Conv2d(32, 64, (3, 1), stride=(2, 1), padding=(1, 0))
        self.g = nn.GELU()

    def forward(self, x):
        """x: [B, 1, P, R]  (amplitude per pulse per range bin)"""
        x = self.g(self.c1(x))      # [B, 16, P, R]
        x = self.g(self.c2(x))      # [B, 32, P//2, R]
        x = self.c3(x)              # [B, 64, P//4, R]
        return x.squeeze(2)         # [B, 64, R]


# ────────────────────────────────────────────────────────────
#  CVOCA Fusion
# ────────────────────────────────────────────────────────────

class CVOCAFusion(nn.Module):
    """Complex-Valued Coupled Fusion: A · exp(jθ) polar reconstruction.

    F_amp [B,64,N] → A = softplus(proj)           (non-negative magnitude)
    F_phase [B,128,N] → C,S = L2_norm(cos,sin)    (unit direction)
    Z_real = A·C, Z_imag = A·S                     (polar reconstruction)
    Y = ComplexConv([Z_real, Z_imag])              (coupled fusion)
    output = [Y_real, Y_imag, |Y|]                  (3×64ch → 64ch)
    """

    def __init__(self):
        super().__init__()
        self.phase_proj = nn.Conv1d(128, 64, 1)
        self.amp_proj = nn.Conv1d(64, 64, 1)
        self.cos_proj = nn.Conv1d(64, 64, 1)
        self.sin_proj = nn.Conv1d(64, 64, 1)
        self.conv_R = nn.Conv1d(64, 64, 3, padding=1)
        self.conv_I = nn.Conv1d(64, 64, 3, padding=1)
        self.out = nn.Conv1d(192, 64, 1)

    def forward(self, F_amp, F_phase):
        eps = 1e-10
        Fp = self.phase_proj(F_phase)        # [B, 64, N]
        A = F.softplus(self.amp_proj(F_amp))  # [B, 64, N], A ≥ 0

        Cr = self.cos_proj(Fp)
        Sr = self.sin_proj(Fp)
        n = torch.sqrt(Cr ** 2 + Sr ** 2 + eps)
        C, S = Cr / n, Sr / n                # L2-normalized direction

        Zr = A * C                           # polar real
        Zi = A * S                           # polar imag

        Yr = self.conv_R(Zr) - self.conv_I(Zi)   # complex conv real
        Yi = self.conv_R(Zi) + self.conv_I(Zr)   # complex conv imag
        Ym = torch.sqrt(Yr ** 2 + Yi ** 2 + eps)  # magnitude

        return self.out(torch.cat([Yr, Yi, Ym], 1))  # [B, 64, N]


# ────────────────────────────────────────────────────────────
#  E4 Backbone  (stitches all branches)
# ────────────────────────────────────────────────────────────

class E4Backbone(nn.Module):
    """Complete E4 feature extractor → [B, 1216, N]."""

    def __init__(self):
        super().__init__()
        self.stgnn = STGNNBackbone(4)       # 1024ch
        self.penc = PhaseEncoder()          # 128ch
        self.aenc = AmpEncoder()            # 64ch
        self.cvoca = CVOCAFusion()          # 64ch (fused)
        self.gf1 = nn.Linear(6, 32)
        self.gf2 = nn.Linear(32, 128)
        self.gr = nn.ReLU()

    def forward(self, x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global):
        N = x_main.size(3)
        st = self.stgnn(x_main)                      # [B, 1024, N]
        Fa = self.aenc(x_amp_map, x_amp_cell)         # [B, 64, N]
        Fp = self.penc(x_phase_cell)                   # [B, 128, N]
        Fap = self.cvoca(Fa, Fp)                       # [B, 64, N]

        g = self.gr(self.gf2(self.gr(self.gf1(x_global))))
        g = g.unsqueeze(-1).expand(-1, -1, N)          # [B, 128, N]

        return torch.cat([st, Fap, g], 1)              # [B, 1216, N]


class RawPolarBackbone(nn.Module):
    """Raw-Polar E4 feature extractor → [B, 1216, N].

    Replaces statistical PhaseEncoder/AmpEncoder with raw-polar encoders:
      x_polar_amp   [B,1,P,N]  A=|X| (amplitude per pulse)
      x_polar_phase [B,2,P,N]  [cosθ, sinθ] = [I/A, Q/A] (unit direction)
    Network learns pulse evolution itself (4→2→1 temporal convs).
    All downstream (CVOCA, Global, detector) unchanged.
    """

    def __init__(self):
        super().__init__()
        self.stgnn = STGNNBackbone(4)       # 1024ch (I/Q+PL+RPH main branch)
        self.penc = RawPhaseEncoder()       # 128ch
        self.aenc = RawAmpEncoder()         # 64ch
        self.cvoca = CVOCAFusion()          # 64ch (fused)
        self.gf1 = nn.Linear(6, 32)
        self.gf2 = nn.Linear(32, 128)
        self.gr = nn.ReLU()

    def forward(self, x_main, x_polar_amp, x_polar_phase, x_global):
        N = x_main.size(3)
        st = self.stgnn(x_main)                      # [B, 1024, N]
        Fa = self.aenc(x_polar_amp)                   # [B, 64, N]
        Fp = self.penc(x_polar_phase)                 # [B, 128, N]
        Fap = self.cvoca(Fa, Fp)                      # [B, 64, N]

        g = self.gr(self.gf2(self.gr(self.gf1(x_global))))
        g = g.unsqueeze(-1).expand(-1, -1, N)         # [B, 128, N]

        return torch.cat([st, Fap, g], 1)             # [B, 1216, N]


# ────────────────────────────────────────────────────────────
#  Detector Heads
# ────────────────────────────────────────────────────────────

class D0Detector(nn.Module):
    """Original ST-GNN detector: Conv1d(1216→512→2)."""

    def __init__(self, in_ch=1216):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 512, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(512, 2, 1)

    def forward(self, f):
        return self.conv2(self.relu(self.conv1(f)))


class D2Detector(nn.Module):
    """Multi-scale Local-CFAR Detector Head.

    1216ch → Conv1d(k=1→256)
    ├── DepthwiseConv(k=3,  groups=256)  → local contrast
    ├── DepthwiseConv(k=7,  groups=256)  → neighbourhood stats
    └── DepthwiseConv(k=15, groups=256)  → regional background
    → concat(768) → Conv1d(768→512) → GELU → Conv1d(512→2)
    """

    def __init__(self, in_ch=1216):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, 256, 1)
        self.d3  = nn.Conv1d(256, 256, 3,  padding=1,  groups=256)
        self.d7  = nn.Conv1d(256, 256, 7,  padding=3,  groups=256)
        self.d15 = nn.Conv1d(256, 256, 15, padding=7,  groups=256)
        self.fuse = nn.Conv1d(768, 512, 1)
        self.c2  = nn.Conv1d(512, 2, 1)

    def forward(self, f):
        h = self.c1(f)
        b3 = self.d3(h)
        b7 = self.d7(h)
        b15 = self.d15(h)
        return self.c2(F.gelu(self.fuse(torch.cat([b3, b7, b15], 1))))


class FdGateDetector(nn.Module):
    """D2-FdGate: gated residual F_d injection, minimal change over D2.

    Preserves the exact D2 main path (768→512→2), adds only:
      F_d → Conv1d(256→512) → F'_d
      Z = H_ms + λ · F'_d        (λ learnable, init=0)
    When λ=0, model is identical to E4-D2-old.
    """

    def __init__(self, in_ch=1216, lambda_init=0.0):
        super().__init__()
        # ── Original D2 path (unchanged) ──
        self.c1 = nn.Conv1d(in_ch, 256, 1)
        self.d3  = nn.Conv1d(256, 256, 3,  padding=1,  groups=256)
        self.d7  = nn.Conv1d(256, 256, 7,  padding=3,  groups=256)
        self.d15 = nn.Conv1d(256, 256, 15, padding=7,  groups=256)
        self.fuse = nn.Conv1d(768, 512, 1)
        self.c2   = nn.Conv1d(512, 2, 1)

        # ── F_d gate branch (minimal addition) ──
        self.gate_proj = nn.Conv1d(256, 512, 1)          # F_d → F'_d
        self.lambda_ = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, f):
        F_d = self.c1(f)                                  # [B, 256, R]

        # Original D2 path
        b3  = self.d3(F_d)
        b7  = self.d7(F_d)
        b15 = self.d15(F_d)
        H_ms = F.gelu(self.fuse(torch.cat([b3, b7, b15], 1)))  # [B, 512, R]

        # Gated F_d injection
        F_prime = self.gate_proj(F_d)                     # [B, 512, R]
        Z = H_ms + self.lambda_ * F_prime                 # [B, 512, R]

        return self.c2(Z)                                  # [B, 2, R]


class FdLogitDetector(nn.Module):
    """D2-FdLogit: logit-level F_d fusion, safest minimal change.

    Preserves the exact D2 output. Adds a separate small head for F_d:
      logits_H = Head_H(H_ms)           ← original D2
      logits_F = Head_F(F_d)            ← auxiliary head (256→128→64→2)
      logits = logits_H + β · logits_F  (β learnable, init=0)
    When β=0, model is identical to E4-D2-old.
    """

    def __init__(self, in_ch=1216, beta_init=0.0):
        super().__init__()
        # ── Original D2 path (unchanged) ──
        self.c1 = nn.Conv1d(in_ch, 256, 1)
        self.d3  = nn.Conv1d(256, 256, 3,  padding=1,  groups=256)
        self.d7  = nn.Conv1d(256, 256, 7,  padding=3,  groups=256)
        self.d15 = nn.Conv1d(256, 256, 15, padding=7,  groups=256)
        self.fuse = nn.Conv1d(768, 512, 1)
        self.c2   = nn.Conv1d(512, 2, 1)

        # ── F_d auxiliary head (small, separate) ──
        self.f_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 2, 1),
        )
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, f):
        F_d = self.c1(f)                                  # [B, 256, R]

        # Original D2 path
        b3  = self.d3(F_d)
        b7  = self.d7(F_d)
        b15 = self.d15(F_d)
        H_ms = F.gelu(self.fuse(torch.cat([b3, b7, b15], 1)))  # [B, 512, R]
        logits_H = self.c2(H_ms)                          # [B, 2, R]

        # F_d auxiliary head
        logits_F = self.f_head(F_d)                       # [B, 2, R]

        return logits_H + self.beta * logits_F             # [B, 2, R]


class FdHDetector(nn.Module):
    """DEPRECATED: D2-FdH (old version, kept for backward compat).

    Replaced by FdGateDetector and FdLogitDetector which make minimal
    changes to the original D2 path.
    """
    # Alias to FdGateDetector for backward compatibility
    pass


class RdjDetector(nn.Module):
    """D2-RDJ: Relative Difference Judgment detector.

    F_d → DWConv(3/7/15) → H_ms                        (unchanged D2 backbone)
    F_d → Proj_d(256→128) → GN → U_d                   (cell feature space)
    H_ms → Proj_h(512→128) → GN → U_h                  (background space)
    D_s = U_d − U_h,  D_a = |U_d − U_h|               (explicit difference)
    Concat[D_s, D_a] → Conv(256→128) → GELU → Dropout → Conv(128→2)

    The classifier sees ONLY the difference features D_s, D_a — it cannot
    access raw F_d or H_ms directly. This forces the network to learn
    "how different is this cell from its local background?"
    """

    def __init__(self, in_ch=1216, dropout=0.1):
        super().__init__()
        # ── D2 multi-scale backbone (unchanged) ──
        self.c1 = nn.Conv1d(in_ch, 256, 1)
        self.d3  = nn.Conv1d(256, 256, 3,  padding=1,  groups=256)
        self.d7  = nn.Conv1d(256, 256, 7,  padding=3,  groups=256)
        self.d15 = nn.Conv1d(256, 256, 15, padding=7,  groups=256)
        self.fuse = nn.Conv1d(768, 512, 1)

        # ── Projection to comparison space ──
        self.proj_d = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.GroupNorm(8, 128),
        )
        self.proj_h = nn.Sequential(
            nn.Conv1d(512, 128, 1),
            nn.GroupNorm(8, 128),
        )

        # ── Difference classifier ──
        self.cls = nn.Sequential(
            nn.Conv1d(256, 128, 1),        # 128(signed) + 128(abs) = 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, f):
        # D2 backbone
        F_d = self.c1(f)                                 # [B, 256, R]
        b3  = self.d3(F_d)
        b7  = self.d7(F_d)
        b15 = self.d15(F_d)
        H_ms = F.gelu(self.fuse(torch.cat([b3, b7, b15], 1)))  # [B, 512, R]

        # Project to comparison space
        U_d = self.proj_d(F_d)                           # [B, 128, R]
        U_h = self.proj_h(H_ms)                          # [B, 128, R]

        # Explicit differences
        D_s = U_d - U_h                                  # [B, 128, R] signed
        D_a = torch.abs(D_s)                             # [B, 128, R] absolute

        # Classify from differences only
        D_cat = torch.cat([D_s, D_a], dim=1)             # [B, 256, R]
        return self.cls(D_cat)                            # [B, 2, R]


class GdcmDetector(nn.Module):
    """GDCM: Group-Directional + Channel-wise Magnitude detector.

    Keeps U_d, U_h ∈ [B,128,R] (GroupNorm 8 groups) and splits the 128
    channels into G=8 semantic groups of C_g=16:

      Δ = U_d − U_h
      ρ_g = RMS(Δ_g) = sqrt((1/16)·Σ_{c∈g} Δ²_c + ε)     [B,8,R]
      D_s,c = Δ_c / (ρ_{g(c)} + ε)                        [B,128,R]
          → group-normalised direction: tells WHICH channels inside a
            group are enhanced/suppressed; scale-invariant (Δ_g → k·Δ_g
            leaves D_s unchanged), so intensity is truly removed.
      D_a (variant):
        a0: log(1 + |Δ|) only (NO D_s)                    [B,128,R]  ← FINAL
        g1: |Δ|                                           [B,128,R]
        g2: |Δ| / (|U_d| + |U_h| + ε)                     [B,128,R] relative
        g3: log(1 + |Δ|)                                  [B,128,R]
        g4: ρ_g (group magnitude)                         [B,8,R]
      g0: D_s = Δ, D_a = |Δ|   (identical to RDJ baseline)

    Classifier is EXACTLY the RDJ head:
      concat[D_s, D_a] (256 or 136 ch) → Conv1d→128 → GELU → Dropout(0.1)
      → Conv1d(128→2)
    """

    def __init__(self, in_ch=1216, variant='a0', dropout=0.1, G=8, eps=1e-6):
        super().__init__()
        assert variant in ('g0', 'g1', 'g2', 'g3', 'g4', 'a0'), f"Unknown GDCM variant: {variant}"
        self.variant = variant
        self.G = G
        self.eps = eps

        # ── D2 multi-scale backbone (unchanged) ──
        self.c1 = nn.Conv1d(in_ch, 256, 1)
        self.d3  = nn.Conv1d(256, 256, 3,  padding=1,  groups=256)
        self.d7  = nn.Conv1d(256, 256, 7,  padding=3,  groups=256)
        self.d15 = nn.Conv1d(256, 256, 15, padding=7,  groups=256)
        self.fuse = nn.Conv1d(768, 512, 1)

        # ── Projection to comparison space (unchanged) ──
        self.proj_d = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.GroupNorm(8, 128),
        )
        self.proj_h = nn.Sequential(
            nn.Conv1d(512, 128, 1),
            nn.GroupNorm(8, 128),
        )

        # ── Classifier (identical to RDJ head) ──
        if variant == 'a0':
            cin = 128                       # D_a only (log(1+|Δ|))
        elif variant == 'g4':
            cin = 136                       # D_s + group magnitude
        else:
            cin = 256
        self.cls = nn.Sequential(
            nn.Conv1d(cin, 128, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 2, 1),
        )

    def _group_rho(self, D):
        """Per-group RMS of D over the 16 channels of each group → [B,G,R]."""
        B, C, R = D.shape
        G = self.G
        Cg = C // G                       # 16
        Dg = D.view(B, G, Cg, R)
        rho = torch.sqrt((Dg ** 2).mean(dim=2) + self.eps)   # [B,G,R]
        return rho, Cg

    def forward(self, f):
        eps = self.eps
        F_d = self.c1(f)                                 # [B, 256, R]
        b3  = self.d3(F_d)
        b7  = self.d7(F_d)
        b15 = self.d15(F_d)
        H_ms = F.gelu(self.fuse(torch.cat([b3, b7, b15], 1)))  # [B, 512, R]

        U_d = self.proj_d(F_d)                           # [B, 128, R]
        U_h = self.proj_h(H_ms)                          # [B, 128, R]

        D = U_d - U_h                                    # [B, 128, R]

        if self.variant == 'a0':
            # A0 (final): channel-wise log magnitude only — simplest & most stable
            D_a = torch.log1p(torch.abs(D))              # [B, 128, R]
            return self.cls(D_a)
        if self.variant == 'g0':
            D_s = D
            D_a = torch.abs(D)
        else:
            rho_g, Cg = self._group_rho(D)               # [B,G,R]
            rho_128 = rho_g.repeat_interleave(Cg, dim=1)  # [B,128,R]
            D_s = D / (rho_128 + eps)                    # group-normalised direction

            if self.variant == 'g1':
                D_a = torch.abs(D)
            elif self.variant == 'g2':
                D_a = torch.abs(D) / (torch.abs(U_d) + torch.abs(U_h) + eps)
            elif self.variant == 'g3':
                D_a = torch.log1p(torch.abs(D))
            else:  # g4: group magnitude [B,G,R]
                D_a = rho_g

        x = torch.cat([D_s, D_a], dim=1)                 # [B,256,R] or [B,136,R]
        return self.cls(x)                                # [B, 2, R]


class E4D2(nn.Module):
    """Complete E4-D2 model: E4Backbone + detector head.

    Detector options:
      'rdj'     (default) : relative difference judgment (D_s, D_a)
      'gdcm_a0'           : A0 final — D_a only log(1+|Δ|)  → 128ch ← DEFAULT (final)
      'gdcm_g0'           : GDCM baseline  (Δ, |Δ|)          → 256ch
      'gdcm_g1'           : GDCM group-dir + |Δ|             → 256ch
      'gdcm_g2'           : GDCM group-dir + relative |Δ|    → 256ch
      'gdcm_g3'           : GDCM group-dir + log(1+|Δ|)      → 256ch
      'gdcm_g4'           : GDCM group-dir + group mag       → 136ch
      'fdlogit'           : logit-level F_d injection
      'd2'                : original D2 (H_ms only)
      'fdgate'            : gated residual F_d injection
      'd0'                : original ST-GNN detector

    Parameters: ~4.35M (rdj), ~4.31M (d2)
    """

    def __init__(self, detector: str = 'gdcm_a0'):
        super().__init__()
        self.backbone = E4Backbone()
        if detector == 'rdj':
            self.detector = RdjDetector(1216)
        elif detector.startswith('gdcm_'):
            variant = detector.split('_')[1]
            self.detector = GdcmDetector(1216, variant=variant)
        elif detector == 'd2':
            self.detector = D2Detector(1216)
        elif detector == 'fdgate':
            self.detector = FdGateDetector(1216)
        elif detector == 'fdlogit':
            self.detector = FdLogitDetector(1216)
        elif detector == 'd2fdh':
            self.detector = FdGateDetector(1216)  # backward compat: old name → gate version
        elif detector == 'd0':
            self.detector = D0Detector(1216)
        else:
            raise ValueError(f"Unknown detector: {detector}")

    def forward(self, x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global):
        f = self.backbone(x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global)
        return self.detector(f)  # [B, 2, N]

    @torch.no_grad()
    def predict(self, x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global):
        """Return target probability per range bin [B, N]."""
        self.eval()
        logits = self(x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global)
        return torch.softmax(logits, 1)[:, 1, :]


class E4D2RawPolar(nn.Module):
    """Raw-Polar E4-D2: RawPolarBackbone + detector head.

    Inputs:
      x_main        [B, 4, P, N]  I/P99, Q/P99, PL_raw, RPH/P99 (main branch)
      x_polar_amp   [B, 1, P, N]  A=|X| amplitude per pulse
      x_polar_phase [B, 2, P, N]  [cosθ, sinθ] unit phase direction
      x_global      [B, 6]        global clutter statistics

    Detector options (same as E4D2):
      'rdj' (default), 'd2', 'fdgate', 'fdlogit', 'd0'
    """

    def __init__(self, detector: str = 'rdj'):
        super().__init__()
        self.backbone = RawPolarBackbone()
        if detector == 'rdj':
            self.detector = RdjDetector(1216)
        elif detector == 'd2':
            self.detector = D2Detector(1216)
        elif detector == 'fdgate':
            self.detector = FdGateDetector(1216)
        elif detector == 'fdlogit':
            self.detector = FdLogitDetector(1216)
        elif detector == 'd0':
            self.detector = D0Detector(1216)
        else:
            raise ValueError(f"Unknown detector: {detector}")

    def forward(self, x_main, x_polar_amp, x_polar_phase, x_global):
        f = self.backbone(x_main, x_polar_amp, x_polar_phase, x_global)
        return self.detector(f)  # [B, 2, N]

    @torch.no_grad()
    def predict(self, x_main, x_polar_amp, x_polar_phase, x_global):
        """Return target probability per range bin [B, N]."""
        self.eval()
        logits = self(x_main, x_polar_amp, x_polar_phase, x_global)
        return torch.softmax(logits, 1)[:, 1, :]


def count_params(model):
    """Return parameter count and human-readable string."""
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e6:
        return n, f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return n, f"{n/1e3:.1f}K"
    return n, str(n)
