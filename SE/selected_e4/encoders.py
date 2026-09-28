import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseEncoder(nn.Module):
    """Encode 3ch phase cell stats: [temporal_mean, temporal_std, phase_stability]."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 64, 3, padding=1)
        self.c2 = nn.Conv1d(64, 128, 3, padding=1)
        self.r = nn.ReLU()

    def forward(self, x):
        return self.r(self.c2(self.r(self.c1(x))))

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
        return self.fuse_r(self.fuse(torch.cat([a, c], 1)))

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

        return self.out(torch.cat([Yr, Yi, Ym], 1))
