import torch
import torch.nn as nn
import torch.nn.functional as F

class GdcmDetector(nn.Module):
    """Selected GDCM-A0 detector: log1p(abs(U_d-U_h)) only."""

    def __init__(self, in_ch=1216, variant='a0', dropout=0.1, G=8, eps=1e-06):
        super().__init__()
        if variant != 'a0':
            raise ValueError('Only selected GDCM-A0 is supported')
        self.variant = variant
        self.G = G
        self.eps = eps
        self.c1 = nn.Conv1d(in_ch, 256, 1)
        self.d3 = nn.Conv1d(256, 256, 3, padding=1, groups=256)
        self.d7 = nn.Conv1d(256, 256, 7, padding=3, groups=256)
        self.d15 = nn.Conv1d(256, 256, 15, padding=7, groups=256)
        self.fuse = nn.Conv1d(768, 512, 1)
        self.proj_d = nn.Sequential(nn.Conv1d(256, 128, 1), nn.GroupNorm(8, 128))
        self.proj_h = nn.Sequential(nn.Conv1d(512, 128, 1), nn.GroupNorm(8, 128))
        cin = 128
        self.cls = nn.Sequential(nn.Conv1d(cin, 128, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(128, 2, 1))

    def forward(self, f):
        eps = self.eps
        F_d = self.c1(f)
        b3 = self.d3(F_d)
        b7 = self.d7(F_d)
        b15 = self.d15(F_d)
        H_ms = F.gelu(self.fuse(torch.cat([b3, b7, b15], 1)))
        U_d = self.proj_d(F_d)
        U_h = self.proj_h(H_ms)
        D = U_d - U_h
        D_a = torch.log1p(torch.abs(D))
        return self.cls(D_a)
