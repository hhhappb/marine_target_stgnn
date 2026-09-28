import torch
import torch.nn as nn
from .encoders import PhaseEncoder, AmpEncoder, CVOCAFusion
from .stgnn_backbone import ConfiguredSTGNNBackbone

class E4Backbone(nn.Module):
    """Complete E4 feature extractor → [B, 1216, N]."""

    def __init__(self, P=8, spatiotemporal=None):
        super().__init__()
        self.stgnn = ConfiguredSTGNNBackbone(spatiotemporal)
        self.penc = PhaseEncoder()
        self.aenc = AmpEncoder(P=P)
        self.cvoca = CVOCAFusion()
        self.gf1 = nn.Linear(6, 32)
        self.gf2 = nn.Linear(32, 128)
        self.gr = nn.ReLU()

    def forward(self, x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global):
        N = x_main.size(3)
        st = self.stgnn(x_main)
        Fa = self.aenc(x_amp_map, x_amp_cell)
        Fp = self.penc(x_phase_cell)
        Fap = self.cvoca(Fa, Fp)
        g = self.gr(self.gf2(self.gr(self.gf1(x_global))))
        g = g.unsqueeze(-1).expand(-1, -1, N)
        return torch.cat([st, Fap, g], 1)
