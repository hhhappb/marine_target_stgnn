import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfiguredSTGNNBackbone(nn.Module):
    """Explicit four-slot E4 main branch, preserving its input stem and pooling."""

    def __init__(self, config):
        super().__init__()
        required = {'spatial1', 'temporal1', 'spatial2', 'temporal2'}
        if set(config) != required:
            raise ValueError(f'E4 slot configuration must contain exactly {required}')
        from .registry import build_spatial_graph
        from .registry import build_temporal_module
        self.ft1 = nn.Conv2d(4, 32, (1, 3), padding=(0, 1))
        self.ft2 = nn.Conv2d(32, 64, (1, 3), padding=(0, 1))
        self.ftr = nn.ReLU()
        self.s1 = build_spatial_graph(config['spatial1'], 64, 128)
        self.t1 = build_temporal_module(config['temporal1'], 128, 256)
        self.s2 = build_spatial_graph(config['spatial2'], 256, 512)
        self.t2 = build_temporal_module(config['temporal2'], 512, 1024)
        if any(getattr(m, 'requires_raw_echoes', False) for m in (self.t1, self.t2)):
            raise ValueError('E4 feature branch cannot supply raw complex evidence')

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError('E4 main branch requires [B,4,P,N]')
        f = self.ftr(self.ft2(self.ftr(self.ft1(x))))
        f = self.t1(F.relu(self.s1(f)))
        f = self.t2(F.relu(self.s2(f)))
        return f.mean(dim=2)
