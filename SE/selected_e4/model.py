import json
from pathlib import Path
import torch
import torch.nn as nn
from .e4_backbone import E4Backbone
from .detector import GdcmDetector

class E4Model(nn.Module):
    """Selected P8/N14 E4; five feature inputs -> [B,2,14] logits."""
    def __init__(self):
        super().__init__()
        self.config = json.loads(Path(__file__).with_name('config.json').read_text())
        self.backbone = E4Backbone(P=8, spatiotemporal=self.config['spatiotemporal'])
        self.detector = GdcmDetector(1216, variant='a0')

    def forward(self, x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global):
        batch = x_main.size(0)
        expected = ((batch,4,8,14),(batch,3,14),(batch,1,8,14),(batch,4,14),(batch,6))
        values = (x_main,x_phase_cell,x_amp_map,x_amp_cell,x_global)
        if any(tuple(value.shape) != shape for value,shape in zip(values,expected)):
            raise ValueError(f'Expected feature shapes {expected}')
        f = self.backbone(*values)
        return self.detector(f)
