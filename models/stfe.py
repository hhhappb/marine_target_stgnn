import torch
import torch.nn as nn
from .layers.gat_conv import GATConv
from .layers.tfe import TFELayer


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpatialFeatureExtractor, self).__init__()
        self.gat = GATConv(in_features, out_features)

    def forward(self, x, adj):
        return self.gat(x, adj)


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalFeatureExtractor, self).__init__()
        self.tfe = TFELayer(in_channels, out_channels)

    def forward(self, x):
        return self.tfe(x)


class STFE(nn.Module):
    def __init__(self, spatial_in, spatial_out, temporal_in, temporal_out):
        super(STFE, self).__init__()
        self.sfe = SpatialFeatureExtractor(spatial_in, spatial_out)
        self.tfe = TemporalFeatureExtractor(temporal_in, temporal_out)

    def forward(self, x_list, adj_list):
        spatial_out = []
        for x, adj in zip(x_list, adj_list):
            h = self.sfe(x, adj)
            spatial_out.append(h)

        spatial_stack = torch.stack(spatial_out, dim=1)
        temporal_out = self.tfe(spatial_stack)
        return temporal_out
