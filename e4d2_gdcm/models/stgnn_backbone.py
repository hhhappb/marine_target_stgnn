"""ST-GNN shared building blocks: GAT, STFE, TFE, STGNNBackbone.

These components are shared between ST-GNN baseline and E4-D2 models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for spatial feature extraction across range bins."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_val = 0.1
        self.alpha = 0.2

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(out_features, 1))
        self.a2 = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        B, C, N = input.size()
        x = input.permute(0, 2, 1).reshape(-1, self.in_features)
        Wh = torch.matmul(x, self.W).view(B, N, self.out_features)

        e1 = torch.matmul(Wh, self.a1).squeeze(-1)
        e2 = torch.matmul(Wh, self.a2).squeeze(-1)
        e = e1.unsqueeze(2) + e2.unsqueeze(1)
        e = self.leakyrelu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        if adj.dim() == 3:
            adj_2d = adj[0]
        else:
            adj_2d = adj
        attention = torch.where(adj_2d > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout_val, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return h_prime.permute(0, 2, 1)


class STFE(nn.Module):
    """Spatial Feature Extractor: GAT across range bins within each pulse."""

    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        self.gat = GraphAttentionLayer(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_list):
        B, N = x_list[0].size(0), x_list[0].size(2)
        P = len(x_list)
        adj = self._generate_graph(x_list)
        outputs = []
        for p in range(P):
            h = self.gat(x_list[p], adj)
            h = self.relu(h)
            outputs.append(h)
        return torch.stack(outputs, dim=2)

    @staticmethod
    def _generate_graph(x_list):
        N = x_list[0].size(2)
        adj = torch.eye(N, device=x_list[0].device)
        adj = adj + torch.diag(torch.ones(N - 1, device=x_list[0].device), 1)
        adj = adj + torch.diag(torch.ones(N - 1, device=x_list[0].device), -1)
        return adj


class TFE(nn.Module):
    """Temporal Feature Extractor: ConvGRU-like gating across pulses."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_u = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1),
                             stride=(2, 1), padding=(1, 0))
        self.W_o = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1),
                             stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        X_u = torch.sigmoid(self.W_u(x))
        X_o = torch.tanh(self.W_o(x))
        return X_u * X_o


class STGNNBackbone(nn.Module):
    """ST-GNN main branch: in_ch I/Q+features → 1024ch spatial-temporal features.

    Used by both ST-GNN baseline (in_ch=2) and E4-D2 (in_ch=4).
    """

    def __init__(self, in_ch=4):
        super().__init__()
        self.ft1 = nn.Conv2d(in_ch, 32, (1, 3), padding=(0, 1))
        self.ft2 = nn.Conv2d(32, 64, (1, 3), padding=(0, 1))
        self.ftr = nn.ReLU()
        self.s1 = STFE(64, 128)
        self.t1 = TFE(128, 256)
        self.s2 = STFE(256, 512)
        self.t2 = TFE(512, 1024)

    def forward(self, x):
        f = self.ftr(self.ft2(self.ftr(self.ft1(x))))
        xl = [f[:, :, p, :] for p in range(f.size(2))]
        s1v = self.s1(xl)
        t1v = self.t1(s1v)
        t1l = [t1v[:, :, p, :] for p in range(t1v.size(2))]
        s2v = self.s2(t1l)
        t2v = self.t2(s2v)
        # P=4 时时间维为 1；P=8/16（v2.1 PAX-L1 模块匹配 wrapper）时时间维为 P/4，
        # 取时间维均值压缩，保持输出 [B, 1024, N] 不变（P=4 时与 squeeze 等价）。
        return t2v.mean(dim=2)  # [B, 1024, N]
