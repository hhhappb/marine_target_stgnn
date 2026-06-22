from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OriginalSTFEGraph(nn.Module):
    """原 ST-GNN 的 SFE：固定相邻距离图 + additive GAT。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dropout = float(dropout)
        self.weight = nn.Parameter(torch.empty(self.in_channels, self.out_channels))
        self.attn_left = nn.Parameter(torch.empty(self.out_channels, 1))
        self.attn_right = nn.Parameter(torch.empty(self.out_channels, 1))
        self.leaky_relu = nn.LeakyReLU(float(alpha))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_left, gain=1.414)
        nn.init.xavier_uniform_(self.attn_right, gain=1.414)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"OriginalSTFEGraph 期望输入 [B, C, P, N]，实际为 {tuple(x.shape)}。")
        bsz, channels, pulses, ranges = x.shape
        if channels != self.in_channels:
            raise ValueError(f"OriginalSTFEGraph 通道数不匹配：期望 {self.in_channels}，实际 {channels}。")

        h = x.permute(0, 2, 1, 3).reshape(bsz * pulses, channels, ranges)
        out = F.relu(self._gat(h, ranges))
        return out.reshape(bsz, pulses, self.out_channels, ranges).permute(0, 2, 1, 3).contiguous()

    def _gat(self, h: torch.Tensor, ranges: int) -> torch.Tensor:
        batch = h.size(0)
        nodes = h.permute(0, 2, 1).reshape(-1, self.in_channels)
        wh = torch.matmul(nodes, self.weight).view(batch, ranges, self.out_channels)

        e_left = torch.matmul(wh, self.attn_left).squeeze(-1)
        e_right = torch.matmul(wh, self.attn_right).squeeze(-1)
        scores = self.leaky_relu(e_left.unsqueeze(2) + e_right.unsqueeze(1))

        adj = self._local_adj(ranges, h.device)
        scores = scores.masked_fill(~adj.unsqueeze(0), -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(attention, wh).permute(0, 2, 1)

    @staticmethod
    def _local_adj(ranges: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(ranges, device=device)
        return (idx[:, None] - idx[None, :]).abs() <= 1
