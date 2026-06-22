from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RangeAttentionBase


class RadarPriorDynamicSFE(nn.Module):
    """雷达先验动态 SFE：用静态距离先验和慢时间动态先验完整替换原 SFE。

    节点是距离单元。静态图采用距离衰减和局部窗口增强；动态图由当前观测窗口内
    每个距离单元的慢时间签名相似性生成，并限制在少量非局部补边上，避免全连接过混合。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        static_gamma: float = 0.5,
        static_delta: int = 5,
        static_weight: float = 0.7,
        dynamic_topk: int = 2,
        dynamic_temperature: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 <= float(static_gamma) <= 1.0:
            raise ValueError("static_gamma 必须在 [0, 1] 内。")
        if not 0.0 <= float(static_weight) <= 1.0:
            raise ValueError("static_weight 必须在 [0, 1] 内。")
        if float(dynamic_temperature) <= 0.0:
            raise ValueError("dynamic_temperature 必须为正数。")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.static_gamma = float(static_gamma)
        self.static_delta = max(0, int(static_delta))
        self.static_weight = float(static_weight)
        self.dynamic_topk = max(0, int(dynamic_topk))
        self.dynamic_temperature = float(dynamic_temperature)
        self.linear = nn.Linear(self.in_channels, self.out_channels)
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"RadarPriorDynamicSFE 期望输入 [B, C, P, N]，实际为 {tuple(x.shape)}。")
        bsz, channels, pulses, ranges = x.shape
        if channels != self.in_channels:
            raise ValueError(f"RadarPriorDynamicSFE 通道数不匹配：期望 {self.in_channels}，实际 {channels}。")

        node_features = x.permute(0, 2, 3, 1).reshape(bsz * pulses, ranges, channels)
        support = self.linear(node_features)
        adjacency = self._hybrid_adjacency(x)
        attention = adjacency.repeat_interleave(pulses, dim=0)
        out = torch.matmul(attention, support)
        out = out.reshape(bsz, pulses, ranges, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return self.dropout(self.norm(out))

    def _hybrid_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        ranges = x.size(3)
        static_adj = self._static_adjacency(ranges, x.device, x.dtype)
        local_mask = RangeAttentionBase.local_mask(ranges, x.device, self.static_delta)
        dynamic_adj = self._dynamic_adjacency(x, local_mask)
        hybrid = self.static_weight * static_adj.unsqueeze(0) + (1.0 - self.static_weight) * dynamic_adj
        return self._symmetric_normalize(hybrid)

    def _static_adjacency(self, ranges: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        distance = RangeAttentionBase.distance_matrix(ranges, device).to(dtype=dtype)
        decay = self.static_gamma / (distance + 1.0)
        local = ((distance <= self.static_delta).to(dtype=dtype)) * (1.0 - self.static_gamma)
        adj = decay + local
        return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _dynamic_adjacency(self, x: torch.Tensor, static_mask: torch.Tensor) -> torch.Tensor:
        signature = x.permute(0, 3, 2, 1).reshape(x.size(0), x.size(3), -1)
        signature = F.normalize(signature, dim=-1)
        scores = torch.matmul(signature, signature.transpose(-1, -2)) / self.dynamic_temperature
        allowed = self._allowed_dynamic_edges(scores, static_mask)
        scores = scores.masked_fill(~allowed, -1e9)
        return torch.softmax(scores, dim=-1)

    def _allowed_dynamic_edges(self, scores: torch.Tensor, static_mask: torch.Tensor) -> torch.Tensor:
        batch, ranges, _ = scores.shape
        eye = torch.eye(ranges, device=scores.device, dtype=torch.bool)
        local = static_mask | eye
        allowed = local.unsqueeze(0).expand(batch, ranges, ranges).clone()
        if self.dynamic_topk <= 0:
            return allowed

        candidate_scores = scores.masked_fill(allowed, float("-inf"))
        k = min(self.dynamic_topk, ranges)
        indices = torch.topk(candidate_scores, k=k, dim=-1).indices
        dynamic = torch.zeros_like(allowed)
        dynamic.scatter_(-1, indices, True)
        dynamic = dynamic & torch.isfinite(candidate_scores)
        return allowed | dynamic

    @staticmethod
    def _symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt = degree.rsqrt()
        return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)
