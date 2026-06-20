from __future__ import annotations

import math

import torch
import torch.nn as nn


class RangeAttentionBase(nn.Module):
    """空间图公共基类：提供距离单元注意力投影和常用距离/邻域工具。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        bsz, channels, pulses, ranges = x.shape
        h = x.permute(0, 2, 3, 1).reshape(bsz * pulses, ranges, channels)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        return q, k, v, (bsz, pulses, ranges, q.size(-1))

    def finish(self, attention: torch.Tensor, v: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
        bsz, pulses, ranges, _ = shape
        out = torch.matmul(attention, v)
        out = self.out_proj(out)
        return out.reshape(bsz, pulses, ranges, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))

    @staticmethod
    def distance_matrix(size: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(size, device=device)
        return (idx[:, None] - idx[None, :]).abs().float()

    @staticmethod
    def local_mask(size: int, device: torch.device, radius: int) -> torch.Tensor:
        return RangeAttentionBase.distance_matrix(size, device) <= radius
