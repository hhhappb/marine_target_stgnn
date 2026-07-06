from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffBiCAMTFE(nn.Module):
    """差分增强双向时序协同注意力 TFE。

    输入输出契约：
        输入 [B, C_in, P, N]，其中 P 是慢时间 pulse 维，N 是 range cell。
        输出 [B, C_out, ceil(P/2), N]，保持原 ST-GNN TFE 的时间压缩语义。

    雷达含义：
        相邻 pulse 差分用于突出目标造成的慢时间扰动；前后 pulse 协同注意力
        用于要求扰动同时被前向和后向上下文支持，降低单点海杂波尖峰的误导。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_diff: bool = True,
        use_coattention: bool = True,
        use_prev: bool = True,
        use_next: bool = True,
        shuffle_diff: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.use_diff = bool(use_diff)
        self.use_coattention = bool(use_coattention)
        self.use_prev = bool(use_prev)
        self.use_next = bool(use_next)
        self.shuffle_diff = bool(shuffle_diff)

        if self.num_heads <= 0:
            raise ValueError("DiffBiCAMTFE 的 num_heads 必须为正整数。")
        if self.in_channels % self.num_heads != 0:
            raise ValueError(
                f"DiffBiCAMTFE 要求 in_channels 可被 num_heads 整除，"
                f"实际 in_channels={self.in_channels}, num_heads={self.num_heads}。"
            )
        if not (self.use_prev or self.use_next):
            raise ValueError("DiffBiCAMTFE 至少需要开启 use_prev 或 use_next。")

        self.diff_proj = nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=True)
        self.diff_gate = nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=True)

        self.norm_attn = nn.GroupNorm(1, self.in_channels)
        self.q = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        self.kv_prev = nn.Conv2d(self.in_channels, self.in_channels * 2, kernel_size=1, bias=False)
        self.kv_next = nn.Conv2d(self.in_channels, self.in_channels * 2, kernel_size=1, bias=False)
        self.q_dw = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=self.in_channels,
            bias=False,
        )
        self.kv_prev_dw = nn.Conv2d(
            self.in_channels * 2,
            self.in_channels * 2,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=self.in_channels * 2,
            bias=False,
        )
        self.kv_next_dw = nn.Conv2d(
            self.in_channels * 2,
            self.in_channels * 2,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=self.in_channels * 2,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.attn_out = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.fusion_gate = nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout2d(dropout)
        self.compress = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"DiffBiCAMTFE 期望输入 [B, C, P, N]，实际为 {tuple(x.shape)}。")
        if x.size(1) != self.in_channels:
            raise ValueError(f"DiffBiCAMTFE 通道数不匹配：期望 {self.in_channels}，实际 {x.size(1)}。")
        if x.size(2) < 2:
            raise ValueError("DiffBiCAMTFE 至少需要 P>=2 才能计算慢时间前后差分。")

        base = self._diff_enhance(x) if self.use_diff else x
        if not self.use_coattention:
            return self.compress(base)

        co = self._co_attention(self.norm_attn(base))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([base, co], dim=1)))
        fused = base + gate * self.dropout(co)
        return self.compress(fused)

    def _diff_enhance(self, x: torch.Tensor) -> torch.Tensor:
        prev, nxt = self._shift_neighbors(x)
        diff = torch.cat([x - prev, nxt - x], dim=1)
        if self.shuffle_diff:
            # negative control：破坏差分与当前 pulse 的时间对应关系，保留差分幅度分布。
            diff = diff.roll(shifts=1, dims=2)
        diff_feature = torch.tanh(self.diff_proj(diff))
        gate = torch.sigmoid(self.diff_gate(torch.cat([x, diff_feature], dim=1)))
        return x + gate * diff_feature

    def _co_attention(self, x: torch.Tensor) -> torch.Tensor:
        prev, nxt = self._shift_neighbors(x)
        q = self.q_dw(self.q(x))
        kv_prev = self.kv_prev_dw(self.kv_prev(prev))
        kv_next = self.kv_next_dw(self.kv_next(nxt))
        k_prev, v_prev = kv_prev.chunk(2, dim=1)
        k_next, v_next = kv_next.chunk(2, dim=1)

        q = self._heads(q)
        k_prev = self._heads(k_prev)
        v_prev = self._heads(v_prev)
        k_next = self._heads(k_next)
        v_next = self._heads(v_next)

        q = F.normalize(q, dim=-1)
        k_prev = F.normalize(k_prev, dim=-1)
        k_next = F.normalize(k_next, dim=-1)

        outputs = []
        if self.use_prev:
            attn_prev = torch.matmul(q, k_prev.transpose(-2, -1)) * self.temperature
            prob_prev = attn_prev.softmax(dim=-1)
            outputs.append(torch.matmul(prob_prev, v_prev))
        if self.use_next:
            attn_next = torch.matmul(q, k_next.transpose(-2, -1)) * self.temperature
            prob_next = attn_next.softmax(dim=-1)
            outputs.append(torch.matmul(prob_next, v_next))
        if self.use_prev and self.use_next:
            attn_co = (prob_prev * prob_next).softmax(dim=-1)
            outputs = [torch.matmul(attn_co, v_prev), torch.matmul(attn_co, v_next)]

        out = torch.stack(outputs, dim=0).mean(dim=0)
        return self.attn_out(self._merge_heads(out, x.size(0), x.size(2), x.size(3)))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, pulses, ranges = x.shape
        head_dim = channels // self.num_heads
        x = x.permute(0, 2, 1, 3).reshape(batch * pulses, self.num_heads, head_dim, ranges)
        return x

    def _merge_heads(self, x: torch.Tensor, batch: int, pulses: int, ranges: int) -> torch.Tensor:
        head_dim = self.in_channels // self.num_heads
        x = x.reshape(batch, pulses, self.num_heads * head_dim, ranges)
        return x.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _shift_neighbors(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.cat([x[:, :, :1, :], x[:, :, :-1, :]], dim=2)
        nxt = torch.cat([x[:, :, 1:, :], x[:, :, -1:, :]], dim=2)
        return prev, nxt
