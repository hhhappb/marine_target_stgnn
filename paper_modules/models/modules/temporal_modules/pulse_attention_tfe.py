from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseAttentionOnlyTFE(nn.Module):
    """只加入按距离单元独立计算的 P×P 慢时间注意力残差。

    输入 `[B,C,P,N]` 会重排为 `[B*N,H,P,D]`，因此注意力只在同一 range
    cell 的 pulse 维上计算，不在 N 个距离单元之间混合；最后仍使用原
    ST-GNN 的 sigmoid(update) × tanh(output) 门控压缩。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_dim: int = 64,
        num_heads: int = 4,
        residual_scale: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.residual_scale = float(residual_scale)
        self.use_attention = bool(use_attention)
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("PulseAttentionOnlyTFE 的通道数必须为正整数。")
        if self.attention_dim <= 0 or self.attention_dim % self.num_heads != 0:
            raise ValueError("PulseAttentionOnlyTFE 要求 attention_dim 能被 num_heads 整除。")
        if self.residual_scale < 0:
            raise ValueError("PulseAttentionOnlyTFE 的 residual_scale 不能为负数。")

        self.update = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
        )
        self.output = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
        )
        self.q = nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1, bias=False)
        self.k = nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1, bias=False)
        self.attention_out = nn.Conv2d(self.attention_dim, self.in_channels, kernel_size=1, bias=True)
        self.counterfactual_mode = "learned"
        self.last_attention_shape: tuple[int, int, int, int] | None = None
        self.last_diagnostics: dict[str, float] = {}
        self.last_attention_weights: torch.Tensor | None = None
        self.last_attention_logits: torch.Tensor | None = None
        self.last_cell_diagnostics: dict[str, torch.Tensor] = {}
        self.diagnostic_logit_multiplier = 1.0

    def set_counterfactual_mode(self, mode: str) -> None:
        """设置冻结 checkpoint 的注意力反事实模式。"""
        allowed = {"learned", "uniform", "identity", "residual_off"}
        if mode not in allowed:
            raise ValueError(f"未知 pulse attention 反事实模式: {mode!r}，允许值为 {sorted(allowed)}。")
        self.counterfactual_mode = mode

    def set_diagnostic_logit_multiplier(self, value: float) -> None:
        """设置仅用于冻结 checkpoint 诊断的预 softmax logits 倍数。"""
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"diagnostic_logit_multiplier 必须为有限正数，实际为 {value}。")
        self.diagnostic_logit_multiplier = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"PulseAttentionOnlyTFE 期望输入 [B,C,P,N]，实际为 {tuple(x.shape)}。")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"PulseAttentionOnlyTFE 通道数不匹配：期望 {self.in_channels}，实际 {x.size(1)}。"
            )
        if x.size(2) < 2:
            raise ValueError("PulseAttentionOnlyTFE 至少需要 P>=2 才能计算 pulse attention。")

        enhanced = x
        mode = self.counterfactual_mode
        if self.use_attention and self.residual_scale != 0.0 and mode != "residual_off":
            attention, weights = self._pulse_attention(x, mode)
            residual = self.residual_scale * self.attention_out(attention)
            enhanced = x + residual
            self._record_diagnostics(x, enhanced, residual, weights)
        else:
            self.last_attention_shape = None
            self.last_attention_weights = None
            self.last_attention_logits = None
            self._record_diagnostics(x, x, torch.zeros_like(x), None)

        return torch.sigmoid(self.update(enhanced)) * torch.tanh(self.output(enhanced))

    def _pulse_attention(self, x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, pulses, ranges = x.shape
        head_dim = self.attention_dim // self.num_heads
        self.last_attention_logits = None

        def reshape_heads(values: torch.Tensor) -> torch.Tensor:
            values = values.permute(0, 3, 2, 1).reshape(batch * ranges, pulses, self.attention_dim)
            values = values.view(batch * ranges, pulses, self.num_heads, head_dim)
            return values.transpose(1, 2).contiguous()

        v = reshape_heads(self.v(x))
        if mode == "learned":
            q = F.normalize(reshape_heads(self.q(x)), dim=-1)
            k = F.normalize(reshape_heads(self.k(x)), dim=-1)
            weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(head_dim))
            weights = weights * self.diagnostic_logit_multiplier
            self.last_attention_logits = weights.detach()
            weights = weights.softmax(dim=-1)
        elif mode == "uniform":
            weights = torch.full(
                (batch * ranges, self.num_heads, pulses, pulses),
                1.0 / float(pulses),
                device=x.device,
                dtype=v.dtype,
            )
        elif mode == "identity":
            weights = torch.eye(pulses, device=x.device, dtype=v.dtype).view(1, 1, pulses, pulses)
            weights = weights.expand(batch * ranges, self.num_heads, pulses, pulses)
        else:
            raise ValueError(f"_pulse_attention 不支持模式: {mode!r}。")
        self.last_attention_shape = tuple(int(value) for value in weights.shape)
        self.last_attention_weights = weights.detach()

        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(batch * ranges, pulses, self.attention_dim)
        attended = attended.reshape(batch, ranges, pulses, self.attention_dim)
        attended = attended.permute(0, 3, 2, 1).contiguous()
        return attended, weights

    def _record_diagnostics(
        self,
        x: torch.Tensor,
        enhanced: torch.Tensor,
        residual: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> None:
        input_rms = x.detach().float().square().mean().sqrt()
        residual_rms = residual.detach().float().square().mean().sqrt()
        diagnostics = {
            "attention_residual_ratio": float((residual_rms / (input_rms + 1e-8)).item()),
            "tfe_input_rms": float(input_rms.item()),
            "tfe_enhanced_rms": float(enhanced.detach().float().square().mean().sqrt().item()),
        }
        batch, _, pulses, ranges = x.shape
        cell_input_rms = x.detach().float().square().mean(dim=(1, 2)).sqrt()
        cell_residual_rms = residual.detach().float().square().mean(dim=(1, 2)).sqrt()
        cell_enhanced_rms = enhanced.detach().float().square().mean(dim=(1, 2)).sqrt()
        self.last_cell_diagnostics = {
            "attention_residual_ratio": cell_residual_rms / (cell_input_rms + 1e-8),
            "tfe_input_rms": cell_input_rms,
            "tfe_enhanced_rms": cell_enhanced_rms,
        }
        if weights is not None:
            entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=-1)
            entropy = entropy / math.log(float(pulses))
            cell_entropy = entropy.mean(dim=(1, 2)).reshape(batch, ranges)
            cell_max_weight = weights.max(dim=-1).values.mean(dim=(1, 2)).reshape(batch, ranges)
            cell_diagonal = weights.diagonal(dim1=-2, dim2=-1).mean(dim=(1, 2)).reshape(batch, ranges)
            diagnostics.update(
                {
                    "attention_entropy": float(cell_entropy.detach().mean().item()),
                    "attention_max_weight": float(cell_max_weight.detach().mean().item()),
                    "attention_diagonal_fraction": float(cell_diagonal.detach().mean().item()),
                }
            )
            self.last_cell_diagnostics.update(
                {
                    "attention_entropy": cell_entropy.detach(),
                    "attention_max_weight": cell_max_weight.detach(),
                    "attention_diagonal_fraction": cell_diagonal.detach(),
                }
            )
        else:
            diagnostics.update(
                {
                    "attention_entropy": 0.0,
                    "attention_max_weight": 0.0,
                    "attention_diagonal_fraction": 0.0,
                }
            )
            zero = torch.zeros((batch, ranges), device=x.device, dtype=torch.float32)
            self.last_cell_diagnostics.update(
                {
                    "attention_entropy": zero,
                    "attention_max_weight": zero,
                    "attention_diagonal_fraction": zero,
                }
            )
        self.last_diagnostics = diagnostics
