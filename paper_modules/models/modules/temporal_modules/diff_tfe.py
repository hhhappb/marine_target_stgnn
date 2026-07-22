from __future__ import annotations

import torch
import torch.nn as nn


class DiffTFE(nn.Module):
    """只加入归一化慢时间差分残差的 TFE。

    输入输出契约：输入 `[B,C,P,N]`，输出 `[B,C_out,ceil(P/2),N]`。
    增强关闭时，前向路径严格等价于原 ST-GNN 的 sigmoid(update) × tanh(output)
    门控压缩；差分分支只在当前模块的 enhanced_x 中增加局部慢时间残差。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diff_scale: float = 0.1,
        use_diff: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.diff_scale = float(diff_scale)
        self.use_diff = bool(use_diff)
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("DiffTFE 的通道数必须为正整数。")
        if self.diff_scale < 0:
            raise ValueError("DiffTFE 的 diff_scale 不能为负数。")

        # 这两个卷积与原 ST-GNN TFE 保持同样的参数形状和门控语义。
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

        self.diff_norm = nn.GroupNorm(1, self.in_channels * 2)
        self.diff_proj = nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=True)
        self.diff_gate = nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=True)
        self.last_diagnostics: dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"DiffTFE 期望输入 [B,C,P,N]，实际为 {tuple(x.shape)}。")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"DiffTFE 通道数不匹配：期望 {self.in_channels}，实际 {x.size(1)}。"
            )
        if x.size(2) < 2:
            raise ValueError("DiffTFE 至少需要 P>=2 才能计算慢时间差分。")

        enhanced = x
        if self.use_diff and self.diff_scale != 0.0:
            prev, nxt = self._shift_neighbors(x)
            diff = torch.cat([x - prev, nxt - x], dim=1)
            diff_feature = torch.tanh(self.diff_proj(self.diff_norm(diff)))
            diff_gate = torch.sigmoid(self.diff_gate(torch.cat([x, diff_feature], dim=1)))
            residual = self.diff_scale * diff_gate * diff_feature
            enhanced = x + residual
            self._record_diagnostics(x, enhanced, residual, diff_gate)
        else:
            self._record_diagnostics(x, x, torch.zeros_like(x), torch.zeros_like(x))

        return torch.sigmoid(self.update(enhanced)) * torch.tanh(self.output(enhanced))

    def _record_diagnostics(
        self,
        x: torch.Tensor,
        enhanced: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        gate_values = gate.detach().float().reshape(-1)
        quantiles = torch.quantile(gate_values, torch.tensor([0.05, 0.95], device=gate.device))
        input_rms = x.detach().float().square().mean().sqrt()
        residual_rms = residual.detach().float().square().mean().sqrt()
        self.last_diagnostics = {
            "diff_gate_mean": float(gate_values.mean().item()),
            "diff_gate_std": float(gate_values.std(unbiased=False).item()),
            "diff_gate_p05": float(quantiles[0].item()),
            "diff_gate_p95": float(quantiles[1].item()),
            "diff_residual_ratio": float((residual_rms / (input_rms + 1e-8)).item()),
            "tfe_input_rms": float(input_rms.item()),
            "tfe_enhanced_rms": float(enhanced.detach().float().square().mean().sqrt().item()),
        }

    @staticmethod
    def _shift_neighbors(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.cat([x[:, :, :1, :], x[:, :, :-1, :]], dim=2)
        nxt = torch.cat([x[:, :, 1:, :], x[:, :, -1:, :]], dim=2)
        return prev, nxt
