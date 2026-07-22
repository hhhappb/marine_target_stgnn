from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ZeroInitializedPointwiseProjection(nn.Module):
    """不消耗随机数的零初始化 1×1 投影，用于保持公共参数初始化序列。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias)


class ScaleNormalizedDifferenceDecompositionTFE(nn.Module):
    """用尺度归一化趋势/曲率证据有界调制原 ST-GNN TFE。

    输入为 `[B,C,P,N]`。时间证据只沿有序 pulse 维计算，主路输入不做
    归一化；输出保持 `[B,C_out,ceil(P/2),N]`。P=2 时趋势使用单边差分，
    曲率退化为零，以兼容外部 P=4 时的第二级 TFE。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta_max: float = 0.1,
        eps: float = 1e-6,
        use_modulation: bool = True,
        collect_diagnostics: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.use_modulation = bool(use_modulation)
        self.collect_diagnostics = bool(collect_diagnostics)

        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("ScaleNormalizedDifferenceDecompositionTFE 的通道数必须为正整数。")
        if not math.isfinite(self.beta_max) or not 0.0 < self.beta_max <= 1.0:
            raise ValueError(
                "ScaleNormalizedDifferenceDecompositionTFE 要求 0 < beta_max <= 1，"
                f"实际为 {self.beta_max}。"
            )
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(
                "ScaleNormalizedDifferenceDecompositionTFE 要求 eps 为有限正数，"
                f"实际为 {self.eps}。"
            )

        # 先构造原 TFE 参数；零参数证据投影不消耗随机数，保证公共参数初始化序列不变。
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
        self.evidence_proj = _ZeroInitializedPointwiseProjection(
            2 * self.in_channels,
            self.in_channels,
        )

        self.last_diagnostics: dict[str, float] = {}
        self.last_cell_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        if self.use_modulation:
            with torch.autocast(device_type=x.device.type, enabled=False):
                trend, curvature, normalized_curvature = self._build_evidence(x)
                evidence = torch.cat([trend, normalized_curvature], dim=1)
                modulation = self.beta_max * torch.tanh(self.evidence_proj(evidence))
            enhanced = x * (1.0 + modulation.to(dtype=x.dtype))
            if self.collect_diagnostics:
                self._record_diagnostics(x, enhanced, trend, curvature, normalized_curvature, modulation)
        else:
            enhanced = x
            if self.collect_diagnostics:
                with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                    trend, curvature, normalized_curvature = self._build_evidence(x)
                    modulation = torch.zeros_like(x, dtype=torch.float32)
                    self._record_diagnostics(
                        x,
                        enhanced,
                        trend,
                        curvature,
                        normalized_curvature,
                        modulation,
                    )

        return torch.sigmoid(self.update(enhanced)) * torch.tanh(self.output(enhanced))

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(
                "ScaleNormalizedDifferenceDecompositionTFE 期望输入 [B,C,P,N]，"
                f"实际为 {tuple(x.shape)}。"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                "ScaleNormalizedDifferenceDecompositionTFE 通道数不匹配："
                f"期望 {self.in_channels}，实际 {x.size(1)}。"
            )
        if x.size(2) < 2:
            raise ValueError("ScaleNormalizedDifferenceDecompositionTFE 至少需要 P>=2。")
        if not torch.is_floating_point(x):
            raise TypeError("ScaleNormalizedDifferenceDecompositionTFE 需要浮点特征输入。")

    def _build_evidence(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats_x = x.float()
        scale = torch.sqrt(stats_x.square().mean(dim=2, keepdim=True) + self.eps)
        z = stats_x / scale

        left_trend = z[:, :, 1:2, :] - z[:, :, :1, :]
        right_trend = z[:, :, -1:, :] - z[:, :, -2:-1, :]
        if z.size(2) == 2:
            trend = torch.cat([left_trend, right_trend], dim=2)
            curvature = torch.zeros_like(z)
        else:
            center_trend = 0.5 * (z[:, :, 2:, :] - z[:, :, :-2, :])
            trend = torch.cat([left_trend, center_trend, right_trend], dim=2)
            center_curvature = z[:, :, 2:, :] - 2.0 * z[:, :, 1:-1, :] + z[:, :, :-2, :]
            boundary = torch.zeros_like(z[:, :, :1, :])
            curvature = torch.cat([boundary, center_curvature, boundary], dim=2)

        normalized_curvature = curvature / math.sqrt(12.0)
        return trend, curvature, normalized_curvature

    def _record_diagnostics(
        self,
        x: torch.Tensor,
        enhanced: torch.Tensor,
        trend: torch.Tensor,
        curvature: torch.Tensor,
        normalized_curvature: torch.Tensor,
        modulation: torch.Tensor,
    ) -> None:
        input_value = x.detach().float()
        enhanced_value = enhanced.detach().float()
        trend_value = trend.detach().float()
        curvature_value = curvature.detach().float()
        normalized_curvature_value = normalized_curvature.detach().float()
        modulation_value = modulation.detach().float()

        input_rms = input_value.square().mean().sqrt()
        enhanced_rms = enhanced_value.square().mean().sqrt()
        trend_rms = trend_value.square().mean().sqrt()
        curvature_rms = curvature_value.square().mean().sqrt()
        normalized_curvature_rms = normalized_curvature_value.square().mean().sqrt()

        self.last_diagnostics = {
            "difference_beta_max": self.beta_max,
            "difference_modulation_enabled": float(self.use_modulation),
            "difference_trend_rms": float(trend_rms.item()),
            "difference_curvature_rms_before_scaling": float(curvature_rms.item()),
            "difference_curvature_rms_after_scaling": float(normalized_curvature_rms.item()),
            "difference_trend_curvature_rms_ratio": float(
                (normalized_curvature_rms / (trend_rms + self.eps)).item()
            ),
            "difference_modulation_rms": float(modulation_value.square().mean().sqrt().item()),
            "difference_modulation_abs_max": float(modulation_value.abs().max().item()),
            "difference_enhanced_input_rms_ratio": float(
                (enhanced_rms / (input_rms + self.eps)).item()
            ),
        }

        reduce_dims = (1, 2)
        input_cell_rms = input_value.square().mean(dim=reduce_dims).sqrt()
        enhanced_cell_rms = enhanced_value.square().mean(dim=reduce_dims).sqrt()
        self.last_cell_diagnostics = {
            "trend_rms": trend_value.square().mean(dim=reduce_dims).sqrt(),
            "curvature_rms_before_scaling": curvature_value.square().mean(dim=reduce_dims).sqrt(),
            "curvature_rms_after_scaling": normalized_curvature_value.square().mean(dim=reduce_dims).sqrt(),
            "modulation_rms": modulation_value.square().mean(dim=reduce_dims).sqrt(),
            "enhanced_input_rms_ratio": enhanced_cell_rms / (input_cell_rms + self.eps),
        }
