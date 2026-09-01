from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_complex_slow_time_evidence(
    raw_echoes: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """构造相邻脉冲相位增量与对数幅度差证据。"""
    echoes = raw_echoes.to(dtype=torch.complex64)
    magnitude = echoes.abs()
    previous = echoes[:, :-1, :]
    current = echoes[:, 1:, :]
    denominator = (previous.abs() * current.abs()).clamp_min(eps)
    normalized_product = current * previous.conj() / denominator

    phase_real = torch.zeros_like(magnitude)
    phase_imag = torch.zeros_like(magnitude)
    log_amplitude_difference = torch.zeros_like(magnitude)
    phase_real[:, 1:, :] = normalized_product.real
    phase_imag[:, 1:, :] = normalized_product.imag
    log_amplitude_difference[:, 1:, :] = torch.tanh(
        torch.log(current.abs().clamp_min(eps))
        - torch.log(previous.abs().clamp_min(eps))
    )
    return torch.stack(
        [phase_real, phase_imag, log_amplitude_difference], dim=1
    )


class _ZeroInitializedEvidenceTemporalProjection(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_channels, 3, 3, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=(2, 1),
            padding=(1, 0),
        )


class ComplexEvidenceResidualTFE(nn.Module):
    """将对齐到当前级时间轴的复数慢时间证据加入 TFE 预激活。"""

    requires_raw_echoes = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 1e-6,
        evidence_mode: str = "normal",
        collect_diagnostics: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.eps = float(eps)
        self.evidence_mode = str(evidence_mode).lower()
        self.collect_diagnostics = bool(collect_diagnostics)

        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("ComplexEvidenceResidualTFE 的通道数必须为正整数。")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("ComplexEvidenceResidualTFE 要求 eps 为有限正数。")
        if self.evidence_mode not in {"normal", "off", "shuffle"}:
            raise ValueError(
                "ComplexEvidenceResidualTFE evidence_mode 仅支持 normal/off/shuffle。"
            )

        # 原 TFE 参数先构造；零参数证据分支不消耗随机数，保持公共初始化序列。
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
        self.evidence_update = _ZeroInitializedEvidenceTemporalProjection(
            self.out_channels
        )
        self.evidence_output = _ZeroInitializedEvidenceTemporalProjection(
            self.out_channels
        )
        self.last_diagnostics: dict[str, float] = {}

    def forward(
        self,
        x: torch.Tensor,
        *,
        raw_echoes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(x, raw_echoes)
        update = self.update(x)
        output = self.output(x)
        if self.evidence_mode != "off":
            assert raw_echoes is not None
            with torch.autocast(device_type=x.device.type, enabled=False):
                evidence_echoes = self._intervene_on_evidence(raw_echoes)
                evidence = self._build_evidence(evidence_echoes)
                update_residual = self.evidence_update(evidence)
                output_residual = self.evidence_output(evidence)
            update = update + update_residual.to(dtype=update.dtype)
            output = output + output_residual.to(dtype=output.dtype)
            if self.collect_diagnostics:
                self._record_diagnostics(
                    update,
                    output,
                    evidence,
                    update_residual,
                    output_residual,
                )
        return torch.sigmoid(update) * torch.tanh(output)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        raw_echoes: torch.Tensor | None,
    ) -> None:
        if x.dim() != 4:
            raise ValueError(
                f"ComplexEvidenceResidualTFE 期望 [B,C,P,N]，实际为 {tuple(x.shape)}。"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                "ComplexEvidenceResidualTFE 通道数不匹配："
                f"期望 {self.in_channels}，实际 {x.size(1)}。"
            )
        if x.size(2) < 2:
            raise ValueError("ComplexEvidenceResidualTFE 至少需要 P>=2。")
        if not torch.is_floating_point(x):
            raise TypeError("ComplexEvidenceResidualTFE 需要浮点隐藏特征。")
        if self.evidence_mode == "off":
            return
        if raw_echoes is None:
            raise ValueError(
                "ComplexEvidenceResidualTFE 启用证据时必须提供 raw_echoes。"
            )
        if not torch.is_complex(raw_echoes):
            raise TypeError("ComplexEvidenceResidualTFE 的 raw_echoes 必须是复数张量。")
        expected = (x.size(0), x.size(2), x.size(3))
        if raw_echoes.dim() != 3 or tuple(raw_echoes.shape) != expected:
            raise ValueError(
                "ComplexEvidenceResidualTFE 期望 raw_echoes "
                f"shape={expected}，实际为 {tuple(raw_echoes.shape)}。"
            )

    def _intervene_on_evidence(self, raw_echoes: torch.Tensor) -> torch.Tensor:
        if self.evidence_mode != "shuffle":
            return raw_echoes
        shuffled = (
            torch.flip(raw_echoes, dims=(0,))
            if raw_echoes.size(0) > 1
            else raw_echoes
        )
        range_shift = max(1, raw_echoes.size(2) // 2)
        return torch.roll(shuffled, shifts=range_shift, dims=2)

    def _build_evidence(self, raw_echoes: torch.Tensor) -> torch.Tensor:
        return build_complex_slow_time_evidence(raw_echoes, self.eps)

    def _record_diagnostics(
        self,
        update: torch.Tensor,
        output: torch.Tensor,
        evidence: torch.Tensor,
        update_residual: torch.Tensor,
        output_residual: torch.Tensor,
    ) -> None:
        evidence_value = evidence.detach().float()
        update_value = update.detach().float()
        output_value = output.detach().float()
        update_residual_value = update_residual.detach().float()
        output_residual_value = output_residual.detach().float()
        self.last_diagnostics = {
            "complex_evidence_residual_phase_real_rms": float(
                evidence_value[:, 0].square().mean().sqrt()
            ),
            "complex_evidence_residual_phase_imag_rms": float(
                evidence_value[:, 1].square().mean().sqrt()
            ),
            "complex_evidence_residual_log_amp_diff_rms": float(
                evidence_value[:, 2].square().mean().sqrt()
            ),
            "complex_evidence_residual_update_rms": float(
                update_residual_value.square().mean().sqrt()
            ),
            "complex_evidence_residual_output_rms": float(
                output_residual_value.square().mean().sqrt()
            ),
            "complex_evidence_residual_update_ratio": float(
                update_residual_value.square().mean().sqrt()
                / (update_value.square().mean().sqrt() + self.eps)
            ),
            "complex_evidence_residual_output_ratio": float(
                output_residual_value.square().mean().sqrt()
                / (output_value.square().mean().sqrt() + self.eps)
            ),
        }

    def get_training_diagnostics(self) -> dict[str, float]:
        weight_squared = 0.0
        gradient_squared = 0.0
        gradient_present = False
        for module in (self.evidence_update, self.evidence_output):
            for parameter in module.parameters():
                weight_squared += float(parameter.detach().float().square().sum())
                if parameter.grad is not None:
                    gradient_present = True
                    gradient_squared += float(
                        parameter.grad.detach().float().square().sum()
                    )
        return {
            "complex_evidence_residual_weight_norm": weight_squared**0.5,
            "complex_evidence_residual_gradient_norm": gradient_squared**0.5,
            "complex_evidence_residual_gradient_present": float(gradient_present),
        }
