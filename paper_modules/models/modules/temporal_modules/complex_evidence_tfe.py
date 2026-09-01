from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ZeroInitializedEvidenceProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias)


class ComplexEvidenceTFE(nn.Module):
    """用原始复数慢时间证据有界调制原 ST-GNN TFE1。"""

    requires_raw_echoes = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta_max: float = 0.1,
        eps: float = 1e-6,
        use_modulation: bool = True,
        evidence_mode: str = "normal",
        collect_diagnostics: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.use_modulation = bool(use_modulation)
        self.evidence_mode = str(evidence_mode).lower()
        self.collect_diagnostics = bool(collect_diagnostics)

        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("ComplexEvidenceTFE 的通道数必须为正整数。")
        if not math.isfinite(self.beta_max) or not 0.0 < self.beta_max <= 1.0:
            raise ValueError("ComplexEvidenceTFE 要求 0 < beta_max <= 1。")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("ComplexEvidenceTFE 要求 eps 为有限正数。")
        if self.evidence_mode not in {"normal", "off", "shuffle"}:
            raise ValueError(
                "ComplexEvidenceTFE evidence_mode 仅支持 normal/off/shuffle。"
            )

        # 先构造原 TFE 参数；零参数投影不消耗随机数，保持公共初始化序列。
        self.update = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=(3, 1), stride=(2, 1), padding=(1, 0),
        )
        self.output = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=(3, 1), stride=(2, 1), padding=(1, 0),
        )
        self.evidence_proj = _ZeroInitializedEvidenceProjection(3, self.in_channels)
        self.last_diagnostics: dict[str, float] = {}
        self.last_cell_diagnostics: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        *,
        raw_echoes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(x, raw_echoes)
        enhanced = x
        if self.use_modulation and self.evidence_mode != "off":
            assert raw_echoes is not None
            with torch.autocast(device_type=x.device.type, enabled=False):
                evidence_echoes = self._intervene_on_evidence(raw_echoes)
                evidence = self._build_evidence(evidence_echoes)
                modulation = self.beta_max * torch.tanh(self.evidence_proj(evidence))
            enhanced = x * (1.0 + modulation.to(dtype=x.dtype))
            if self.collect_diagnostics:
                self._record_diagnostics(x, enhanced, evidence, modulation)
        return torch.sigmoid(self.update(enhanced)) * torch.tanh(self.output(enhanced))

    def _validate_inputs(
        self,
        x: torch.Tensor,
        raw_echoes: torch.Tensor | None,
    ) -> None:
        if x.dim() != 4:
            raise ValueError(f"ComplexEvidenceTFE 期望 [B,C,P,N]，实际为 {tuple(x.shape)}。")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"ComplexEvidenceTFE 通道数不匹配：期望 {self.in_channels}，实际 {x.size(1)}。"
            )
        if x.size(2) < 2:
            raise ValueError("ComplexEvidenceTFE 至少需要 P>=2。")
        if not torch.is_floating_point(x):
            raise TypeError("ComplexEvidenceTFE 需要浮点隐藏特征。")
        if not self.use_modulation:
            return
        if raw_echoes is None:
            raise ValueError("ComplexEvidenceTFE 启用调制时必须提供 raw_echoes。")
        if not torch.is_complex(raw_echoes):
            raise TypeError("ComplexEvidenceTFE 的 raw_echoes 必须是复数张量。")
        expected = (x.size(0), x.size(2), x.size(3))
        if raw_echoes.dim() != 3 or tuple(raw_echoes.shape) != expected:
            raise ValueError(
                f"ComplexEvidenceTFE 期望 raw_echoes shape={expected}，实际为 {tuple(raw_echoes.shape)}。"
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
        echoes = raw_echoes.to(dtype=torch.complex64)
        magnitude = echoes.abs()
        previous = echoes[:, :-1, :]
        current = echoes[:, 1:, :]
        denominator = (previous.abs() * current.abs()).clamp_min(self.eps)
        normalized_product = current * previous.conj() / denominator

        phase_real = torch.zeros_like(magnitude)
        phase_imag = torch.zeros_like(magnitude)
        log_amplitude_difference = torch.zeros_like(magnitude)
        phase_real[:, 1:, :] = normalized_product.real
        phase_imag[:, 1:, :] = normalized_product.imag
        log_amplitude_difference[:, 1:, :] = torch.tanh(
            torch.log(current.abs().clamp_min(self.eps))
            - torch.log(previous.abs().clamp_min(self.eps))
        )
        return torch.stack(
            [phase_real, phase_imag, log_amplitude_difference], dim=1
        )

    def _record_diagnostics(
        self,
        x: torch.Tensor,
        enhanced: torch.Tensor,
        evidence: torch.Tensor,
        modulation: torch.Tensor,
    ) -> None:
        evidence_value = evidence.detach().float()
        modulation_value = modulation.detach().float()
        input_rms = x.detach().float().square().mean().sqrt()
        enhanced_rms = enhanced.detach().float().square().mean().sqrt()
        self.last_diagnostics = {
            "complex_evidence_phase_real_rms": float(evidence_value[:, 0].square().mean().sqrt()),
            "complex_evidence_phase_imag_rms": float(evidence_value[:, 1].square().mean().sqrt()),
            "complex_evidence_log_amp_diff_rms": float(evidence_value[:, 2].square().mean().sqrt()),
            "complex_evidence_modulation_rms": float(modulation_value.square().mean().sqrt()),
            "complex_evidence_modulation_abs_max": float(modulation_value.abs().max()),
            "complex_evidence_modulation_saturation_ratio": float(
                (modulation_value.abs() >= 0.9 * self.beta_max).float().mean()
            ),
            "complex_evidence_enhanced_input_rms_ratio": float(
                enhanced_rms / (input_rms + self.eps)
            ),
        }
        self.last_cell_diagnostics = {
            "phase_real_rms": evidence_value[:, 0].square().mean(dim=1).sqrt(),
            "phase_imag_rms": evidence_value[:, 1].square().mean(dim=1).sqrt(),
            "log_amp_diff_rms": evidence_value[:, 2].square().mean(dim=1).sqrt(),
            "modulation_rms": modulation_value.square().mean(dim=(1, 2)).sqrt(),
        }

    def get_training_diagnostics(self) -> dict[str, float]:
        if not self.use_modulation:
            return {}
        weight_squared = 0.0
        gradient_squared = 0.0
        gradient_present = False
        for parameter in self.evidence_proj.parameters():
            weight_squared += float(parameter.detach().float().square().sum())
            if parameter.grad is not None:
                gradient_present = True
                gradient_squared += float(parameter.grad.detach().float().square().sum())
        return {
            "complex_evidence_projection_weight_norm": weight_squared**0.5,
            "complex_evidence_projection_gradient_norm": gradient_squared**0.5,
            "complex_evidence_projection_gradient_present": float(gradient_present),
        }
