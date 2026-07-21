from __future__ import annotations

import torch
import torch.nn as nn


class FixedUniformTemporalMixerTFE(nn.Module):
    """固定均匀的 pulse 聚合残差与原始 TFE 门控。

    输入 `[B,C,P,N]` 在每个距离单元内沿 pulse 维做均值混合，不使用 Q/K、
    softmax 或跨距离注意力；随后复用原 ST-GNN 的 sigmoid(update) × tanh(output)
    门控压缩。该模块用于检验 pulse 聚合本身是否足以解释 learned attention 的收益。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mixer_dim: int = 64,
        residual_scale: float = 0.1,
        use_mixer: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.mixer_dim = int(mixer_dim)
        self.residual_scale = float(residual_scale)
        self.use_mixer = bool(use_mixer)
        if self.in_channels <= 0 or self.out_channels <= 0 or self.mixer_dim <= 0:
            raise ValueError("FixedUniformTemporalMixerTFE 的通道数必须为正整数。")
        if self.residual_scale < 0:
            raise ValueError("FixedUniformTemporalMixerTFE 的 residual_scale 不能为负数。")

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
        self.value = nn.Conv2d(self.in_channels, self.mixer_dim, kernel_size=1, bias=False)
        self.mixer_out = nn.Conv2d(self.mixer_dim, self.in_channels, kernel_size=1, bias=True)
        self.last_diagnostics: dict[str, float] = {}
        self.last_mixer_shape: tuple[int, int, int, int] | None = None
        self.last_cell_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"FixedUniformTemporalMixerTFE 期望输入 [B,C,P,N]，实际为 {tuple(x.shape)}。"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"FixedUniformTemporalMixerTFE 通道数不匹配：期望 {self.in_channels}，实际 {x.size(1)}。"
            )
        if x.size(2) < 2:
            raise ValueError("FixedUniformTemporalMixerTFE 至少需要 P>=2 才能进行 pulse 聚合。")

        enhanced = x
        if self.use_mixer and self.residual_scale != 0.0:
            values = self.value(x)
            mixed = values.mean(dim=2, keepdim=True).expand(-1, -1, x.size(2), -1)
            residual = self.residual_scale * self.mixer_out(mixed)
            enhanced = x + residual
            self.last_mixer_shape = (
                int(x.size(0) * x.size(3)),
                1,
                int(x.size(2)),
                int(x.size(2)),
            )
            self._record_diagnostics(x, enhanced, residual)
        else:
            self.last_mixer_shape = None
            self._record_diagnostics(x, x, torch.zeros_like(x))

        return torch.sigmoid(self.update(enhanced)) * torch.tanh(self.output(enhanced))

    def _record_diagnostics(
        self,
        x: torch.Tensor,
        enhanced: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        input_rms = x.detach().float().square().mean().sqrt()
        residual_rms = residual.detach().float().square().mean().sqrt()
        batch, _, pulses, ranges = x.shape
        cell_input_rms = x.detach().float().square().mean(dim=(1, 2)).sqrt()
        cell_residual_rms = residual.detach().float().square().mean(dim=(1, 2)).sqrt()
        cell_enhanced_rms = enhanced.detach().float().square().mean(dim=(1, 2)).sqrt()
        self.last_cell_diagnostics = {
            "attention_entropy": torch.full(
                (batch, ranges), 1.0 if self.use_mixer and self.residual_scale != 0.0 else 0.0,
                device=x.device,
                dtype=torch.float32,
            ),
            "attention_max_weight": torch.full(
                (batch, ranges),
                1.0 / float(pulses) if self.use_mixer and self.residual_scale != 0.0 else 0.0,
                device=x.device,
                dtype=torch.float32,
            ),
            "attention_diagonal_fraction": torch.full(
                (batch, ranges),
                1.0 / float(pulses) if self.use_mixer and self.residual_scale != 0.0 else 0.0,
                device=x.device,
                dtype=torch.float32,
            ),
            "attention_residual_ratio": cell_residual_rms / (cell_input_rms + 1e-8),
            "tfe_input_rms": cell_input_rms,
            "tfe_enhanced_rms": cell_enhanced_rms,
        }
        self.last_diagnostics = {
            "attention_entropy": float(self.last_cell_diagnostics["attention_entropy"].mean().item()),
            "attention_max_weight": float(self.last_cell_diagnostics["attention_max_weight"].mean().item()),
            "attention_diagonal_fraction": float(
                self.last_cell_diagnostics["attention_diagonal_fraction"].mean().item()
            ),
            "attention_residual_ratio": float((residual_rms / (input_rms + 1e-8)).item()),
            "tfe_input_rms": float(input_rms.item()),
            "tfe_enhanced_rms": float(enhanced.detach().float().square().mean().sqrt().item()),
        }
