from __future__ import annotations

import torch
import torch.nn as nn


class RealImagPLRPHFeatures(nn.Module):
    """I/Q + PL/RPH 雷达先验特征通道。

    PL 表示慢时间脉冲维上的相位线性度。
    RPH 表示每个距离单元的多普勒峰值相对高度。

    输入：
        复数回波 [B, P, N]
    输出：
        浮点特征 [B, 4, P, N] = [I, Q, PL, RPH]
    """

    out_channels = 4

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = float(eps)

    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(echoes):
            raise TypeError("RealImagPLRPHFeatures 需要复数输入，形状为 [B, P, N]。")
        if echoes.dim() != 3:
            raise ValueError(f"RealImagPLRPHFeatures 期望输入 [B, P, N]，实际为 {tuple(echoes.shape)}。")
        if echoes.size(1) < 2:
            raise ValueError("PL/RPH 特征至少需要两个脉冲。")

        pl, rph = self._compute_pl_rph(echoes)
        pulses = echoes.size(1)
        pl_b = pl.unsqueeze(1).expand(-1, pulses, -1)
        rph_b = rph.unsqueeze(1).expand(-1, pulses, -1)
        return torch.stack([echoes.real, echoes.imag, pl_b, rph_b], dim=1)

    def _compute_pl_rph(self, echoes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, pulses, _ = echoes.shape
        del batch

        real_dtype = echoes.real.dtype
        device = echoes.device
        eps = self.eps

        # 沿脉冲维展开相位，避免 -pi/pi 跳变破坏线性估计。
        phase = torch.angle(echoes)
        phase_diff = phase[:, 1:, :] - phase[:, :-1, :]
        wrapped_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        phase_unwrapped = torch.cat(
            [phase[:, :1, :], phase[:, :1, :] + torch.cumsum(wrapped_diff, dim=1)],
            dim=1,
        )

        pulse_idx = torch.arange(pulses, dtype=real_dtype, device=device)
        pulse_mean = pulse_idx.mean()
        pulse_demean = pulse_idx.view(1, -1, 1) - pulse_mean
        var_pulse = (pulse_demean**2).sum(dim=1) / (pulses - 1) + eps

        phase_mean = phase_unwrapped.mean(dim=1, keepdim=True)
        phase_demean = phase_unwrapped - phase_mean
        # PL 使用相位与脉冲序号的相关系数绝对值，保留 [0, 1] 物理值域。
        cov = (phase_demean * pulse_demean).sum(dim=1) / (pulses - 1)
        var_phase = (phase_demean**2).sum(dim=1) / (pulses - 1) + eps
        pl = torch.abs(cov / (torch.sqrt(var_pulse) * torch.sqrt(var_phase) + eps))

        # RPH 衡量慢时间 Doppler 谱峰值相对平均谱幅的突出程度。
        doppler_abs = torch.abs(torch.fft.fft(echoes, dim=1))
        rph = doppler_abs.max(dim=1).values / (doppler_abs.mean(dim=1) + eps)
        return pl.to(real_dtype), rph.to(real_dtype)
