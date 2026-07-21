from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagAwareAntiAliasTFE(nn.Module):
    """可学习局部 lag 滤波与原 ST-GNN TFE 门控的组合。

    输入 `[B,C,P,N]`，只沿有序 pulse 维做局部混合，不在距离单元之间混合；
    输出保持原 ST-GNN TFE 的 `[B,C_out,ceil(P/2),N]` 语义。

    滤波核使用 softmax 归一化，并以 `[1,2,1]` 或 `[1,4,6,4,1]`
    初始化。`gamma` 通过 sigmoid 限制在 `[0, gamma_max]`，控制原始特征与
    局部慢时间滤波结果之间的凸组合。默认 replicate padding 避免短 pulse
    窗口在边界处引入零填充衰减。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        filter_groups: int = 1,
        gamma_init: float = 0.05,
        gamma_max: float = 1.0,
        use_filter: bool = True,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.filter_groups = int(filter_groups)
        self.gamma_max = float(gamma_max)
        self.use_filter = bool(use_filter)
        self.padding_mode = str(padding_mode)

        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("LagAwareAntiAliasTFE 的通道数必须为正整数。")
        if self.kernel_size not in {3, 5}:
            raise ValueError("LagAwareAntiAliasTFE 当前只支持 kernel_size=3 或 5。")
        if self.filter_groups <= 0 or self.in_channels % self.filter_groups != 0:
            raise ValueError(
                "LagAwareAntiAliasTFE 要求 filter_groups 为正整数且能整除 in_channels，"
                f"实际为 filter_groups={self.filter_groups}, in_channels={self.in_channels}。"
            )
        if not 0.0 < self.gamma_max <= 1.0:
            raise ValueError("LagAwareAntiAliasTFE 要求 0 < gamma_max <= 1。")
        if not 0.0 < gamma_init < self.gamma_max:
            raise ValueError(
                "LagAwareAntiAliasTFE 要求 0 < gamma_init < gamma_max，"
                f"实际为 gamma_init={gamma_init}, gamma_max={gamma_max}。"
            )
        if self.padding_mode not in {"replicate", "reflect"}:
            raise ValueError("LagAwareAntiAliasTFE 的 padding_mode 只支持 replicate 或 reflect。")

        initial_kernel = self._initial_kernel(self.kernel_size)
        initial_logits = initial_kernel.log().view(1, 1, self.kernel_size, 1)
        self.filter_logits = nn.Parameter(initial_logits.repeat(self.filter_groups, 1, 1, 1))

        gamma_ratio = gamma_init / self.gamma_max
        gamma_logit = math.log(gamma_ratio / (1.0 - gamma_ratio))
        self.gamma_logit = nn.Parameter(torch.tensor([gamma_logit], dtype=torch.float32))

        # 这两个卷积保持原 ST-GNN TFE 的参数形状、stride 和门控语义。
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

        self.last_diagnostics: dict[str, float] = {}
        self.last_cell_diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _initial_kernel(kernel_size: int) -> torch.Tensor:
        if kernel_size == 3:
            values = [1.0, 2.0, 1.0]
        elif kernel_size == 5:
            values = [1.0, 4.0, 6.0, 4.0, 1.0]
        else:
            raise ValueError(f"不支持的初始 lag kernel_size: {kernel_size}。")
        kernel = torch.tensor(values, dtype=torch.float32)
        return kernel / kernel.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"LagAwareAntiAliasTFE 期望输入 [B,C,P,N]，实际为 {tuple(x.shape)}。"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"LagAwareAntiAliasTFE 通道数不匹配：期望 {self.in_channels}，实际 {x.size(1)}。"
            )
        if x.size(2) < 2:
            raise ValueError("LagAwareAntiAliasTFE 至少需要 P>=2。")

        if self.use_filter:
            mixed, weights = self._temporal_filter(x)
            gamma = self.gamma_max * torch.sigmoid(self.gamma_logit)
            enhanced = (1.0 - gamma) * x + gamma * mixed
            self._record_diagnostics(x, mixed, enhanced, weights, gamma)
        else:
            mixed = x
            weights = self._filter_weights()
            gamma = x.new_zeros(1)
            enhanced = x
            self._record_diagnostics(x, mixed, enhanced, weights, gamma)

        return torch.sigmoid(self.update(enhanced)) * torch.tanh(self.output(enhanced))

    def _temporal_filter(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = self.kernel_size // 2
        padded = F.pad(x, (0, 0, padding, padding), mode=self.padding_mode)
        weights = self._filter_weights().to(dtype=x.dtype, device=x.device)
        weights = weights.repeat_interleave(self.in_channels // self.filter_groups, dim=0)
        mixed = F.conv2d(padded, weights, stride=1, padding=0, groups=self.in_channels)
        return mixed, weights

    def _filter_weights(self) -> torch.Tensor:
        return F.softmax(self.filter_logits, dim=2)

    @staticmethod
    def _high_frequency_ratio(x: torch.Tensor) -> float:
        if x.size(2) < 4:
            return 0.0
        spectrum = torch.fft.rfft(x.detach().float(), dim=2)
        energy = spectrum.abs().square()
        frequencies = torch.fft.rfftfreq(x.size(2), device=x.device)
        high_frequency = frequencies >= 0.25
        total = energy.sum(dim=2).mean()
        high = energy[:, :, high_frequency, :].sum(dim=2).mean()
        return float((high / (total + 1e-8)).item())

    def _record_diagnostics(
        self,
        x: torch.Tensor,
        mixed: torch.Tensor,
        enhanced: torch.Tensor,
        weights: torch.Tensor,
        gamma: torch.Tensor,
    ) -> None:
        input_rms = x.detach().float().square().mean().sqrt()
        mixed_rms = mixed.detach().float().square().mean().sqrt()
        enhanced_rms = enhanced.detach().float().square().mean().sqrt()
        delta_rms = (mixed.detach().float() - x.detach().float()).square().mean().sqrt()
        input_cell_rms = x.detach().float().square().mean(dim=(1, 2)).sqrt()
        mixed_cell_rms = mixed.detach().float().square().mean(dim=(1, 2)).sqrt()
        enhanced_cell_rms = enhanced.detach().float().square().mean(dim=(1, 2)).sqrt()
        delta_cell_rms = (mixed.detach().float() - x.detach().float()).square().mean(dim=(1, 2)).sqrt()
        input_high_frequency_ratio = self._high_frequency_ratio_per_cell(x)
        mixed_high_frequency_ratio = self._high_frequency_ratio_per_cell(mixed)
        self.last_cell_diagnostics = {
            "mixed_delta_ratio": delta_cell_rms / (input_cell_rms + 1e-8),
            "enhanced_input_rms_ratio": enhanced_cell_rms / (input_cell_rms + 1e-8),
            "input_high_frequency_ratio": input_high_frequency_ratio,
            "mixed_high_frequency_ratio": mixed_high_frequency_ratio,
        }
        weights_mean = weights.detach().float().mean(dim=0).flatten()
        diagnostics = {
            "lag_gamma": float(gamma.detach().mean().item()),
            "lag_input_rms": float(input_rms.item()),
            "lag_mixed_rms": float(mixed_rms.item()),
            "lag_enhanced_rms": float(enhanced_rms.item()),
            "lag_mixed_delta_ratio": float((delta_rms / (input_rms + 1e-8)).item()),
            "lag_enhanced_input_rms_ratio": float((enhanced_rms / (input_rms + 1e-8)).item()),
            "lag_input_high_frequency_ratio": self._high_frequency_ratio(x),
            "lag_mixed_high_frequency_ratio": self._high_frequency_ratio(mixed),
        }
        for index, value in enumerate(weights_mean.tolist()):
            lag = index - self.kernel_size // 2
            name = f"lag_weight_{lag:+d}".replace("+", "p").replace("-", "m")
            diagnostics[name] = float(value)
        self.last_diagnostics = diagnostics

    @staticmethod
    def _high_frequency_ratio_per_cell(x: torch.Tensor) -> torch.Tensor:
        """按 range cell 计算慢时间高频能量占比，返回 `[B,N]`。"""
        if x.size(2) < 4:
            return torch.zeros(
                (x.size(0), x.size(3)),
                device=x.device,
                dtype=torch.float32,
            )
        spectrum = torch.fft.rfft(x.detach().float(), dim=2)
        energy = spectrum.abs().square()
        frequencies = torch.fft.rfftfreq(x.size(2), device=x.device)
        high_frequency = frequencies >= 0.25
        total = energy.sum(dim=(1, 2))
        high = energy[:, :, high_frequency, :].sum(dim=(1, 2))
        return high / (total + 1e-8)
