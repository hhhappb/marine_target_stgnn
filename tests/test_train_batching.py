from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from paper_modules.experiments.train import train_one_epoch


class TinyComplexDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, echoes: torch.Tensor) -> torch.Tensor:
        feature = echoes.real.mean(dim=1) * self.scale
        return torch.stack([-feature, feature], dim=1)


def make_dataset() -> TensorDataset:
    real = torch.tensor(
        [[[1.0, -1.0]], [[2.0, -2.0]], [[3.0, -3.0]], [[4.0, -4.0]], [[5.0, -5.0]]]
    )
    imag = torch.zeros_like(real)
    labels = torch.tensor([[1, 0]] * 5)
    return TensorDataset(real, imag, labels)


def run_epoch(model: nn.Module, batch_size: int, accumulation_steps: int) -> dict[str, float]:
    loader = DataLoader(make_dataset(), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    _, metrics = train_one_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=1,
        epochs=1,
        show_progress=False,
        log_interval=0,
        grad_clip=None,
        gradient_accumulation_steps=accumulation_steps,
    )
    return metrics


def test_gradient_accumulation_matches_effective_batch_and_steps_tail_window() -> None:
    accumulated = TinyComplexDetector()
    direct = TinyComplexDetector()
    direct.load_state_dict(accumulated.state_dict())

    accumulated_metrics = run_epoch(accumulated, batch_size=1, accumulation_steps=2)
    direct_metrics = run_epoch(direct, batch_size=2, accumulation_steps=1)

    assert accumulated_metrics["optimizer_steps"] == 3
    assert direct_metrics["optimizer_steps"] == 3
    np.testing.assert_allclose(
        accumulated.scale.detach().numpy(),
        direct.scale.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_gradient_accumulation_rejects_non_positive_steps() -> None:
    model = TinyComplexDetector()
    loader = DataLoader(make_dataset(), batch_size=1)
    with pytest.raises(ValueError, match="必须为正整数"):
        train_one_epoch(
            model=model,
            loader=loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"),
            epoch=1,
            epochs=1,
            show_progress=False,
            log_interval=0,
            grad_clip=None,
            gradient_accumulation_steps=0,
        )


def test_training_metrics_and_logs_omit_accuracy(capsys: pytest.CaptureFixture[str]) -> None:
    model = TinyComplexDetector()
    loader = DataLoader(make_dataset(), batch_size=2, shuffle=False)
    _, metrics = train_one_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        epoch=1,
        epochs=1,
        show_progress=False,
        log_interval=1,
        grad_clip=None,
        gradient_accumulation_steps=1,
    )
    assert "accuracy" not in metrics
    assert "acc=" not in capsys.readouterr().out
