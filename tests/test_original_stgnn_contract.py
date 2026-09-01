from __future__ import annotations

import pytest
import torch

from paper_modules.models.original_stgnn import OriginalSTGNN


@pytest.mark.parametrize(
    ("pulses", "expected_tfe2_length"),
    [(4, 1), (16, 4)],
)
def test_original_stgnn_keeps_detector_interface_for_long_windows(
    pulses: int,
    expected_tfe2_length: int,
) -> None:
    model = OriginalSTGNN(
        {"model": {"name": "original_stgnn", "pulses": pulses, "range_cells": 14}}
    ).eval()
    echoes = torch.randn(1, pulses, 14, dtype=torch.complex64)

    with torch.no_grad():
        logits, features = model(echoes, return_features=True)

    tfe2_out = features["temporal_features"]
    assert tuple(tfe2_out.shape) == (1, 1024, expected_tfe2_length, 14)
    assert tuple(logits.shape) == (1, 2, 14)
    assert torch.isfinite(logits).all()
    if expected_tfe2_length == 1:
        expected_logits = model.backbone.detector(tfe2_out.squeeze(2))
        assert torch.equal(logits, expected_logits)
