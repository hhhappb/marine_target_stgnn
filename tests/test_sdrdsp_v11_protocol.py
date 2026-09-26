from __future__ import annotations

from collections import Counter

import torch

from paper_modules.experiments.train import ScrBalancedBatchSampler
from scripts.preprocess_sdrdsp_v11 import choose_targets, window_starts


def test_v11_window_counts_follow_document_boundary() -> None:
    assert len(window_starts(6940)) == 1734
    assert len(window_starts(6520)) == 1629
    assert window_starts(6940)[-1] == 6932


def test_v11_target_positions_are_deterministic_and_separated() -> None:
    first = choose_targets(1000)
    second = choose_targets(1000)
    assert first == second
    assert len(first) == 5
    assert min(b - a for a, b in zip(first, first[1:])) >= 10


def test_scr_balanced_batches_rotate_extra_slots() -> None:
    scr_values = torch.tensor([scr for scr in range(14) for _ in range(24)])
    sampler = ScrBalancedBatchSampler(scr_values, batch_size=24, seed=42)
    iterator = iter(sampler)
    first = Counter(scr_values[next(iterator)].tolist())
    second = Counter(scr_values[next(iterator)].tolist())
    assert set(first.values()) == {1, 2}
    assert set(second.values()) == {1, 2}
    assert max(first.values()) - min(first.values()) == 1
    assert first != second
