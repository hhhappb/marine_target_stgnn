from __future__ import annotations

import pytest
import torch

from paper_modules.models.modules.spatial_graphs.radar_prior_dynamic_sfe import RadarPriorDynamicSFE
from paper_modules.models.modules.spatial_graphs.registry import build_spatial_graph


def test_dynamic_graph_keeps_only_local_neighbors_and_static_prior_stays_distant() -> None:
    torch.manual_seed(42)
    module = RadarPriorDynamicSFE(4, 8, static_delta=5)
    features = torch.randn(2, 4, 4, 14)
    index = torch.arange(14)
    local = (index[:, None] - index[None, :]).abs() <= 5

    dynamic = module._dynamic_adjacency(features, local)

    assert torch.count_nonzero(dynamic[:, ~local]) == 0
    torch.testing.assert_close(dynamic.sum(-1), torch.ones(2, 14))
    assert module._static_adjacency(14, features.device, features.dtype)[0, 13] > 0


def test_retired_nonlocal_parameter_fails_loudly() -> None:
    with pytest.raises(ValueError, match="dynamic_topk"):
        build_spatial_graph({"type": "radar_prior_dynamic_sfe", "dynamic_topk": 2}, 4, 8)
