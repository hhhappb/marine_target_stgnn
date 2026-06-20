import torch.nn as nn

from .spatial_graphs import build_spatial_graph


class SpatialGraphModule(nn.Module):
    """空间图模块包装器：按配置选择一个具体空间建模思路。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        graph_type: str = "local_3",
        k: int = 1,
        use_distance_decay: bool = False,
        distance_decay: float = 0.25,
        use_feature_similarity: bool = False,
        similarity_weight: float = 0.1,
    ):
        super().__init__()
        cfg = {
            "type": graph_type,
            "k": k,
            "use_distance_decay": use_distance_decay,
            "distance_decay": distance_decay,
            "use_feature_similarity": use_feature_similarity,
            "similarity_weight": similarity_weight,
        }
        self.impl = build_spatial_graph(cfg, in_channels, out_channels)

    def forward(self, x):
        return self.impl(x)
