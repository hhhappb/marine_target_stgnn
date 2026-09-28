"""SE model components.

The model core lives in the sibling package ``selected_e4`` and is re-exported
here unchanged -- the files under ``selected_e4/`` are NOT moved or copied, they
are the canonical, state-dict-verified implementation:

    E4Model                -- five feature inputs -> [B, 2, N] logits
    E4Backbone             -- ST-GNN 1024ch + PhaseEncoder 128 + AmpEncoder 64
                              + CVOCAFusion 64 + global MLP 128 = 1216ch
    GdcmDetector           -- GDCM-A0 head (log(1+|delta|) judgment)
    ConfiguredSTGNNBackbone-- the four-slot main branch (selected slots)
    RadarPriorDynamicSFE   -- spatial1/spatial2 (SE slot)
    PulseAttentionOnlyTFE  -- temporal1 (A1 pulse attention, SE slot)
    STGNNTemporalGate      -- temporal2 (original gated TFE, SE slot)
"""

try:  # run with the package dir on sys.path (e.g. `python -m experiments.train_ipix`)
    from selected_e4.model import E4Model
    from selected_e4.e4_backbone import E4Backbone
    from selected_e4.detector import GdcmDetector
    from selected_e4.encoders import PhaseEncoder, AmpEncoder, CVOCAFusion
    from selected_e4.stgnn_backbone import ConfiguredSTGNNBackbone
    from selected_e4.spatial import RadarPriorDynamicSFE
    from selected_e4.temporal_attention import PulseAttentionOnlyTFE
    from selected_e4.temporal_gate import STGNNTemporalGate
    from selected_e4.registry import build_spatial_graph, build_temporal_module
except ImportError:  # run as the `SE` package (e.g. `python -m SE.experiments.train_ipix`)
    from SE.selected_e4.model import E4Model
    from SE.selected_e4.e4_backbone import E4Backbone
    from SE.selected_e4.detector import GdcmDetector
    from SE.selected_e4.encoders import PhaseEncoder, AmpEncoder, CVOCAFusion
    from SE.selected_e4.stgnn_backbone import ConfiguredSTGNNBackbone
    from SE.selected_e4.spatial import RadarPriorDynamicSFE
    from SE.selected_e4.temporal_attention import PulseAttentionOnlyTFE
    from SE.selected_e4.temporal_gate import STGNNTemporalGate
    from SE.selected_e4.registry import build_spatial_graph, build_temporal_module


def count_params(model):
    """Return (n_params, human_readable_string) for an SE model."""
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e6:
        return n, f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return n, f"{n/1e3:.1f}K"
    return n, str(n)


__all__ = [
    'E4Model', 'E4Backbone', 'GdcmDetector',
    'PhaseEncoder', 'AmpEncoder', 'CVOCAFusion',
    'ConfiguredSTGNNBackbone', 'RadarPriorDynamicSFE',
    'PulseAttentionOnlyTFE', 'STGNNTemporalGate',
    'build_spatial_graph', 'build_temporal_module',
    'count_params',
]
