from .spatial import RadarPriorDynamicSFE
from .temporal_attention import PulseAttentionOnlyTFE
from .temporal_gate import STGNNTemporalGate

def build_spatial_graph(config, in_channels, out_channels):
    if config.get('type') != 'radar_prior_dynamic_sfe' or 'dynamic_topk' in config:
        raise ValueError('Only selected local dynamic spatial module is supported')
    return RadarPriorDynamicSFE(in_channels, out_channels,
        static_gamma=float(config['static_gamma']), static_delta=int(config['static_delta']),
        static_weight=float(config['static_weight']), dynamic_temperature=float(config['dynamic_temperature']),
        dropout=float(config['dropout']))

def build_temporal_module(config, in_channels, out_channels):
    if config.get('type') == 'stgnn_tfe':
        return STGNNTemporalGate(in_channels, out_channels)
    if config.get('type') == 'pulse_attention_only_tfe':
        return PulseAttentionOnlyTFE(in_channels, out_channels,
            attention_dim=int(config['attention_dim']), num_heads=int(config['num_heads']),
            residual_scale=float(config['residual_scale']), use_attention=bool(config['use_attention']))
    raise ValueError('Only A1 and original temporal modules are supported')
