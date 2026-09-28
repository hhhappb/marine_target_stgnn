"""SE preprocessing re-export.

The SE preprocessor is exactly ``selected_e4.preprocess.E4D2Preprocessor``.
It is re-exported here (not re-implemented) so that ``SE.utils.preprocess``
mirrors the layout of the reference package while keeping a single source of
truth for the feature contract:

    x_main       [B, 4, P, N]  I/P99, Q/P99, PL_raw, RPH/P99
    x_phase_cell [B, 3, N]     temporal_mean, temporal_std, phase_stability
    x_amp_map    [B, 1, P, N]  log1p(|X|/P99) envelope
    x_amp_cell   [B, 4, N]     amp_mean, amp_std, amp_p95, amp_peak_ratio
    x_global     [B, 6]        global_rms, p95, p99, max, mean, std

Fit on training echoes only; ``transform`` returns a dict of the 5 tensors.
"""

try:  # package dir on sys.path
    from selected_e4.preprocess import (
        E4D2Preprocessor,
        compute_pl_raw, compute_rph_raw,
        compute_cell_features, compute_global_features, compute_amp_features,
    )
except ImportError:  # `SE` package
    from SE.selected_e4.preprocess import (
        E4D2Preprocessor,
        compute_pl_raw, compute_rph_raw,
        compute_cell_features, compute_global_features, compute_amp_features,
    )

__all__ = [
    'E4D2Preprocessor',
    'compute_pl_raw', 'compute_rph_raw',
    'compute_cell_features', 'compute_global_features', 'compute_amp_features',
]
