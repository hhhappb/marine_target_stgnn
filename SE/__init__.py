"""SE: selected P=8 E4 model -- self-contained package.

SE = the E4 parallel-representation backbone (ST-GNN 1024ch + PhaseEncoder 128
+ AmpEncoder 64 + CVOCAFusion 64 + global MLP 128 = 1216ch) with the *selected*
spatiotemporal slots instead of the classic STFE/TFE:

    spatial1 / spatial2 : radar_prior_dynamic_sfe    (RadarPriorDynamicSFE)
    temporal1           : pulse_attention_only_tfe   (A1 pulse attention, TFE1)
    temporal2           : stgnn_tfe                  (original gated TFE2)

then the GDCM-A0 detector head (``log(1 + |U_d - U_h|)`` judgment).

The canonical model core lives in ``selected_e4/`` and is neither copied nor
modified here; this package only re-exports it and adds the training /
inference scaffolding (utils, losses, configs, experiments, reports).

Fixed operating point: P = 8 pulses, N = 14 range cells (IPIX v2.x),
4,461,538 parameters.

Quick start::

    from SE import E4Model, E4D2Preprocessor
"""

from .selected_e4 import E4Model, E4D2Preprocessor
from .models import count_params

__version__ = "1.0"
__all__ = ["E4Model", "E4D2Preprocessor", "count_params"]
