from __future__ import annotations

import torch.nn as nn

from .amplitude_phase import AmplitudePhaseFeatures
from .doppler_stats import DopplerStatsFeatures
from .phase_diffs import PhaseAmplitudeDiffFeatures
from .real_imag import RealImagFeatures


RADAR_FEATURES: dict[str, type[nn.Module]] = {
    "real_imag": RealImagFeatures,
    "real_imag_phase_amp": AmplitudePhaseFeatures,
    "real_imag_amp_phase": AmplitudePhaseFeatures,
    "real_imag_phase_amp_diffs": PhaseAmplitudeDiffFeatures,
    "phase_amp_diffs": PhaseAmplitudeDiffFeatures,
    "radar_prior_full": DopplerStatsFeatures,
    "full": DopplerStatsFeatures,
}


def build_raw_feature_extractor(feature_type: str) -> nn.Module:
    try:
        return RADAR_FEATURES[feature_type]()
    except KeyError as exc:
        known = ", ".join(sorted(RADAR_FEATURES))
        raise ValueError(f"Unknown radar feature type: {feature_type}. Known: {known}") from exc
