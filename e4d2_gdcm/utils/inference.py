"""Inference with pretrained E4-D2 model.

Usage:
    from e4d2 import load_model, run_inference

    model, prep = load_model('checkpoints/e4d2_best.pt')
    probs = run_inference(model, prep, X_complex, device='cuda')
    # probs: [B, N] target probability per range bin
"""

import os, sys
import numpy as np
import torch

from e4d2_gdcm.models.e4d2 import E4D2
from e4d2_gdcm.utils.preprocess import E4D2Preprocessor


def load_model(checkpoint_path, device='cuda'):
    """Load pretrained E4-D2 model and its preprocessor configuration.

    Args:
        checkpoint_path: path to .pt checkpoint saved by train.py
        device: 'cuda' or 'cpu'

    Returns:
        model: E4D2 instance in eval mode
        preprocessor: E4D2Preprocessor fitted with training stats
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = E4D2()
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    prep = E4D2Preprocessor()
    prep.P99 = ckpt['preprocessor_P99']
    prep._phase_mean = ckpt['preprocessor_phase_mean']
    prep._phase_std  = ckpt['preprocessor_phase_std']
    prep._global_mean = ckpt['preprocessor_global_mean']
    prep._global_std  = ckpt['preprocessor_global_std']
    prep._amp_cell_mean = ckpt['preprocessor_amp_cell_mean']
    prep._amp_cell_std  = ckpt['preprocessor_amp_cell_std']
    prep._fitted = True

    model.to(device).eval()
    return model, prep


def run_inference(model, preprocessor, X_complex, device='cuda'):
    """Run inference on complex-valued radar data.

    Args:
        model: E4D2 model
        preprocessor: fitted E4D2Preprocessor
        X_complex: [B, 4, 256] complex64 numpy array
        device: torch device

    Returns:
        target_probs: [B, 256] float32 numpy array
            Probability of target at each range bin (threshold at PFA=0.001).
    """
    inputs = preprocessor.transform(X_complex)
    with torch.no_grad():
        lo = model(
            torch.tensor(inputs['x_main'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_phase_cell'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_amp_map'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_amp_cell'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_global'], dtype=torch.float32, device=device),
        )
    return torch.softmax(lo, 1)[:, 1, :].cpu().numpy()  # target probability


def calibrate_threshold(model, preprocessor, clutter_data, target_pfa=0.001, device='cuda'):
    """Calibrate detection threshold from pure clutter samples.

    Args:
        model: E4D2 model
        preprocessor: fitted E4D2Preprocessor
        clutter_data: [N, 4, 256] complex64 array — pure clutter (no targets)
        target_pfa: desired false alarm rate (default 0.001)

    Returns:
        threshold: float, clutter score below which → "target"
    """
    inputs = preprocessor.transform(clutter_data)
    with torch.no_grad():
        lo = model(
            torch.tensor(inputs['x_main'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_phase_cell'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_amp_map'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_amp_cell'], dtype=torch.float32, device=device),
            torch.tensor(inputs['x_global'], dtype=torch.float32, device=device),
        )
    clutter_scores = torch.softmax(lo, 1)[:, 0, :].cpu().numpy().flatten()
    sorted_scores = np.sort(clutter_scores)
    Nc = len(sorted_scores)
    th_idx = max(0, min(int(np.ceil(target_pfa * Nc)) - 1, Nc - 1))
    return float(sorted_scores[th_idx])


def detect(model, preprocessor, X_complex, threshold, device='cuda'):
    """Binary detection at calibrated threshold.

    Returns:
        decisions: [B, 256] bool array, True = target detected
    """
    probs = run_inference(model, preprocessor, X_complex, device)
    clutter_score = 1.0 - probs
    return clutter_score <= threshold
