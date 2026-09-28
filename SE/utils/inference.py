"""Inference with a trained SE model.

Usage:
    from utils.inference import load_model, run_inference

    model, prep = load_model('checkpoints/se_best.pt')
    probs = run_inference(model, prep, X_complex)      # [B, N] target prob
    h     = calibrate_threshold(model, prep, clutter)  # Eq.15 train-clutter
    dec   = detect(model, prep, X_complex, h)          # [B, N] bool

The SE package ships no weights. ``load_model`` reads a checkpoint written by
``experiments.train_ipix`` / ``experiments.run_v21_ipix``: it expects
``model_state_dict`` (an ``E4Model`` state dict) and ``preprocessor`` (the
fitted :class:`E4D2Preprocessor` state, see :func:`preprocessor_state_dict`).
"""

import numpy as np
import torch

try:  # package dir on sys.path
    from selected_e4 import E4Model, E4D2Preprocessor
except ImportError:  # `SE` package
    from SE.selected_e4 import E4Model, E4D2Preprocessor

# The real fitted attributes of selected_e4.preprocess.E4D2Preprocessor.
PREPROCESSOR_FIELDS = (
    "P99",
    "_phase_mean", "_phase_std",
    "_global_mean", "_global_std",
    "_amp_cell_mean", "_amp_cell_std",
)

EPS = 1e-10


def preprocessor_state_dict(prep):
    """Serialise a fitted E4D2Preprocessor to a plain dict of numpy arrays."""
    if not getattr(prep, "_fitted", False):
        raise RuntimeError("Cannot serialise an unfitted preprocessor.")
    return {name: getattr(prep, name) for name in PREPROCESSOR_FIELDS}


def restore_preprocessor(state):
    """Rebuild a fitted E4D2Preprocessor from :func:`preprocessor_state_dict`."""
    prep = E4D2Preprocessor()
    for name in PREPROCESSOR_FIELDS:
        if name not in state:
            raise KeyError(f"preprocessor state is missing {name!r}")
        setattr(prep, name, state[name])
    prep._fitted = True
    return prep


def _normalise_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def load_model(checkpoint_path, device="cpu", strict=True):
    """Load a trained SE model plus its fitted preprocessor.

    Args:
        checkpoint_path: .pt file with ``model_state_dict`` + ``preprocessor``
        device: torch device string or torch.device
        strict: passed to ``load_state_dict``

    Returns:
        (model, preprocessor) with model in eval() on ``device``.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = E4Model()

    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt  # a bare state dict
    if state is None:
        raise KeyError("checkpoint has no 'model_state_dict'/'state_dict'")
    model.load_state_dict(state, strict=strict)

    if "preprocessor" in ckpt:
        prep = restore_preprocessor(ckpt["preprocessor"])
    elif "preprocessor_P99" in ckpt:  # e4d2-style flat keys
        prep = E4D2Preprocessor()
        prep.P99 = ckpt["preprocessor_P99"]
        prep._phase_mean = ckpt["preprocessor_phase_mean"]
        prep._phase_std = ckpt["preprocessor_phase_std"]
        prep._global_mean = ckpt["preprocessor_global_mean"]
        prep._global_std = ckpt["preprocessor_global_std"]
        prep._amp_cell_mean = ckpt["preprocessor_amp_cell_mean"]
        prep._amp_cell_std = ckpt["preprocessor_amp_cell_std"]
        prep._fitted = True
    else:
        raise KeyError("checkpoint has no preprocessor statistics")

    device = _normalise_device(device)
    model.to(device).eval()
    return model, prep


def _forward_logits(model, preprocessor, X_complex, device, chunk=512):
    """Preprocess + forward in chunks -> numpy logits [B, 2, N]."""
    device = _normalise_device(device)
    logits = []
    for i in range(0, len(X_complex), chunk):
        d = preprocessor.transform(X_complex[i:i + chunk])
        inp = {k: torch.tensor(v, dtype=torch.float32, device=device)
               for k, v in d.items()}
        with torch.no_grad():
            logits.append(model(**inp).cpu().numpy())
    return np.concatenate(logits, axis=0)


def run_inference(model, preprocessor, X_complex, device="cpu", temperature=1.0):
    """Target probability per range bin.

    Args:
        X_complex: [B, P, N] complex64
        temperature: softmax temperature (logits / T); > 1 de-saturates
    Returns:
        target_probs: [B, N] float32  = softmax(logits / T)[:, 1, :]
    """
    lo = _forward_logits(model, preprocessor, X_complex, device)
    return torch.softmax(torch.tensor(lo / float(temperature)), 1)[:, 1, :].numpy()


def estimate_temperature(model, preprocessor, clutter_data, device="cpu",
                         clip=(2.0, 16.0)):
    """Auto temperature = mean |logit| over clutter cells, clipped to ``clip``.

    The SE logits saturate on IPIX; without temperature scaling the Eq.15
    clutter-probability quantile collapses at h=1.0 and PFA is uncontrolled.
    """
    lo = _forward_logits(model, preprocessor, clutter_data, device)
    return float(np.clip(np.abs(lo).mean(), clip[0], clip[1]))


def calibrate_threshold(model, preprocessor, clutter_data, target_pfa=0.001,
                        device="cpu", temperature=1.0):
    """Eq.15 threshold from pure-clutter samples.

    Sorts the clutter probability ``o0 = softmax(logits / T)[:, 0, :]`` over all
    supplied clutter cells and takes ``o0_sorted[ceil(PFA*Nc) - 1]``.

    Args:
        clutter_data: [N, P, N_range] complex64, target-free echoes
    Returns:
        threshold: float; ``o0 <= threshold`` decides "target".
    """
    lo = _forward_logits(model, preprocessor, clutter_data, device)
    o0 = torch.softmax(torch.tensor(lo / float(temperature)), 1)[:, 0, :].numpy().ravel()
    o0.sort()
    n_c = o0.size
    idx = max(0, int(np.ceil(target_pfa * n_c)) - 1)
    return float(o0[min(idx, n_c - 1)])


def detect(model, preprocessor, X_complex, threshold, device="cpu", temperature=1.0):
    """Binary detection at a calibrated threshold.

    Returns:
        decisions: [B, N] bool, True = target detected.
    """
    probs = run_inference(model, preprocessor, X_complex, device, temperature)
    o0 = 1.0 - probs
    return o0 <= threshold
