"""Preprocessing for E4-D2 model input.

All functions operate on complex-valued numpy arrays of shape [N_samples, P, N_range].

Inputs produced:
  x_main       [B, 4, P, N]  I/P99, Q/P99, PL_raw, RPH/P99
  x_phase_cell [B, 3, N]     temporal_mean, temporal_std, phase_stability (z-scored)
  x_amp_map    [B, 1, P, N]  log1p(|X|/P99) envelope
  x_amp_cell   [B, 4, N]     amp_mean, amp_std, amp_p95, amp_peak_ratio (z-scored)
  x_global     [B, 6]        global_rms, p95, p99, max, mean, std (z-scored)

Dependencies: numpy, scipy (for .mat loading)
"""

import numpy as np


# ────────────────────────────────────────────────────────────
#  Phase Linearity (PL)
# ────────────────────────────────────────────────────────────

def compute_pl_raw(X_complex):
    """Phase Linearity: |corr(p, unwrapped_phase)| across pulses.

    Args:
        X_complex: [B, P, N] complex array
    Returns:
        PL: [B, N] float32, raw values in [0, ~1.15]
    """
    eps = 1e-10
    B, P, N = X_complex.shape
    pidx = np.arange(P, dtype=np.float32)
    pm = pidx.mean()
    psv = np.sqrt(((pidx - pm) ** 2).mean()) + eps

    ph = np.angle(X_complex)
    pu = np.zeros_like(ph)
    pu[:, 0, :] = ph[:, 0, :]
    for p in range(1, P):
        d = ph[:, p, :] - ph[:, p - 1, :]
        pu[:, p, :] = pu[:, p - 1, :] + np.arctan2(np.sin(d), np.cos(d))

    pmv = pu.mean(axis=1, keepdims=True)
    pdmv = pu - pmv
    pd = pidx.reshape(1, -1, 1) - pm
    cov = (pdmv * pd).sum(axis=1) / (P - 1)
    vp = (pdmv ** 2).sum(axis=1) / (P - 1) + eps
    r = cov / (psv * np.sqrt(vp) + eps)
    return np.abs(r).astype(np.float32)


# ────────────────────────────────────────────────────────────
#  Relative Peak Height (RPH) — Doppler
# ────────────────────────────────────────────────────────────

def compute_rph_raw(X_complex):
    """Relative Peak Height: max(|FFT|) / mean(|FFT|).

    Args:
        X_complex: [B, P, N] complex array
    Returns:
        RPH: [B, N] float32, raw values in [~1, ~4+]
    """
    eps = 1e-10
    fa = np.abs(np.fft.fft(X_complex, axis=1))
    return (fa.max(axis=1) / (fa.mean(axis=1) + eps)).astype(np.float32)


# ────────────────────────────────────────────────────────────
#  Cell-level features
# ────────────────────────────────────────────────────────────

def compute_cell_features(X_complex):
    """Compute 8 cell-level statistics per range bin (aggregated across pulses).

    X_complex: [B, P, N]
    Returns [B, 8, N]:
      [0] local_rms        sqrt(mean(|X|²)) across pulses
      [1] local_p95        P95 of |X| across pulses
      [2] local_mean       mean(|X|) across pulses
      [3] local_std        std(|X|) across pulses
      [4] local_contrast   mean(|X|) / (local_rms + eps)
      [5] temporal_mean    |mean(X)| across pulses
      [6] temporal_std     std(angle(X)) across pulses
      [7] phase_stability  1 - temporal_std/<CE><80>
    """
    eps = 1e-10
    amp = np.abs(X_complex)                           # [B, P, N]
    lr = np.sqrt(np.mean(amp ** 2, axis=1) + eps)     # [B, N]
    lp = np.percentile(amp, 95, axis=1)               # [B, N]
    lm = np.mean(amp, axis=1)                         # [B, N]
    ls = np.std(amp, axis=1) + eps                    # [B, N]
    ct = lm / (lr + eps)                              # [B, N]
    tm = np.abs(np.mean(X_complex, axis=1))           # [B, N]
    ts = np.std(np.angle(X_complex), axis=1) + eps    # [B, N]
    ps = 1.0 - ts / np.pi                             # [B, N]
    return np.stack([lr, lp, lm, ls, ct, tm, ts, ps], axis=1).astype(np.float32)


# ────────────────────────────────────────────────────────────
#  Global features
# ────────────────────────────────────────────────────────────

def compute_global_features(X_complex):
    """Compute 6 global (sample-level) statistics.

    X_complex: [B, P, N]
    Returns [B, 6]:
      [0] global_rms    sqrt(mean(|X|<B2>))
      [1] global_p95    P95(|X|)
      [2] global_p99    P99(|X|)
      [3] global_max    max(|X|)
      [4] global_mean   mean(|X|)
      [5] global_std    std(|X|)
    """
    eps = 1e-10
    amp = np.abs(X_complex).reshape(len(X_complex), -1)    # [B, P*N]
    return np.stack([
        np.sqrt(np.mean(amp ** 2, axis=1) + eps),
        np.percentile(amp, 95, axis=1),
        np.percentile(amp, 99, axis=1),
        amp.max(axis=1),
        amp.mean(axis=1),
        amp.std(axis=1) + eps,
    ], axis=1).astype(np.float32)


# ────────────────────────────────────────────────────────────
#  Raw Polar features  (for Raw-Polar E4)
# ────────────────────────────────────────────────────────────

def compute_polar_raw(X_complex, P99_scale):
    """Compute raw polar representation [A, cosθ, sinθ] per pulse.

    Args:
        X_complex: [B, P, N] complex64  (P=4 pulses, N=256 range)
        P99_scale: float, P99 of |training clutter| for normalisation

    Returns:
        polar_amp:   [B, 1, P, N] float32   A = |X| / P99
        polar_phase: [B, 2, P, N] float32   [cosθ, sinθ] = [I/A, Q/A]
    """
    eps = 1e-10
    X = X_complex
    A = np.abs(X)                                       # [B, P, N]
    A_norm = (A / P99_scale).astype(np.float32)         # [B, P, N]
    C = (X.real / (A + eps)).astype(np.float32)         # cosθ
    S = (X.imag / (A + eps)).astype(np.float32)         # sinθ
    polar_amp = A_norm[:, np.newaxis, :, :]             # [B, 1, P, N]
    polar_phase = np.stack([C, S], axis=1)              # [B, 2, P, N]
    return polar_amp, polar_phase


def compute_cs_polar(X_complex, P99_scale, tau=None):
    """Confidence-scaled polar: phase direction weighted by amplitude reliability.

    Phase is unreliable when A≈0 (noise dominates). We gate the unit
    direction [cosθ, sinθ] by g(A) = A/(A+τ):

      C_g = g(A)·cosθ,   S_g = g(A)·sinθ

    g(A) ∈ (0, 1): weak amplitude → low phase-confidence weight.
    This uses raw amplitude as phase reliability — no hand-crafted
    statistics. Avoids degenerate I/Q reconstruction since A and
    [C_g, S_g] still pass through separate learnable encoders.

    Args:
        X_complex: [B, P, N] complex64
        P99_scale: float, P99 of |training clutter|
        tau: gating parameter. If None, defaults to 0.1 × P99_scale
             (g(0.1·P99) = 0.5).

    Returns:
        polar_amp:   [B, 1, P, N] float32   A = |X| / P99
        polar_phase: [B, 2, P, N] float32   [g(A)cosθ, g(A)sinθ]
    """
    eps = 1e-10
    if tau is None:
        tau = 0.1 * P99_scale
    X = X_complex
    A = np.abs(X)                                       # [B, P, N]
    A_norm = (A / P99_scale).astype(np.float32)         # [B, P, N]
    g = A / (A + tau)                                   # [B, P, N] reliability
    Cg = (g * X.real / (A + eps)).astype(np.float32)    # g(A)·cosθ
    Sg = (g * X.imag / (A + eps)).astype(np.float32)    # g(A)·sinθ
    polar_amp = A_norm[:, np.newaxis, :, :]             # [B, 1, P, N]
    polar_phase = np.stack([Cg, Sg], axis=1)            # [B, 2, P, N]
    return polar_amp, polar_phase


def compute_delta_cs(X_complex, P99_scale, eps=1e-6):
    """Delta-CS: adjacent-pulse relative phase direction [cosΔθ, sinΔθ].

    Replaces the ABSOLUTE phase representation [cosθ, sinθ] (vulnerable to
    the absolute-phase shortcut) with the PULSE-TO-PULSE phase difference:

      D_{p,r} = X_{p,r} · X_{p-1,r}^*  = A_p·A_{p-1}·e^{j(θ_p − θ_{p-1})}
      Q_{p,r} = D_{p,r} / |D_{p,r}|
      C_Δ = Re(Q),  S_Δ = Im(Q)

    This is globally phase-rotation invariant:
      DeltaCS(X·e^{jψ}) = DeltaCS(X)      (ψ ~ U(-π,π))

    and avoids -π↔π phase unwrapping by using the conjugate product directly.

    Padding (first frame has no predecessor → zero vector):
      C_Δ = [0, cosΔθ01, cosΔθ12, cosΔθ23]
      S_Δ = [0, sinΔθ01, sinΔθ12, sinΔθ23]

    The output keeps the SAME tensor shape [B, 2, P, N] as the absolute
    RawCS representation, so RawPhaseEncoder needs NO structural change.

    Numerical guard: denominator is clamped at eps (no new amplitude gate,
    for a fair comparison with RawCS).

    Args:
        X_complex: [B, P, N] complex64
        P99_scale: float, P99 of |training clutter|
        eps: clamp for the unit-normalisation denominator (default 1e-6)

    Returns:
        polar_amp:   [B, 1, P, N] float32   A = |X| / P99
        polar_phase: [B, 2, P, N] float32   [C_Δ, S_Δ], first pulse = 0
    """
    X = X_complex.astype(np.complex128)
    A = np.abs(X)                                       # [B, P, N]
    A_norm = (A / P99_scale).astype(np.float32)         # [B, P, N]

    D = X[:, 1:, :] * np.conj(X[:, :-1, :])             # [B, P-1, N]
    denom = np.abs(D)                                   # A_p·A_{p-1}
    denom = np.maximum(denom, eps)
    Q = D / denom                                       # unit complex diff
    C = Q.real.astype(np.float32)                       # cosΔθ
    S = Q.imag.astype(np.float32)                       # sinΔθ

    # Pad first pulse with zero vector (no predecessor)
    C = np.concatenate([np.zeros((X.shape[0], 1, X.shape[2]), dtype=np.float32), C], axis=1)
    S = np.concatenate([np.zeros((X.shape[0], 1, X.shape[2]), dtype=np.float32), S], axis=1)

    polar_amp = A_norm[:, np.newaxis, :, :]             # [B, 1, P, N]
    polar_phase = np.stack([C, S], axis=1)              # [B, 2, P, N]
    return polar_amp, polar_phase


# ────────────────────────────────────────────────────────────
#  Amplitude features
# ────────────────────────────────────────────────────────────

def compute_amp_features(X_complex, P99_scale):
    """Compute amplitude map and per-range-bin cell stats.

    BUGFIX v2: axis=1 (pulse dim, not axis=2 range dim).

    Args:
        X_complex: [B, P, N] complex64  (P=4 pulses, N=256 range)
        P99_scale: float, P99 of |training clutter| for normalisation

    Returns:
        amp_map:  [B, 1, P, N]  log1p(|X|/P99) envelope
        amp_cell: [B, 4, N]     amp_mean, amp_std, amp_p95, amp_peak_ratio
                                (aggregated across pulses → per range bin)
    """
    eps = 1e-10
    # amp_map: [B, P, N] — per-pulse, per-range log-amplitude
    amp_map = np.log1p(np.abs(X_complex) / P99_scale).astype(np.float32)
    amp_map_4d = amp_map[:, np.newaxis, :, :]           # [B, 1, P, N]

    # Aggregate across PULSES (axis=1) to get per-range-bin stats → [B, N]
    amp_mean = amp_map.mean(axis=1)                     # was: axis=2 (range)  ← BUG
    amp_std  = amp_map.std(axis=1) + eps                # was: axis=2          ← BUG
    amp_p95  = np.percentile(amp_map, 95, axis=1)       # was: axis=2          ← BUG
    amp_pr   = amp_p95 / (amp_mean + eps)

    amp_cell = np.stack([amp_mean, amp_std, amp_p95, amp_pr], axis=1).astype(np.float32)  # [B, 4, N]
    return amp_map_4d, amp_cell


# ────────────────────────────────────────────────────────────
#  Full preprocessing pipeline
# ────────────────────────────────────────────────────────────

class E4D2Preprocessor:
    """Preprocessor that fits statistics on training data and transforms all splits.

    Usage:
        prep = E4D2Preprocessor()
        prep.fit(X_train_complex)               # fit on training set
        inputs = prep.transform(X_complex)       # returns dict with all 5 tensors
    """

    def __init__(self):
        self.P99 = None
        self._phase_mean = None
        self._phase_std = None
        self._global_mean = None
        self._global_std = None
        self._amp_cell_mean = None
        self._amp_cell_std = None
        self._fitted = False

    def fit(self, X_train):
        """Fit all normalisation statistics from training complex data.

        Args:
            X_train: [N_train, P, N] complex64 array
        """
        self.P99 = float(np.percentile(np.abs(X_train).ravel(), 99) + 1e-10)

        # Cell features → fit on full 8ch, but we only use phase subset (ch 5,6,7)
        cell = compute_cell_features(X_train)                         # [N, 8, N_range]
        cell_log = np.log1p(np.abs(cell) + 1e-10)                    # log1p before stats
        self._phase_mean = cell_log.mean(axis=(0, 2), keepdims=True)  # [1, 8, 1]
        self._phase_std  = cell_log.std(axis=(0, 2), keepdims=True) + 1e-10

        # Global
        gb = compute_global_features(X_train)
        gb_log = np.log1p(np.abs(gb) + 1e-10)
        self._global_mean = gb_log.mean(axis=0, keepdims=True)       # [1, 6]
        self._global_std  = gb_log.std(axis=0, keepdims=True) + 1e-10

        # Amplitude cell
        _, amp_cell = compute_amp_features(X_train, self.P99)         # [N, 4, N_range]
        amp_cell_log = np.log1p(np.abs(amp_cell) + 1e-10)
        self._amp_cell_mean = amp_cell_log.mean(axis=(0, 2), keepdims=True)  # [1, 4, 1]
        self._amp_cell_std  = amp_cell_log.std(axis=(0, 2), keepdims=True) + 1e-10

        self._fitted = True
        return self

    def transform(self, X):
        """Transform complex data → model input dict.

        Args:
            X: [B, P, N] complex64 array

        Returns:
            dict with keys:
                x_main       [B, 4, P, N]  float32
                x_phase_cell [B, 3, N]     float32
                x_amp_map    [B, 1, P, N]  float32
                x_amp_cell   [B, 4, N]     float32
                x_global     [B, 6]        float32
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        eps = 1e-10
        real_part = X.real.astype(np.float32)
        imag_part = X.imag.astype(np.float32)

        # Main branch: [I/P99, Q/P99, PL_raw, RPH/P99] broadcast to all pulses
        PL = compute_pl_raw(X)                                       # [B, N]
        RPH = compute_rph_raw(X)                                     # [B, N]
        x_main = np.stack([
            real_part / self.P99,
            imag_part / self.P99,
            np.broadcast_to(PL[:, np.newaxis, :], (X.shape[0], X.shape[1], X.shape[2])),
            np.broadcast_to((RPH / self.P99)[:, np.newaxis, :], (X.shape[0], X.shape[1], X.shape[2])),
        ], axis=1).astype(np.float32)

        # Phase cell (only channels 5,6,7: temporal_mean, temporal_std, phase_stability)
        cell = compute_cell_features(X)                              # [B, 8, N]
        cell_log = np.log1p(np.abs(cell[:, [5, 6, 7], :]) + eps)    # [B, 3, N]
        x_phase_cell = ((cell_log - self._phase_mean[:, [5, 6, 7], :])
                         / (self._phase_std[:, [5, 6, 7], :] + eps)).astype(np.float32)

        # Amplitude
        x_amp_map, amp_cell_raw = compute_amp_features(X, self.P99)  # [B,1,P,N], [B,4,N]
        amp_cell_log = np.log1p(np.abs(amp_cell_raw) + eps)
        x_amp_cell = ((amp_cell_log - self._amp_cell_mean)
                      / (self._amp_cell_std + eps)).astype(np.float32)

        # Global
        gb = compute_global_features(X)
        gb_log = np.log1p(np.abs(gb) + eps)
        x_global = ((gb_log - self._global_mean) / (self._global_std + eps)).astype(np.float32)

        return {
            'x_main':       x_main,
            'x_phase_cell': x_phase_cell,
            'x_amp_map':    x_amp_map,
            'x_amp_cell':   x_amp_cell,
            'x_global':     x_global,
        }


# ────────────────────────────────────────────────────────────
#  Data loading  (compatible with original .mat format)
# ────────────────────────────────────────────────────────────

def load_mat_scipy(mat_path, N_range=256, pulse_offset=1955):
    """Load staring radar .mat file (scipy format) and crop to N_range.

    If the data is already N_range wide (pre-cropped), cropping is skipped.

    Args:
        mat_path:     path to .mat file
        N_range:      number of range bins to keep (default 256)
        pulse_offset: original centre pulse index (default 1955)

    Returns:
        data: [total_pulses, N_range] complex64
    """
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    key = [k for k in mat if not k.startswith('__')][0]
    data = mat[key].astype(np.complex64)

    # If data is already N_range wide, skip cropping (pre-cropped data)
    if data.shape[1] == N_range:
        return data

    SB = pulse_offset - N_range // 2
    return data[:, SB:SB + N_range]


def _try_load_hdf5_mat(mat_path):
    """Attempt HDF5 .mat load; returns None if h5py unavailable or format mismatch."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(mat_path, 'r') as f:
            # Try IQData keys
            if 'I' in f and 'Q' in f:
                I = f['I'][:].T.astype(np.float64)
                Q = f['Q'][:].T.astype(np.float64)
                return (I + 1j * Q).astype(np.complex64)
            # Try amplitude_data
            if 'amplitude_data' in f:
                dset = f['amplitude_data']
                data = dset['real'][:] + 1j * dset['imag'][:]
                return data.astype(np.complex64).T
            # Generic
            for key in f:
                if hasattr(f[key], 'shape') and len(f[key].shape) >= 2:
                    dset = f[key]
                    try:
                        data = dset['real'][:] + 1j * dset['imag'][:]
                        return data.astype(np.complex64)
                    except (KeyError, ValueError):
                        continue
    except Exception:
        pass
    return None


def auto_load_mat(mat_path):
    """Load a .mat file, auto-detecting scipy vs HDF5 format."""
    try:
        return load_mat_scipy(mat_path)
    except Exception:
        pass
    result = _try_load_hdf5_mat(mat_path)
    if result is not None:
        return result
    raise RuntimeError(f"Cannot load {mat_path} — tried scipy.loadmat and h5py")
