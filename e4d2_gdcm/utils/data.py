"""SDRDSP dataset preprocessing for E4-D2 model (bugfixed v2).

Key fixes vs v1:
  - h5py is lazy-imported (no crash if absent)
  - Data is explicitly cropped to self.N range bins after load
  - hash() replaced with deterministic numpy RandomState
  - STEP=4 (non-overlapping segments, not 8)

Supports two data formats:
  1. Scipy .mat  (e.g. 20210106155330_01_staring.mat)       → scipy.io.loadmat
  2. HDF5  .mat  (e.g. 20250124112948_2003_ST_.../IQData.mat) → h5py (optional)

Output format matches training protocol:
  - SCR-uniform sampling per batch
  - 5 non-overlapping targets per sample
  - Train: -12 to +14 dB (step 2), random target positions & speeds
  - Test:  -24 to +14 dB (step 2), fixed target position (idx 128) & speed (0.4 m/s)

Usage:
    python -m e4d2.sdrdsp \\
        --train data/20210106155330_01_staring.mat \\
        --test  data/20210106155432_01_staring.mat \\
        --output ./data/e4d2_dataset

    # Or programmatically:
    from e4d2_gdcm.sdrdsp import SDRDSPBuilder
    builder = SDRDSPBuilder()
    builder.build(train_mat, test_mat, './data/e4d2_dataset')
"""

import os, sys, argparse
import numpy as np


# ────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────

P = 4                 # pulses per segment
N_RANGE = 256          # range bins
STEP = P               # non-overlapping stride  (was: 8)
PRT = 1.0 / 1600       # pulse repetition interval (s)
WAVELENGTH = 0.03      # radar wavelength (m)
TARGET_POSITIONS = 5   # per sample
REF_WINDOW = 20        # reference cells for clutter power estimation
TARGET_PULSE_IDX = 1955  # original centre pulse


# ────────────────────────────────────────────────────────────
#  File loaders  (lazy h5py — BUGFIX)
# ────────────────────────────────────────────────────────────

def load_scipy_mat(path, N_range=N_RANGE, pulse_offset=TARGET_PULSE_IDX):
    """Load complex data from a scipy-style .mat file, crop to N_range."""
    from scipy.io import loadmat
    mat = loadmat(path)
    for key in mat:
        if not key.startswith('__'):
            data = mat[key]
            if np.iscomplexobj(data):
                break
            if 'real' in key.lower():
                for k2 in mat:
                    if 'imag' in k2.lower():
                        data = data + 1j * mat[k2]
                        break
                break
    else:
        raise ValueError(f"No complex data found in {path}")

    SB = pulse_offset - N_range // 2
    return data.astype(np.complex64)[:, SB:SB + N_range]


def _try_load_hdf5_mat(path, N_range=N_RANGE, pulse_offset=TARGET_PULSE_IDX):
    """Attempt HDF5 .mat load; returns None if h5py unavailable or format mismatch."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, 'r') as f:
            # Try I/Q separate keys
            if 'I' in f and 'Q' in f:
                I = f['I'][:].T.astype(np.float64)
                Q = f['Q'][:].T.astype(np.float64)
                data = (I + 1j * Q).astype(np.complex64)
                SB = pulse_offset - N_range // 2
                return data[:, SB:SB + N_range]

            # Try amplitude_data (h5py stores transposed)
            if 'amplitude_data' in f:
                dset = f['amplitude_data']
                data = dset['real'][:] + 1j * dset['imag'][:]
                data = data.astype(np.complex64).T
                SB = pulse_offset - N_range // 2
                return data[:, SB:SB + N_range]

            # Generic: look for any complex dataset
            for key in f:
                if hasattr(f[key], 'shape') and len(f[key].shape) >= 2:
                    dset = f[key]
                    try:
                        data = dset['real'][:] + 1j * dset['imag'][:]
                        data = data.astype(np.complex64)
                        SB = pulse_offset - N_range // 2
                        return data[:, SB:SB + N_range]
                    except (KeyError, ValueError):
                        continue
    except Exception:
        pass
    return None


def auto_load_mat(path, N_range=N_RANGE, pulse_offset=TARGET_PULSE_IDX):
    """Load a .mat file, auto-detecting scipy vs HDF5 format."""
    try:
        return load_scipy_mat(path, N_range, pulse_offset)
    except Exception:
        pass
    result = _try_load_hdf5_mat(path, N_range, pulse_offset)
    if result is not None:
        return result
    raise RuntimeError(f"Cannot load {path} — tried scipy.loadmat and h5py")


# ────────────────────────────────────────────────────────────
#  Target simulation  (deterministic — no hash())
# ────────────────────────────────────────────────────────────

def gen_positions(seed, N=N_RANGE):
    """Generate non-overlapping target range bin indices (deterministic)."""
    rng = np.random.RandomState(seed + 1000)
    chosen = []
    available = list(range(N))
    for _ in range(TARGET_POSITIONS):
        if not available:
            break
        pv = rng.choice(available)
        chosen.append(pv)
        for r in range(max(0, pv - 10), min(N, pv + 11)):
            if r in available:
                available.remove(r)
    return chosen


def gen_speed(seed):
    """Random radial speed in [0.1, 0.5] m/s (deterministic)."""
    return np.random.RandomState(seed + 2000).uniform(0.1, 0.5)


def inject_target(segment, positions, speed, scr_db, ref_window=REF_WINDOW, rng_seed=0):
    """Inject synthetic moving targets (deterministic — no hash()).

    BUGFIX: uses numpy RandomState with explicit seed, not Python hash().
    """
    seg = segment.copy()
    N = seg.shape[1]

    available = [i for i in range(N) if i not in positions]
    rng = np.random.RandomState(rng_seed)
    ref_indices = rng.choice(available, min(ref_window, len(available)), replace=False)

    clutter_power = np.sum(np.abs(seg[:, ref_indices]) ** 2)
    amp = np.sqrt(clutter_power * (10 ** (scr_db / 10.0)))

    for p in range(P):
        doppler_phase = 4.0 * np.pi * speed * PRT * p / WAVELENGTH
        for pos in positions:
            seg[p, pos] += amp * np.exp(1j * doppler_phase)

    labels = np.zeros(N, dtype=np.int32)
    for pos in positions:
        labels[pos] = 1
    return seg, labels


# ────────────────────────────────────────────────────────────
#  Sliding window segmenter
# ────────────────────────────────────────────────────────────

def extract_segments(data, P=P, step=STEP):
    """Extract all valid P-pulse segments from raw pulse-range matrix.

    BUGFIX: step=P=4 for non-overlapping windows (was 8).
    """
    total, N = data.shape
    segs = []
    for i in range(0, total - P + 1, step):
        segs.append(data[i:i + P, :])
    return np.array(segs, dtype=np.complex64)


# ────────────────────────────────────────────────────────────
#  Dataset builder
# ────────────────────────────────────────────────────────────

class SDRDSPBuilder:
    """Build SDRDSP-style training & test datasets (bugfixed v2)."""

    def __init__(self, P=4, N=256, pulse_offset=TARGET_PULSE_IDX):
        self.P = P
        self.N = N
        self.pulse_offset = pulse_offset

    def build(self, train_mat, test_mat, output_dir,
              train_scr=(-12, 15), test_scr=(-24, 15),
              segs_per_scr=80, seed_offset=0):
        """Build SCR-uniform training + test datasets.

        BUGFIX: data is explicitly cropped to self.N after loading.
        """
        print(f"Loading train: {train_mat}")
        train_full = auto_load_mat(train_mat, N_range=self.N, pulse_offset=self.pulse_offset)
        print(f"  shape after crop→{self.N}: {train_full.shape}")

        print(f"Loading test:  {test_mat}")
        test_full = auto_load_mat(test_mat, N_range=self.N, pulse_offset=self.pulse_offset)
        print(f"  shape after crop→{self.N}: {test_full.shape}")

        train_segs = extract_segments(train_full, self.P, STEP)
        test_segs  = extract_segments(test_full, self.P, STEP)
        print(f"Train segments: {train_segs.shape[0]} (stride={STEP})")
        print(f"Test segments:  {test_segs.shape[0]} (stride={STEP})")

        train_scr_list = list(range(train_scr[0], train_scr[1], 2))
        test_scr_list  = list(range(test_scr[0], test_scr[1], 2))

        # ── Train set ──
        print(f"\nBuilding training set (SCR {train_scr_list[0]}→{train_scr_list[-1]} dB)...")
        X_tr, y_tr, s_tr = [], [], []
        for scr in train_scr_list:
            for si in range(segs_per_scr):
                idx = np.random.randint(0, len(train_segs))
                seg = train_segs[idx].copy()
                positions = gen_positions(seed_offset + scr * 10000 + si)
                speed = gen_speed(seed_offset + scr * 10000 + si + 1)
                seg, labels = inject_target(seg, positions, speed, scr,
                                            rng_seed=abs(seed_offset + scr * 10000 + si) % (2 ** 31))
                X_tr.append(seg)
                y_tr.append(labels)
                s_tr.append(scr)
            print(f"  SCR={scr:>4d} dB: {segs_per_scr} samples")

        # ── Test set ──
        print(f"\nBuilding test set (SCR {test_scr_list[0]}→{test_scr_list[-1]} dB)...")
        X_te, y_te, s_te = [], [], []
        for scr in test_scr_list:
            for si in range(segs_per_scr):
                idx = np.random.randint(0, len(test_segs))
                seg = test_segs[idx].copy()
                seg, labels = inject_target(seg, [self.N // 2], 0.4, scr,
                                            rng_seed=abs(seed_offset + 100000 + scr * 10000 + si) % (2 ** 31))
                X_te.append(seg)
                y_te.append(labels)
                s_te.append(scr)
            print(f"  SCR={scr:>4d} dB: {segs_per_scr} samples")

        # ── Pack & save ──
        X_tr = np.array(X_tr, dtype=np.complex64)
        y_tr = np.array(y_tr, dtype=np.int32)
        s_tr = np.array(s_tr, dtype=np.int32)
        X_te = np.array(X_te, dtype=np.complex64)
        y_te = np.array(y_te, dtype=np.int32)
        s_te = np.array(s_te, dtype=np.int32)

        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, 'train_data.npz'),
                 data=X_tr, labels=y_tr, scr=s_tr)
        np.savez(os.path.join(output_dir, 'test_data.npz'),
                 data=X_te, labels=y_te, scr=s_te)

        print(f"\nSaved to {output_dir}/")
        print(f"  Train: {X_tr.shape}  targets={y_tr.sum():,}")
        print(f"  Test:  {X_te.shape}  targets={y_te.sum():,}")


# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build SDRDSP dataset for E4-D2 (v2)')
    parser.add_argument('--train', required=True, help='Training clutter .mat file')
    parser.add_argument('--test',  required=True, help='Test clutter .mat file')
    parser.add_argument('--output', default='./data/e4d2_dataset')
    parser.add_argument('--segs-per-scr', type=int, default=80)
    parser.add_argument('--n-range', type=int, default=256)
    parser.add_argument('--pulse-offset', type=int, default=TARGET_PULSE_IDX)
    args = parser.parse_args()

    builder = SDRDSPBuilder(P=4, N=args.n_range, pulse_offset=args.pulse_offset)
    builder.build(
        train_mat=args.train,
        test_mat=args.test,
        output_dir=args.output,
        segs_per_scr=args.segs_per_scr,
    )
    print("Done!")


if __name__ == '__main__':
    main()
