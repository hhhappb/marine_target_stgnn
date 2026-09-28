"""IPIX Dartmouth windowed ``.npz`` reader for the SE (P=8) model.

Data contract (as produced by the IPIX window preprocessing, e.g.
``window8_stride4_related``):

    E            [n, P, N] complex64   official ``ipixload`` auto-double
                                       processed echoes (P=8, N=14)
    range_roles  [N] int8/uint8        0 = clutter, 1 = secondary (SRC),
                                       2 = primary (PRC) target cell
    y_range      [n, N] uint8          stored label (primary policy)

Label policy used by the SE protocol is ``primary_strict``: only ``role == 2``
cells are positive; SRC cells are treated as clutter.

File naming: ``{stem}__{pol}__{split}.npz`` with ``split`` in {train, test}.

CLI (print shapes of a single file):

    python utils/data.py "<...>/19931109_191449_starea__hh__train.npz"
"""

import os
import argparse
import numpy as np

N_RANGE = 14           # IPIX range cells
PULSES = 8             # SE fixed pulse count
POLS = ("hh", "hv", "vh", "vv")

# Fig.7 x-axis label (1-based index) -> IPIX file stem.
DATASETS = {
    1: "19931107_135603_starea",
    2: "19931107_141630_starea",
    3: "19931107_145028_starea",
    4: "19931108_213827_starea",
    5: "19931108_220902_starea",
    6: "19931109_191449_starea",
    7: "19931109_202217_starea",
    8: "19931110_001635_starea",
    9: "19931111_163625_starea",
    10: "19931118_023604_stareC0000",
    11: "19931118_035737_stareC0000",
    12: "19931118_162155_stareC0000",
    13: "19931118_162658_stareC0000",
    14: "19931118_174259_stareC0000",
}
STEM_TO_LABEL = {stem: n for n, stem in DATASETS.items()}

QUICK8_POINTS = (
    ("label06", "hh", "hard"),
    ("label06", "hv", "hard"),
    ("label06", "vh", "hard"),
    ("label06", "vv", "hard"),
    ("label02", "hh", "easy"),
    ("label02", "hv", "easy"),
    ("label12", "vh", "easy"),
    ("label07", "vv", "easy"),
)

_QUICK8_STRATA = {(lb, pol): st for lb, pol, st in QUICK8_POINTS}
FULL56_POINTS = tuple(
    (f"label{n:02d}", pol, _QUICK8_STRATA.get((f"label{n:02d}", pol), "full"))
    for n in sorted(DATASETS)
    for pol in POLS
)


def split_path(npz_dir, stem, pol, split):
    """Path of one condition split file."""
    return os.path.join(npz_dir, f"{stem}__{pol}__{split}.npz")


def load_npz(npz_path):
    """Load a windowed ``.npz`` into a plain dict (as stored by the pipeline)."""
    d = np.load(npz_path)
    out = {k: d[k] for k in d.files}
    out["range_roles"] = np.asarray(d["range_roles"]).astype(np.int64)
    return out


def primary_strict_labels(roles, n_range=N_RANGE):
    """primary_strict label vector: only role == 2 is positive."""
    y = np.zeros(int(n_range), dtype=np.int64)
    y[np.asarray(roles) == 2] = 1
    return y


def load_split(npz_dir, stem, pol, split):
    """Load one condition split.

    Returns:
        E     [n, P, N] complex64
        y     [n, N] int64   primary_strict labels (broadcast per window)
        roles [N] int64      range-cell roles (0 clutter / 1 SRC / 2 PRC)
    """
    path = split_path(npz_dir, stem, pol, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"IPIX split not found: {path}")
    d = np.load(path)
    E = d["E"]
    roles = np.asarray(d["range_roles"]).astype(np.int64)
    y_vec = primary_strict_labels(roles, E.shape[2])
    y = np.broadcast_to(y_vec, (E.shape[0], E.shape[2])).copy()
    return E, y, roles


def resolve_condition(token):
    """Resolve a condition token to (label, stem, pol).

    Accepts either ``"<stem>:<pol>"`` (e.g. ``19931109_191449_starea:hh``) or
    ``"labelNN:<pol>"`` (e.g. ``label06:hh``).
    """
    if ":" not in token:
        raise ValueError(f"Condition must be 'stem:pol' or 'labelNN:pol', got {token!r}")
    head, pol = token.split(":", 1)
    if pol not in POLS:
        raise ValueError(f"Unknown polarization {pol!r}; expected one of {POLS}")
    if head.startswith("label") and head[-2:].isdigit():
        n = int(head[-2:])
        if n not in DATASETS:
            raise ValueError(f"Unknown label {head!r}; expected label01..label14")
        return head, DATASETS[n], pol
    if head not in STEM_TO_LABEL:
        raise ValueError(f"Unknown dataset stem {head!r}")
    return f"label{STEM_TO_LABEL[head]:02d}", head, pol


def main():
    ap = argparse.ArgumentParser(description="Inspect an IPIX windowed .npz")
    ap.add_argument("npz_path", help="path to a {stem}__{pol}__{split}.npz file")
    args = ap.parse_args()
    d = load_npz(args.npz_path)
    print(f"file        : {args.npz_path}")
    print(f"E shape     : {d['E'].shape}  dtype={d['E'].dtype}")
    print(f"range_roles : {d['range_roles'].shape}  {d['range_roles'].tolist()}")
    if "y_range" in d:
        print(f"y_range     : {d['y_range'].shape}  dtype={d['y_range'].dtype}")
    roles = d["range_roles"]
    n = d["E"].shape[2] if d["E"].ndim == 3 else N_RANGE
    print(f"primary_strict label: {primary_strict_labels(roles, n).tolist()}")


if __name__ == "__main__":
    main()
