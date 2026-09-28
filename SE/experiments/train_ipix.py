#!/usr/bin/env python3
"""Train + evaluate the SE (selected P=8 E4) model on IPIX Dartmouth, v2.x.

Protocol (v2.1 / PAX-L1 compatible):
  * window8_stride4_related IPIX windows (P=8, N=14)
  * primary_strict labels (only role == 2 / PRC cells are positive)
  * range_roll augmentation: mode=clutter_fill, max_shift=3, include_identity,
    re-sampled per epoch on the COMPLEX echo (features re-transformed after)
  * Eq.15 threshold from ALL training clutter cells + auto temperature scaling
  * per-range-cell Pd / PF (PF_all_non_prc, PF_clutter_only, SRC rate)
  * SE model: selected_e4.E4Model (radar_prior_dynamic_sfe + A1 pulse-attention
    TFE1 + original gated TFE2) + GDCM-A0 detector head.

Run from ``/tmp/ST-GNN/SE`` with this directory on PYTHONPATH::

    PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.train_ipix \\
        --conditions 19931109_191449_starea:hh --epochs 60 \\
        --output checkpoints/se_ipix.json

The functions here are also imported by ``experiments.run_v21_ipix``.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:  # package dir on sys.path
    from selected_e4 import E4Model, E4D2Preprocessor
    from utils.data import (
        N_RANGE, PULSES, POLS, DATASETS, QUICK8_POINTS, FULL56_POINTS,
        load_split, resolve_condition,
    )
    from utils.inference import preprocessor_state_dict
    from losses.losses import get_loss
except ImportError:  # `SE` package
    from SE.selected_e4 import E4Model, E4D2Preprocessor
    from SE.utils.data import (
        N_RANGE, PULSES, POLS, DATASETS, QUICK8_POINTS, FULL56_POINTS,
        load_split, resolve_condition,
    )
    from SE.utils.inference import preprocessor_state_dict
    from SE.losses.losses import get_loss

DEFAULT_DATA_ROOT = "/tmp/ST-GNN/stgnn ipix/ipix_dartmouth/processed"
RANGE_ROLL_MAX_SHIFT = 3
RANGE_ROLL_INCLUDE_IDENTITY = True


# ------------------------------------------------------------------
#  Data
# ------------------------------------------------------------------

def subsample(E, y, max_samples, seed=0):
    """Deterministic subsample of the training windows (debug / speed)."""
    if max_samples is None or len(E) <= max_samples:
        return E, y
    idx = np.random.RandomState(seed).choice(len(E), max_samples, replace=False)
    return E[idx], y[idx].copy()


# ------------------------------------------------------------------
#  range_roll augmentation (v2.1, clutter_fill) on the complex echo
# ------------------------------------------------------------------

def _sample_shift(roles, max_shift, include_identity, rng):
    related = np.flatnonzero(roles > 0)
    if related.size == 0:
        return 0
    minimum = max(-max_shift, -int(related.min()))
    maximum = min(max_shift, N_RANGE - 1 - int(related.max()))
    candidates = np.arange(minimum, maximum + 1, dtype=np.int64)
    if not include_identity:
        candidates = candidates[candidates != 0]
    if candidates.size == 0:
        return 0
    return int(rng.choice(candidates))


def augment_train(E, y, roles, rng, max_shift=RANGE_ROLL_MAX_SHIFT,
                  include_identity=RANGE_ROLL_INCLUDE_IDENTITY):
    """Per-sample range_roll (clutter_fill) on the complex echo.

    E: [n, P, N] complex64; y: [n, N] int64 (primary_strict); roles: [N].
    Shifts the target-related neighbourhood, fills vacated cells with real
    clutter echoes, zeroes the vacated labels.
    """
    n, _, N = E.shape
    clutter = np.flatnonzero(roles == 0)
    shifts = np.zeros(n, dtype=np.int64)
    donors = {}
    for i in range(n):
        s = _sample_shift(roles, max_shift, include_identity, rng)
        shifts[i] = s
        if s != 0:
            donors[i] = clutter[rng.integers(0, clutter.size, size=abs(s))]
    E_aug = np.array(E, copy=True)
    y_aug = np.array(y, copy=True)
    for s in np.unique(shifts):
        if s == 0:
            continue
        idx = np.flatnonzero(shifts == s)
        if s > 0:
            E_aug[idx, :, s:] = E[idx, :, :-s]
            y_aug[idx, s:] = y[idx, :-s]
            vacated = np.arange(s)
        else:
            E_aug[idx, :, :s] = E[idx, :, -s:]
            y_aug[idx, :s] = y[idx, -s:]
            vacated = np.arange(N + s, N)
        for j, pos in enumerate(vacated):
            donor_col = np.array([donors[i][j] for i in idx], dtype=np.int64)
            E_aug[idx, :, pos] = E[idx, :, donor_col]
        y_aug[np.ix_(idx, vacated)] = 0
    return E_aug, y_aug


def prep_tensors(prep, E):
    """Complex [n,P,N] -> the 5 SE input tensors."""
    d = prep.transform(E)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}


# ------------------------------------------------------------------
#  FAR controller (Eq. 15) with SE temperature calibration
# ------------------------------------------------------------------

@torch.no_grad()
def calibrate_threshold(model, tensors, y, pfa, device, chunk=512):
    """Threshold h from TRAINING clutter cells (Eq.15) + auto temperature.

    Returns (h, Nc, train_pfa, T).
    """
    model.eval()
    n = y.shape[0]
    if n == 0:
        raise ValueError("calibrate_threshold received an empty tensor set")

    logit_abs = []
    for i in range(0, n, chunk):
        inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
        la = torch.abs(model(**inp)).cpu().numpy()          # [b,2,N]
        yb = y[i:i + chunk]
        logit_abs.append(la.transpose(0, 2, 1)[yb == 0].ravel())
    la_all = np.concatenate(logit_abs)
    if la_all.size == 0:
        raise ValueError("no clutter cells available for Eq.15 calibration")
    T = float(np.clip(la_all.mean(), 2.0, 16.0))

    o0_all = []
    for i in range(0, n, chunk):
        inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
        lo = model(**inp)
        o0 = torch.softmax(lo / T, 1)[:, 0, :].cpu().numpy()
        yb = y[i:i + chunk]
        o0_all.append(o0[yb == 0])
    o0_all = np.concatenate(o0_all)
    o0_all.sort()
    Nc = o0_all.size
    idx = max(0, int(np.ceil(pfa * Nc)) - 1)
    h = float(o0_all[min(idx, Nc - 1)])
    train_pfa = float(np.mean(o0_all <= h))
    return h, int(Nc), train_pfa, T


@torch.no_grad()
def evaluate(model, tensors, y, roles, h, device, T, chunk=512):
    """Per-range-cell detection metrics at threshold h (o0 <= h -> target)."""
    model.eval()
    o0_all, y_all = [], []
    n = y.shape[0]
    for i in range(0, n, chunk):
        inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
        lo = model(**inp)
        o0_all.append(torch.softmax(lo / T, 1)[:, 0, :].cpu().numpy())
        y_all.append(y[i:i + chunk])
    o0 = np.concatenate(o0_all)
    yb = np.concatenate(y_all)
    prc = roles == 2
    src = roles == 1
    clu = roles == 0
    det = o0 <= h
    tp = float((det & (yb == 1)).sum())
    fn = float((~det & (yb == 1)).sum())
    pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pfa_all = float(det[:, ~prc].mean())
    pfa_clu = float(det[:, clu].mean()) if clu.any() else float("nan")
    src_r = float(det[:, src].mean()) if src.any() else float("nan")
    return {"pd": pd, "pf_all_non_prc": pfa_all, "pf_clutter_only": pfa_clu,
            "src_target_rate": src_r}


# ------------------------------------------------------------------
#  Training
# ------------------------------------------------------------------

def train_condition(E_tr, y_tr, roles, epochs, batch_size, lr, seed, device,
                    grad_accum=8, loss_name="ce"):
    """Train one SE model on one (dataset, polarization) condition.

    Preprocessor statistics are fit on the ORIGINAL training echoes; range_roll
    augmentation is re-sampled per epoch and the features re-transformed so
    PL/RPH/cell/amp features stay aligned with the shifted target cell.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = E4Model().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = get_loss(loss_name)
    prep = E4D2Preprocessor().fit(E_tr)

    n = y_tr.shape[0]
    for _ in range(epochs):
        E_aug, y_aug = augment_train(E_tr, y_tr, roles, rng)
        tensors = prep_tensors(prep, E_aug)
        y_t = torch.tensor(y_aug, dtype=torch.long)
        model.train()
        perm = torch.randperm(n)
        opt.zero_grad()
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            inp = {k: v[idx].to(device) for k, v in tensors.items()}
            lb = y_t[idx].to(device)
            lo = model(**inp)
            loss = crit(lo.permute(0, 2, 1).reshape(-1, 2), lb.reshape(-1))
            loss.backward()
            if (b // batch_size + 1) % grad_accum == 0 or b + batch_size >= n:
                opt.step()
                opt.zero_grad()
    return model, prep


def save_checkpoint(path, model, prep, meta=None):
    """Save an SE checkpoint consumable by ``utils.inference.load_model``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "preprocessor": preprocessor_state_dict(prep),
        "meta": meta or {},
    }, path)


def run_point(E_tr, y_tr, roles_tr, E_te, y_te, roles_te, *, epochs, batch_size,
              lr, pfa, seed, device, grad_accum=8, max_train=None, loss_name="ce"):
    """Train + calibrate + evaluate one condition.

    Returns (result_dict, model, prep).
    """
    E_tr, y_tr = subsample(E_tr, y_tr, max_train)
    t0 = time.time()
    model, prep = train_condition(E_tr, y_tr, roles_tr, epochs, batch_size, lr,
                                  seed, device, grad_accum=grad_accum,
                                  loss_name=loss_name)
    te_tensors = prep_tensors(prep, E_te)
    h, Nc, train_pfa, T = calibrate_threshold(
        model, prep_tensors(prep, E_tr), y_tr, pfa, device)
    res = evaluate(model, te_tensors, y_te, roles_te, h, device, T)
    res.update({
        "train_windows": int(len(E_tr)), "test_windows": int(len(E_te)),
        "h": h, "cal_Nc": int(Nc), "train_pfa": train_pfa, "temperature": float(T),
        "train_seconds": round(time.time() - t0, 1),
    })
    return res, model, prep


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def _default_data_dir(data_root, pulses=PULSES):
    return os.path.join(data_root, f"window{pulses}_stride4_related")


def build_arg_parser(description="Train + evaluate SE (P=8) on IPIX Dartmouth"):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--pulses", type=int, default=PULSES, choices=[PULSES],
                    help="SE is compiled for P=8")
    ap.add_argument("--conditions", type=str, default=None,
                    help="comma list of 'stem:pol' (e.g. 19931109_191449_starea:hh); "
                         "default = quick-8")
    ap.add_argument("--all-56", dest="all_56", action="store_true",
                    help="use the full 56-point cohort (14 files x 4 pols)")
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--pfa", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--loss", type=str, default="ce")
    ap.add_argument("--save-checkpoint", type=str, default=None,
                    help="optional dir to write per-point SE checkpoints")
    ap.add_argument("--output", type=str, default="checkpoints/se_ipix.json")
    return ap


def resolve_points(conditions, all_56):
    """Return list of (label, stem, pol, stratum)."""
    strata_map = dict(((lb, p), st) for lb, p, st in QUICK8_POINTS)
    label_to_stem = {f"label{n:02d}": stem for n, stem in DATASETS.items()}
    if conditions:
        pts = []
        for token in conditions.split(","):
            label, stem, pol = resolve_condition(token.strip())
            pts.append((label, stem, pol, strata_map.get((label, pol), "full")))
        return pts
    source = FULL56_POINTS if all_56 else QUICK8_POINTS
    return [(lb, label_to_stem[lb], pol, st) for lb, pol, st in source]


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main():
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    data_dir = _default_data_dir(args.data_root, args.pulses)
    points = resolve_points(args.conditions, args.all_56)

    print(f"SE (selected P=8 E4) IPIX | layer="
          f"{'full56' if args.all_56 else ('custom' if args.conditions else 'quick8')} "
          f"| points={len(points)} | epochs={args.epochs} | batch={args.batch_size}"
          f"x{args.grad_accum} | lr={args.lr} | P_F={args.pfa} | seed={args.seed} "
          f"| device={device}", flush=True)

    results = {}
    npar = None
    for label, stem, pol, strata in points:
        E_tr, y_tr, roles_tr = load_split(data_dir, stem, pol, "train")
        E_te, y_te, roles_te = load_split(data_dir, stem, pol, "test")
        res, model, prep = run_point(
            E_tr, y_tr, roles_tr, E_te, y_te, roles_te,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            pfa=args.pfa, seed=args.seed, device=device,
            grad_accum=args.grad_accum, max_train=args.max_train,
            loss_name=args.loss)
        if npar is None:
            npar = sum(p.numel() for p in model.parameters())

        key = f"{label}-{pol}"
        results[key] = {
            "dataset": stem, "polarization": pol, "stratum": strata,
            "pulses": args.pulses, "seed": args.seed, "epochs": args.epochs,
            **res,
        }
        print(f"  [{label} {pol} {strata}] Pd={res['pd']:.4f} | "
              f"PF_all_nonPRC={res['pf_all_non_prc']:.5f} "
              f"PF_clutter={res['pf_clutter_only']:.5f} "
              f"SRC={res['src_target_rate']:.5f} h={res['h']:.5f} "
              f"T={res['temperature']:.1f} train_PFA={res['train_pfa']:.5f} "
              f"trainN={res['train_windows']} testN={res['test_windows']} "
              f"({res['train_seconds']:.0f}s)", flush=True)

        if args.save_checkpoint:
            save_checkpoint(os.path.join(args.save_checkpoint, f"{stem}__{pol}.pt"),
                            model, prep, meta={"label": label, "stem": stem,
                                               "pol": pol, **res})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "protocol": {
            "family": "position_generalization", "version": "v2.1", "alias": "PAX-L1",
            "range_cells": N_RANGE, "label_policy": "primary_strict",
            "range_roll": {"mode": "clutter_fill",
                           "max_shift": RANGE_ROLL_MAX_SHIFT,
                           "include_identity": RANGE_ROLL_INCLUDE_IDENTITY},
            "threshold_source": "train_clutter", "nominal_pfa": args.pfa,
            "seed": args.seed, "pulses": args.pulses,
        },
        "train": {"epochs": args.epochs, "batch_size": args.batch_size,
                  "grad_accum": args.grad_accum,
                  "effective_batch": args.batch_size * args.grad_accum,
                  "lr": args.lr, "optimizer": "Adam",
                  "loss": f"{args.loss}_unweighted"},
        "model": "SE (selected P=8 E4: radar_prior_dynamic_sfe + "
                 "pulse_attention_only_tfe TFE1 + stgnn_tfe TFE2 + GDCM-A0)",
        "temperature_scaling": "auto (per-condition mean|logit| over train "
                              "clutter, clip [2,16])",
        "params": npar,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.output}")

    if results:
        pds = [v["pd"] for v in results.values()]
        print(f"mean Pd = {np.mean(pds):.4f} +/- {np.std(pds):.4f} "
              f"({len(pds)} points)")


if __name__ == "__main__":
    main()
