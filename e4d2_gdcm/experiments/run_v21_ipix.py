#!/usr/bin/env python3
"""E4-D2-GDCM-A0 on IPIX under protocol v2.1 (PAX-L1, position-generalization).

Protocol v2.1 = IPIX PAX-L1 (docs: ipix_pax_l1_experiment_20260831_v3/docs/
ipix_pax_l1_protocol.md), applied to the E4-D2-GDCM-A0 detector.

Fixed protocol settings (v2.1 / PAX-L1):
  * range_cells = 14
  * range_roll augmentation: mode=clutter_fill, max_shift=3, include_identity=true
  * train/evaluation label policy = primary_strict (only role==2 cell is target)
  * threshold source = train clutter (Eq. 15 quantile), nominal P_F = 0.001
  * seed = 42 (single seed)
  * primary objective = PD; PF reported only, not an acceptance gate

Layer executed here (per protocol 2.1 + 2.2):
  * quick-8 model screening at P=8 (8 preregistered points)
  * P=16 mechanism probe on the 4-point subset (label06-hh/vv, label02-hh, label07-vv)

Training per point (matches PAX-L1 quick-8 configs):
  * 60 epochs, batch 64 + grad-accum 8 (effective 512), Adam lr 0.001
  * unweighted cross-entropy, no scheduler, no weight decay
  * range_roll augmentation re-sampled per epoch (on the complex echo, then
    the E4-D2 input features are re-transformed so PL/RPH/cell/amp features
    stay aligned with the shifted target cell)
  * softmax temperature scaling (auto per-condition mean|logit| over training
    clutter, clip [2,16]) — the E4-D2 IPIX calibration inherited from v2.0;
    without it the Eq.15 quantile threshold saturates at h=1.0.

Usage (run from /tmp/ST-GNN):
    python -m e4d2_gdcm.experiments.run_v21_ipix --pulses 8
    python -m e4d2_gdcm.experiments.run_v21_ipix --pulses 16
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/tmp/ST-GNN/stgnn ipix")   # data dir relative base

from e4d2_gdcm.models.e4d2 import E4D2, count_params
from e4d2_gdcm.utils.preprocess import E4D2Preprocessor

N_RANGE = 14
PFA = 0.001
SEED = 42
EPOCHS = 60
BATCH = 64
GRAD_ACCUM = 8
LR = 0.001
RANGE_ROLL_MAX_SHIFT = 3
RANGE_ROLL_INCLUDE_IDENTITY = True

DATASETS = {
    2: "19931107_141630_starea",
    6: "19931109_191449_starea",
    7: "19931109_202217_starea",
    12: "19931118_162155_stareC0000",
}
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
PULSE_PROBE_POINTS = (
    ("label06", "hh", "hard"),
    ("label06", "vv", "hard"),
    ("label02", "hh", "easy"),
    ("label07", "vv", "easy"),
)


# ────────────────────────────────────────────────────────────
#  Data loading + primary_strict labels
# ────────────────────────────────────────────────────────────

def load_split(data_dir, stem, pol, split):
    """Load one condition split -> (E [n,P,N] complex64, y [n,N] int64, roles [N])."""
    path = os.path.join(data_dir, f"{stem}__{pol}__{split}.npz")
    d = np.load(path)
    E = d["E"]
    roles = np.asarray(d["range_roles"]).astype(np.int64)
    y_strict = np.zeros(N_RANGE, dtype=np.int64)
    y_strict[roles == 2] = 1
    y = np.broadcast_to(y_strict, (E.shape[0], N_RANGE)).copy()
    return E, y, roles


# ────────────────────────────────────────────────────────────
#  range_roll augmentation (v2.1, clutter_fill) on complex echo
#  Semantics identical to ipix_window.py; data movement vectorised.
# ────────────────────────────────────────────────────────────

def _sample_shift(roles, max_shift, include_identity, rng):
    related = np.flatnonzero(roles > 0)
    minimum = max(-max_shift, -int(related.min()))
    maximum = min(max_shift, N_RANGE - 1 - int(related.max()))
    candidates = np.arange(minimum, maximum + 1, dtype=np.int64)
    if not include_identity:
        candidates = candidates[candidates != 0]
    return int(rng.choice(candidates))


def augment_train(E, y, roles, rng, max_shift=RANGE_ROLL_MAX_SHIFT):
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
        s = _sample_shift(roles, max_shift, RANGE_ROLL_INCLUDE_IDENTITY, rng)
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
            vacated_positions = np.arange(s)
        else:
            E_aug[idx, :, :s] = E[idx, :, -s:]
            y_aug[idx, :s] = y[idx, -s:]
            vacated_positions = np.arange(N + s, N)
        for j, pos in enumerate(vacated_positions):
            donor_col = np.array([donors[i][j] for i in idx], dtype=np.int64)
            E_aug[idx, :, pos] = E[idx, :, donor_col]
        y_aug[np.ix_(idx, vacated_positions)] = 0
    return E_aug, y_aug


def prep_tensors(prep, E):
    """E4-D2 inputs: complex [n,P,N] -> 5 torch tensors."""
    d = prep.transform(E)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}


# ────────────────────────────────────────────────────────────
#  FAR controller (Eq. 15) with E4-D2 temperature calibration
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_threshold(model, tensors, y, pfa, device, chunk=512):
    """Threshold h from TRAINING clutter cells (Eq. 15), auto temperature."""
    model.eval()
    n = y.shape[0]
    logit_abs = []
    for i in range(0, n, chunk):
        inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
        la = torch.abs(model(**inp)).cpu().numpy()
        yb = y[i:i + chunk]
        logit_abs.append(la.transpose(0, 2, 1)[yb == 0].ravel())
    la_all = np.concatenate(logit_abs)
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
    return h, Nc, train_pfa, T


@torch.no_grad()
def evaluate(model, tensors, y, roles, h, device, T, chunk=512):
    model.eval()
    o0_all, y_all = [], []
    n = y.shape[0]
    for i in range(0, n, chunk):
        inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
        lo = model(**inp)
        o0 = torch.softmax(lo / T, 1)[:, 0, :].cpu().numpy()
        o0_all.append(o0)
        y_all.append(y[i:i + chunk])
    o0 = np.concatenate(o0_all)
    yb = np.concatenate(y_all)
    prc = roles == 2
    src = roles == 1
    clu = roles == 0
    det = o0 <= h
    tp = float(((det) & (yb == 1)).sum())
    fn = float(((~det) & (yb == 1)).sum())
    pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pfa_all = float(det[:, ~prc].mean())
    pfa_clu = float(det[:, clu].mean()) if clu.any() else float("nan")
    src_r = float(det[:, src].mean()) if src.any() else float("nan")
    return {"pd": pd, "pf_all_non_prc": pfa_all, "pf_clutter_only": pfa_clu,
            "src_target_rate": src_r}


# ────────────────────────────────────────────────────────────
#  Training (v2.1: 60ep, batch64 + accum8, range_roll online)
# ────────────────────────────────────────────────────────────

def train_condition(E_tr, y_tr, roles, epochs, batch_size, lr, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = E4D2(detector="gdcm_a0", P=E_tr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    prep = E4D2Preprocessor().fit(E_tr)   # fit stats on original training set

    n = y_tr.shape[0]
    for ep in range(epochs):
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
            if (b // batch_size + 1) % GRAD_ACCUM == 0 or b + batch_size >= n:
                opt.step()
                opt.zero_grad()
    return model, prep


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="E4-D2-GDCM-A0 on IPIX under v2.1 (PAX-L1)")
    ap.add_argument("--pulses", type=int, default=8, choices=[8, 16])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--pfa", type=float, default=PFA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--conditions", type=str, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--data-root", type=str,
                    default="/tmp/ST-GNN/stgnn ipix/ipix_dartmouth/processed")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    P = args.pulses
    data_dir = os.path.join(args.data_root, f"window{P}_stride4_related")
    points = QUICK8_POINTS if P == 8 else PULSE_PROBE_POINTS
    layer = "quick8_screening" if P == 8 else "pulse_count_probe"

    if args.conditions:
        cond_set = set(args.conditions.split(","))
        points = [(lb, pol, st) for lb, pol, st in points
                  if f"{DATASETS[int(lb[-2:])]}:{pol}" in cond_set]
    if args.output is None:
        args.output = f"checkpoints/v21_e4d2_gdcm_a0_p{P}.json"

    print(f"v2.1 PAX-L1 E4-D2-GDCM-A0 | P={P} | layer={layer} | points={len(points)} "
          f"| seed={args.seed} | epochs={args.epochs} | batch={args.batch_size}"
          f"x{GRAD_ACCUM} | lr={args.lr} | P_F={args.pfa} | range_roll=clutter_fill"
          f"(max_shift={RANGE_ROLL_MAX_SHIFT}) | label=primary_strict", flush=True)

    results = {}
    npar = None
    for label, pol, strata in points:
        label_num = int(label[-2:])
        stem = DATASETS[label_num]
        E_tr, y_tr, roles_tr = load_split(data_dir, stem, pol, "train")
        E_te, y_te, roles_te = load_split(data_dir, stem, pol, "test")
        if args.max_train and len(E_tr) > args.max_train:
            keep = np.random.RandomState(0).choice(len(E_tr), args.max_train, replace=False)
            E_tr, y_tr = E_tr[keep], y_tr[keep]

        t0 = time.time()
        model, prep = train_condition(E_tr, y_tr, roles_tr, args.epochs,
                                      args.batch_size, args.lr, args.seed, device)
        if npar is None:
            npar, _ = count_params(model)
        te_tensors = prep_tensors(prep, E_te)
        h, Nc, train_pfa, T = calibrate_threshold(model, prep_tensors(prep, E_tr),
                                                  y_tr, args.pfa, device)
        res = evaluate(model, te_tensors, y_te, roles_te, h, device, T)

        results[label + "-" + pol] = {
            "dataset": stem, "polarization": pol, "stratum": strata,
            "pulses": P, "seed": args.seed, "epochs": args.epochs,
            "train_windows": int(len(E_tr)), "test_windows": int(len(E_te)),
            **res, "h": h, "cal_Nc": int(Nc), "train_pfa": train_pfa,
            "temperature": T, "train_seconds": round(time.time() - t0, 1),
        }
        print(f"  [{label} {pol} {strata}] seed={args.seed} "
              f"Pd={res['pd']:.4f} | PF_all_nonPRC={res['pf_all_non_prc']:.5f} "
              f"PF_clutter={res['pf_clutter_only']:.5f} SRC={res['src_target_rate']:.5f} "
              f"h={h:.5f} T={T:.1f} train_PFA={train_pfa:.5f} "
              f"trainN={len(E_tr)} testN={len(E_te)} ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 80)
    print(f"v2.1 PAX-L1 E4-D2-GDCM-A0 P={P} ({layer}) summary | params={npar:,}")
    for strata in ("hard", "easy"):
        vals = [(k, v) for k, v in results.items() if v["stratum"] == strata]
        if vals:
            print(f"\n  [{strata}]")
            for k, v in vals:
                print(f"    {k:>10s}  Pd={v['pd']:.4f}  PF_all={v['pf_all_non_prc']:.5f}")
            print(f"    -> mean Pd = {np.mean([v['pd'] for _, v in vals]):.4f}")
    all_pd = [v["pd"] for v in results.values()]
    print(f"\n  OVERALL mean Pd = {np.mean(all_pd):.4f} ± {np.std(all_pd):.4f} "
          f"({len(all_pd)} points)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "protocol": {"family": "position_generalization", "version": "v2.1",
                     "alias": "PAX-L1", "config_version": "ipix_pax_l1_v1",
                     "range_cells": N_RANGE,
                     "range_roll": {"mode": "clutter_fill", "max_shift": RANGE_ROLL_MAX_SHIFT,
                                    "include_identity": RANGE_ROLL_INCLUDE_IDENTITY},
                     "label_policy": "primary_strict", "threshold_source": "train_clutter",
                     "nominal_pfa": args.pfa, "seed": args.seed,
                     "primary_objective": "PD", "pf_role": "reported_only_not_acceptance_gate",
                     "pulses": P, "layer": layer},
        "train": {"epochs": args.epochs, "batch_size": args.batch_size,
                  "grad_accum": GRAD_ACCUM, "effective_batch": args.batch_size * GRAD_ACCUM,
                  "lr": args.lr, "optimizer": "Adam", "loss": "cross_entropy_unweighted"},
        "model": "E4-D2-GDCM-A0",
        "temperature_scaling": "auto (per-condition mean|logit| over train clutter, clip [2,16])",
        "params": npar,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
