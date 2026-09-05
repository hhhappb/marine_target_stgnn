#!/usr/bin/env python3
"""E4-D2-GDCM-A0 on IPIX Fig.7 (Protocol v2.0) — fair comparison with ST-GNN.

Same protocol as `stgnn ipix/run_fig7_ipix.py` (Protocol v2.0, IPIX Fig.7),
but the detector is the E4-D2-GDCM-A0 model (e4d2_gdcm, final adopted A0 =
D_a-only log(1+|Δ|) judgment head).

Protocol v2.0 (identical to the ST-GNN Fig.7 reproduction):
  * 14 IPIX datasets x 4 polarizations (HH, HV, VV, VH)  ->  56 conditions
  * Temporal split: FIRST 60% range profiles = training, LAST 40% = test
  * Window = 4 continuous range profiles, sliding step = 4 (non-overlap)
  * Target label: primary policy — only PRC cell = target, SRC = clutter
  * batch = 512, Adam lr = 0.001, unweighted CE (paper Eq.16)
  * FAR controller calibrated on ALL TRAINING-set clutter cells (Eq.15)
  * Operating point P_F = 0.001; metric = P_D over PRC cells (Fig.7)

Usage (run from /tmp/ST-GNN):
    python -m e4d2_gdcm.experiments.run_fig7_ipix_v2 \
        --epochs 100 --seeds 42,123,456 \
        --output e4d2_gdcm/checkpoints/fig7_e4d2_gdcm_a0.json
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e4d2_gdcm.models.e4d2 import E4D2, count_params
from e4d2_gdcm.utils.preprocess import E4D2Preprocessor

P = 4                  # window length (Table IV)
STEP = 4               # sliding step = window length (non-overlapping)
N_RANGE = 14           # IPIX range bins
POLS = ["hh", "hv", "vv", "vh"]

# Table I: data label (Fig. 7 x-axis) -> file stem
DATASETS = [
    (1, "19931107_135603_starea"),
    (2, "19931107_141630_starea"),
    (3, "19931107_145028_starea"),
    (4, "19931108_213827_starea"),
    (5, "19931108_220902_starea"),
    (6, "19931109_191449_starea"),
    (7, "19931109_202217_starea"),
    (8, "19931110_001635_starea"),
    (9, "19931111_163625_starea"),
    (10, "19931118_023604_stareC0000"),
    (11, "19931118_035737_stareC0000"),
    (12, "19931118_162155_stareC0000"),
    (13, "19931118_162658_stareC0000"),
    (14, "19931118_174259_stareC0000"),
]


def load_split(data_dir, stem, pol, split):
    """Load one condition split -> (E [n,P,N] complex64, y [n,N] uint8, roles [N] int)."""
    path = os.path.join(data_dir, f"{stem}__{pol}__{split}.npz")
    d = np.load(path)
    return d["E"], d["y_range"], d["range_roles"].astype(np.int64)


def prep_tensors(prep, E):
    """E4D2 inputs: complex [n,P,N] -> 5 torch tensors."""
    d = prep.transform(E)
    return {k: torch.tensor(v, dtype=torch.float32)
            for k, v in d.items()}


@torch.no_grad()
def calibrate_threshold(model, tensors, y, pfa, device, chunk=512, temperature=None):
    """Paper Eq.15: sort o(0) of ALL training clutter cells -> h.

    Temperature scaling (logits / T) de-saturates the softmax so the
    Eq.15 quantile threshold lands strictly below 1.0 (otherwise h=1.0
    declares every cell a target and PFA is uncontrolled).
    If temperature is None, it is auto-estimated per condition as the
    mean |logit| over training clutter cells.

    Returns (h, Nc, train_pfa, T).
    """
    model.eval()
    n = y.shape[0]
    if temperature is None:
        # auto temperature: mean |logit| over training clutter cells
        logit_abs = []
        for i in range(0, n, chunk):
            inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
            la = torch.abs(model(**inp)).cpu().numpy()        # [b,2,N]
            yb = y[i:i + chunk]
            logit_abs.append(la.transpose(0, 2, 1)[yb == 0].ravel())  # clutter cells
        la_all = np.concatenate(logit_abs)
        temperature = float(np.clip(la_all.mean(), 2.0, 16.0))

    o0_all = []
    for i in range(0, n, chunk):
        inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
        lo = model(**inp)
        o0 = torch.softmax(lo / temperature, 1)[:, 0, :].cpu().numpy()   # [b, N]
        yb = y[i:i + chunk]
        o0_all.append(o0[yb == 0])                                        # clutter cells only
    o0_all = np.concatenate(o0_all)
    o0_all.sort()
    Nc = o0_all.size
    idx = max(0, int(np.ceil(pfa * Nc)) - 1)
    h = float(o0_all[min(idx, Nc - 1)])
    train_pfa = float(np.mean(o0_all <= h))
    return h, Nc, train_pfa, temperature


@torch.no_grad()
def evaluate(model, tensors, y, roles, h, device, chunk=512, temperature=1.0):
    """Per-range-cell detection at threshold h (Fig.7 metric = P_D on PRC)."""
    model.eval()
    o0_all, y_all = [], []
    n = y.shape[0]
    for i in range(0, n, chunk):
        inp = {k: v[i:i + chunk].to(device) for k, v in tensors.items()}
        lo = model(**inp)
        o0 = torch.softmax(lo / temperature, 1)[:, 0, :].cpu().numpy()
        o0_all.append(o0)
        y_all.append(y[i:i + chunk])
    o0 = np.concatenate(o0_all)
    yb = np.concatenate(y_all)
    prc = roles == 2
    clu = roles == 0
    det = o0 <= h
    tp = float(((det) & (yb == 1)).sum())
    fn = float(((~det) & (yb == 1)).sum())
    pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pfa_all = float(det[:, ~prc].mean())
    pfa_clu = float(det[:, clu].mean()) if clu.any() else float("nan")
    return pd, pfa_all, pfa_clu


def train_condition(E_tr, y_tr, epochs, batch_size, lr, seed, device):
    """Train E4-D2-GDCM-A0 on one (dataset, polarization) condition."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = E4D2(detector='gdcm_a0').to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()                     # unweighted CE (Eq.16)

    prep = E4D2Preprocessor().fit(E_tr)              # fit per-condition stats
    tensors = prep_tensors(prep, E_tr)
    y_t = torch.tensor(y_tr, dtype=torch.long)

    n = y_t.shape[0]
    n_batches = max(1, n // batch_size)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)                       # CPU index for CPU tensors
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            inp = {k: v[idx].to(device) for k, v in tensors.items()}
            lb = y_t[idx].to(device)
            opt.zero_grad()
            lo = model(**inp)                        # [B, 2, N]
            loss = crit(lo.permute(0, 2, 1).reshape(-1, 2), lb.reshape(-1))
            loss.backward()
            opt.step()
    return model, prep


def main():
    ap = argparse.ArgumentParser(description="E4-D2-GDCM-A0 on IPIX Fig.7 (Protocol v2.0)")
    ap.add_argument("--data-dir", type=str,
                    default="/tmp/ST-GNN/stgnn ipix/ipix_dartmouth/processed/window4_stride4_primary")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--pfa", type=float, default=0.001)
    ap.add_argument("--temperature", type=float, default=None,
                    help="softmax temperature (None = auto per-condition mean|logit|)")
    ap.add_argument("--seeds", type=str, default="42,123,456")
    ap.add_argument("--conditions", type=str, default=None,
                    help="restrict run, e.g. '19931107_135603_starea:hh' (smoke)")
    ap.add_argument("--max-train", type=int, default=None, help="debug only")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output", type=str, default="checkpoints/fig7_e4d2_gdcm_a0.json")
    ap.add_argument("--plot", type=str, default="fig7_e4d2_gdcm_a0.png")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    seeds = [int(s) for s in args.seeds.split(",")]

    stem2label = {stem: label for label, stem in DATASETS}
    if args.conditions:
        conds = [tuple(c.split(":")) for c in args.conditions.split(",")]
        conds = [(stem2label[s], s, p) for s, p in conds]
    else:
        conds = [(label, stem, pol) for label, stem in DATASETS for pol in POLS]

    print(f"Fig.7 IPIX E4-D2-GDCM-A0 (Protocol v2.0) | device={device} | seeds={seeds} "
          f"| epochs={args.epochs} | batch={args.batch_size} | lr={args.lr} "
          f"| P_F={args.pfa} | conditions={len(conds)}")

    results = {}
    npar = None
    for label, stem, pol in conds:
        E_tr, y_tr, roles_tr = load_split(args.data_dir, stem, pol, "train")
        E_te, y_te, roles_te = load_split(args.data_dir, stem, pol, "test")
        roles_te = np.asarray(roles_te)
        if args.max_train and len(E_tr) > args.max_train:
            keep = np.random.RandomState(0).choice(len(E_tr), args.max_train, replace=False)
            E_tr, y_tr = E_tr[keep], y_tr[keep]

        per_seed = []
        for sd in seeds:
            t0 = time.time()
            model, prep = train_condition(E_tr, y_tr, args.epochs, args.batch_size,
                                          args.lr, sd, device)
            if npar is None:
                npar, _ = count_params(model)
            te_tensors = prep_tensors(prep, E_te)
            h, Nc, train_pfa, T = calibrate_threshold(model, prep_tensors(prep, E_tr), y_tr,
                                                      args.pfa, device, temperature=args.temperature)
            pd, pfa_all, pfa_clu = evaluate(model, te_tensors, y_te, roles_te, h, device,
                                            temperature=T)
            per_seed.append({"seed": sd, "pd": pd, "actual_pfa": pfa_all,
                             "pfa_clutter_only": pfa_clu, "train_pfa": train_pfa,
                             "h": h, "cal_Nc": int(Nc), "temperature": T})
            print(f"  [{stem} {pol}] seed={sd} Pd={pd:.4f} train_PFA={train_pfa:.5f} "
                  f"| test_PFA_all_nonPRC={pfa_all:.5f} test_PFA_clutter_only={pfa_clu:.5f} "
                  f"h={h:.5f} T={T:.1f} ({time.time()-t0:.0f}s)", flush=True)

        pds = [r["pd"] for r in per_seed]
        apfas = [r["actual_pfa"] for r in per_seed]
        results.setdefault(label, {})[pol] = {
            "dataset": stem,
            "pd_mean": float(np.mean(pds)),
            "pd_std": float(np.std(pds)),
            "actual_pfa_mean": float(np.mean(apfas)),
            "pfa_clutter_only_mean": float(np.mean([r["pfa_clutter_only"]
                                                    for r in per_seed])),
            "seeds": per_seed,
        }
        print(f"  -> Pd mean/std: {np.mean(pds):.4f} +/- {np.std(pds):.4f}", flush=True)

    # ── Summary ──
    print("\n" + "=" * 92)
    print(f"Fig. 7 E4-D2-GDCM-A0 (IPIX, Protocol v2.0, P_F={args.pfa}, params={npar:,})")
    for pol in POLS:
        vals = [results[l][pol]["pd_mean"] for l, _ in DATASETS if pol in results.get(l, {})]
        if vals:
            print(f"  {pol.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                  f"(mean over {len(vals)} datasets)")
    if all(pol in results.get(l, {}) for l, _ in DATASETS for pol in POLS):
        grand = [results[l][pol]["pd_mean"] for l, _ in DATASETS for pol in POLS]
        print(f"  ALL:  {np.mean(grand):.4f} +/- {np.std(grand):.4f}  (56 conditions)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "protocol": "v2.0", "model": "E4-D2-GDCM-A0",
        "datasets": 14, "polarizations": POLS, "conditions": len(conds),
        "window": P, "stride": STEP, "batch_size": args.batch_size,
        "optimizer": "Adam", "lr": args.lr, "loss": "cross-entropy",
        "pfa": args.pfa, "target_policy": "primary", "metric": "P_D",
        "epochs": args.epochs, "seeds": seeds, "params": npar,
        "temperature_scaling": "auto (per-condition mean|logit|, clip [2,16])"
        if args.temperature is None else args.temperature,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        for ax, pol in zip(axes.ravel(), POLS):
            labels, pds = [], []
            for label, stem in DATASETS:
                r = results.get(label, {}).get(pol)
                if r:
                    labels.append(label)
                    pds.append(r["pd_mean"])
            ax.bar(labels, pds, color="darkorange")
            ax.set_title(f"({'(abcd)'[POLS.index(pol)]}) {pol.upper()}")
            ax.set_xlabel("Data label")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
        for ax in axes.ravel():
            ax.set_ylabel("Detection probability")
        fig.suptitle(f"E4-D2-GDCM-A0 on IPIX (P$_F$ = {args.pfa}, Protocol v2.0)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.plot, dpi=150)
        print(f"Saved: {args.plot}")


if __name__ == "__main__":
    main()
