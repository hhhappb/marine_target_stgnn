#!/usr/bin/env python3
"""SE (selected P=8 E4) on IPIX under protocol v2.1 (PAX-L1).

v2.1 / PAX-L1 fixed settings (identical to the reference runners):
  * range_cells = 14
  * range_roll augmentation: mode=clutter_fill, max_shift=3, include_identity
  * label policy = primary_strict (only role == 2 / PRC cells are positive)
  * threshold source = train clutter (Eq.15 quantile), nominal P_F = 0.001
  * softmax temperature scaling (auto, mean|logit| over train clutter, clip [2,16])
  * seed = 42, primary objective = PD, PF reported only

Layers:
  * quick-8 model screening at P=8 (8 preregistered points, default)
  * full 56-point cohort (--all-56): 14 IPIX files x 4 polarizations

Usage (run from /tmp/ST-GNN/SE with this directory on PYTHONPATH)::

    PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.run_v21_ipix --conditions label06:hh
    PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.run_v21_ipix --all-56

The training / calibration / evaluation primitives are imported from
``experiments.train_ipix`` so both entry points share one implementation.
"""

import argparse
import json
import os

import numpy as np
import torch

try:  # package dir on sys.path
    from utils.data import (
        N_RANGE, PULSES, QUICK8_POINTS, FULL56_POINTS, resolve_condition,
        load_split, DATASETS,
    )
    from experiments.train_ipix import (
        DEFAULT_DATA_ROOT, RANGE_ROLL_MAX_SHIFT, RANGE_ROLL_INCLUDE_IDENTITY,
        _default_data_dir, resolve_device, run_point, save_checkpoint,
    )
except ImportError:  # `SE` package
    from SE.utils.data import (
        N_RANGE, PULSES, QUICK8_POINTS, FULL56_POINTS, resolve_condition,
        load_split, DATASETS,
    )
    from SE.experiments.train_ipix import (
        DEFAULT_DATA_ROOT, RANGE_ROLL_MAX_SHIFT, RANGE_ROLL_INCLUDE_IDENTITY,
        _default_data_dir, resolve_device, run_point, save_checkpoint,
    )

PFA = 0.001
SEED = 42
EPOCHS = 60
BATCH = 64
GRAD_ACCUM = 8
LR = 0.001


def select_points(args):
    """Return (points, layer) where point = (label, stem, pol, stratum)."""
    label_to_stem = {f"label{n:02d}": stem for n, stem in DATASETS.items()}
    if args.all_56:
        points = [(lb, label_to_stem[lb], pol, st) for lb, pol, st in FULL56_POINTS]
        layer = "full56_screening"
    else:
        points = [(lb, label_to_stem[lb], pol, st) for lb, pol, st in QUICK8_POINTS]
        layer = "quick8_screening"

    if args.conditions:
        wanted = {}
        for token in args.conditions.split(","):
            label, stem, pol = resolve_condition(token.strip())
            wanted[(label, pol)] = (label, stem, pol)
        points = [p for p in points if (p[0], p[2]) in wanted]
        # allow conditions outside the quick-8 set to be added explicitly
        present = {(p[0], p[2]) for p in points}
        strata_map = dict(((lb, p), st) for lb, p, st in QUICK8_POINTS)
        for (label, pol), (_, stem, _p) in wanted.items():
            if (label, pol) not in present:
                points.append((label, stem, pol, strata_map.get((label, pol), "full")))
        layer = f"{layer}_restricted"
    return points, layer


def main():
    ap = argparse.ArgumentParser(
        description="SE (selected P=8 E4) on IPIX under v2.1 (PAX-L1)")
    ap.add_argument("--pulses", type=int, default=PULSES, choices=[PULSES],
                    help="SE is compiled for P=8")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--pfa", type=float, default=PFA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--conditions", type=str, default=None,
                    help="comma list, 'labelNN:pol' or 'stem:pol'")
    ap.add_argument("--all-56", dest="all_56", action="store_true",
                    help="full cohort: 14 IPIX files x 4 polarizations (56 points)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--save-checkpoint", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    data_dir = _default_data_dir(args.data_root, args.pulses)
    points, layer = select_points(args)
    if args.output is None:
        args.output = f"checkpoints/v21_se_selected_e4_p{args.pulses}.json"

    print(f"v2.1 PAX-L1 SE (selected P=8 E4) | P={args.pulses} | layer={layer} "
          f"| points={len(points)} | seed={args.seed} | epochs={args.epochs} "
          f"| batch={args.batch_size}x{args.grad_accum} | lr={args.lr} "
          f"| P_F={args.pfa} | range_roll=clutter_fill(max_shift="
          f"{RANGE_ROLL_MAX_SHIFT}) | label=primary_strict | device={device}",
          flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = {}
    npar = None
    if os.path.exists(args.output):          # resume
        try:
            data = json.load(open(args.output))
            prev = data.get("results", {})
            results = {k: v for k, v in prev.items() if "pd" in v}
            npar = data.get("params")
            print(f"resume: {len(results)} points already done", flush=True)
        except Exception:
            pass

    for label, stem, pol, strata in points:
        key = f"{label}-{pol}"
        if key in results:
            continue
        E_tr, y_tr, roles_tr = load_split(data_dir, stem, pol, "train")
        E_te, y_te, roles_te = load_split(data_dir, stem, pol, "test")
        res, model, prep = run_point(
            E_tr, y_tr, roles_tr, E_te, y_te, roles_te,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            pfa=args.pfa, seed=args.seed, device=device,
            grad_accum=args.grad_accum, max_train=args.max_train)
        if npar is None:
            npar = sum(p.numel() for p in model.parameters())

        results[key] = {
            "dataset": stem, "polarization": pol, "stratum": strata,
            "pulses": args.pulses, "seed": args.seed, "epochs": args.epochs,
            **res,
        }
        print(f"  [{label} {pol} {strata}] seed={args.seed} Pd={res['pd']:.4f} "
              f"| PF_all_nonPRC={res['pf_all_non_prc']:.5f} "
              f"PF_clutter={res['pf_clutter_only']:.5f} "
              f"SRC={res['src_target_rate']:.5f} h={res['h']:.5f} "
              f"T={res['temperature']:.1f} train_PFA={res['train_pfa']:.5f} "
              f"trainN={res['train_windows']} testN={res['test_windows']} "
              f"({res['train_seconds']:.0f}s)", flush=True)

        if args.save_checkpoint:
            save_checkpoint(os.path.join(args.save_checkpoint,
                                         f"{stem}__{pol}.pt"),
                            model, prep,
                            meta={"label": label, "stem": stem, "pol": pol, **res})

        with open(args.output, "w") as f:       # incremental checkpoint
            json.dump({"partial": True, "n_done": len(results),
                       "params": npar, "results": results},
                      f, indent=2)

    print("\n" + "=" * 80)
    params_str = f"{npar:,}" if npar is not None else "unknown"
    print(f"v2.1 PAX-L1 SE (selected P=8 E4) P={args.pulses} ({layer}) summary "
          f"| params={params_str}")
    for strata in ("hard", "easy", "full"):
        vals = [(k, v) for k, v in results.items() if v["stratum"] == strata]
        if vals:
            print(f"\n  [{strata}]")
            for k, v in vals:
                print(f"    {k:>10s}  Pd={v['pd']:.4f}  "
                      f"PF_all={v['pf_all_non_prc']:.5f}")
            print(f"    -> mean Pd = {np.mean([v['pd'] for _, v in vals]):.4f}")
    if results:
        all_pd = [v["pd"] for v in results.values()]
        print(f"\n  OVERALL mean Pd = {np.mean(all_pd):.4f} +/- "
              f"{np.std(all_pd):.4f} ({len(all_pd)} points)")

    payload = {
        "protocol": {"family": "position_generalization", "version": "v2.1",
                     "alias": "PAX-L1", "range_cells": N_RANGE,
                     "range_roll": {"mode": "clutter_fill",
                                    "max_shift": RANGE_ROLL_MAX_SHIFT,
                                    "include_identity": RANGE_ROLL_INCLUDE_IDENTITY},
                     "label_policy": "primary_strict",
                     "threshold_source": "train_clutter",
                     "nominal_pfa": args.pfa, "seed": args.seed,
                     "primary_objective": "PD",
                     "pf_role": "reported_only_not_acceptance_gate",
                     "pulses": args.pulses, "layer": layer},
        "train": {"epochs": args.epochs, "batch_size": args.batch_size,
                  "grad_accum": args.grad_accum,
                  "effective_batch": args.batch_size * args.grad_accum,
                  "lr": args.lr, "optimizer": "Adam",
                  "loss": "cross_entropy_unweighted"},
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


if __name__ == "__main__":
    main()
