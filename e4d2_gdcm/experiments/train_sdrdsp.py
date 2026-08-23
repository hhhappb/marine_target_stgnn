"""GDCM-E4 training script (SDRDSP).

E4-D2 with the GDCM detector. Final adopted setting: A0 = D_a only
(log(1+|Δ|)) — simplest and most stable judgement representation.
Variants: gdcm_a0 (final) | gdcm_g0..g4 | rdj.

Protocol v1.1 (mandatory): target initial phase φ0 ~ U(-π,π) per sample.

Usage:
    python -m e4d2_gdcm.experiments.train_sdrdsp \
        --train data/20210106155330_01_staring.mat \
        --test  data/20210106155432_01_staring.mat \
        --detector gdcm_a0 --epochs 200 --seeds 42,123,456 --batch 24 \
        --output checkpoints/gdcm_best.pt

Dependencies: torch, numpy, scipy, scikit-learn
"""

import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e4d2_gdcm.models.e4d2 import E4D2, count_params
from e4d2_gdcm.utils.preprocess import (
    E4D2Preprocessor, auto_load_mat,
)

P = 4
N_RANGE = 2224          # FULL range profile width — NO range cropping (protocol v1.1)
STEP = P
PRT = 1.0 / 1600
WAVELENGTH = 0.03
TARGET_RANGE = 2083     # original target range cell in the FULL range dim
POSITIONS_PER_SAMPLE = 5
PFA_LEVELS = [0.0001, 0.001, 0.01]
PHI0_RNG_SEED = 777


def gen_positions(seed, N=N_RANGE):
    rng = np.random.RandomState(seed + 1000)
    chosen, available = [], list(range(N))
    for _ in range(POSITIONS_PER_SAMPLE):
        if not available:
            break
        pv = rng.choice(available)
        chosen.append(pv)
        for r in range(max(0, pv - 10), min(N, pv + 11)):
            if r in available:
                available.remove(r)
    return chosen


def gen_speed(seed):
    return np.random.RandomState(seed + 2000).uniform(0.1, 0.5)


def inject_target(segment, positions, speed, scr_db, phi0_rng):
    """Inject target with per-sample random initial phase φ0 ~ U(-π,π)."""
    seg = segment.copy()
    N = seg.shape[1]
    available = [i for i in range(N) if i not in set(positions)]
    rng = np.random.RandomState(sum(positions) % (2 ** 31))
    ref_indices = rng.choice(available, min(20, len(available)), replace=False)
    clutter_power = np.sum(np.abs(seg[:, ref_indices]) ** 2) / P
    amp = np.sqrt(clutter_power * (10 ** (scr_db / 10.0)))
    phi0 = phi0_rng.uniform(-np.pi, np.pi)
    for p in range(P):
        phase = 4.0 * np.pi * speed * PRT * p / WAVELENGTH + phi0
        for pos in positions:
            seg[p, pos] += amp * np.exp(1j * phase)
    labels = np.zeros(N, dtype=np.int32)
    for pos in positions:
        labels[pos] = 1
    return seg, labels


def build_dataset(data, scr_list):
    Xs, ys, srs = [], [], []
    phi0_rng = np.random.RandomState(PHI0_RNG_SEED)
    total = data.shape[0]
    for scr in scr_list:
        for st in range(0, total - P, STEP):
            seg = data[st:st + P, :].copy()
            seg, labels = inject_target(seg, gen_positions(st), gen_speed(st), scr, phi0_rng)
            Xs.append(seg); ys.append(labels); srs.append(scr)
    return (np.array(Xs, dtype=np.complex64),
            np.array(ys, dtype=np.int32),
            np.array(srs, dtype=np.int32))


def build_test_data(data, scr_list):
    result = {}
    total = data.shape[0]
    for scr in scr_list:
        Xs, ys = [], []
        for st in range(0, total - P, STEP):
            phi0_rng = np.random.RandomState((scr + 100) * 1000 + st)
            seg = data[st:st + P, :].copy()
            seg, labels = inject_target(seg, [TARGET_RANGE], 0.4, scr, phi0_rng)
            Xs.append(seg); ys.append(labels)
        result[scr] = {'X': np.array(Xs, dtype=np.complex64),
                       'y': np.array(ys, dtype=np.int32)}
    return result


def build_clutter_only(data, n_samples=800):
    """Legacy: pure-clutter calibration set (kept for reference, NOT used in v1.1).
    v1.1 FAR (paper Eq.15) sorts o(0) of ALL training-set clutter cells instead.
    """
    total = data.shape[0]
    segs = []
    for st in range(0, min(total - P, n_samples * STEP), STEP):
        segs.append(data[st:st + P, :])
        if len(segs) >= n_samples:
            break
    return np.array(segs, dtype=np.complex64)


def process_stat(cplx_batch, preprocessor):
    d = preprocessor.transform(cplx_batch)
    return (torch.tensor(d['x_main'], dtype=torch.float32),
            torch.tensor(d['x_phase_cell'], dtype=torch.float32),
            torch.tensor(d['x_amp_map'], dtype=torch.float32),
            torch.tensor(d['x_amp_cell'], dtype=torch.float32),
            torch.tensor(d['x_global'], dtype=torch.float32))


def train_one_seed(seed, preprocessor, X_tr, y_tr, s_tr, device, epochs, batch_size, detector):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = E4D2(detector=detector).to(device)
    n_params, _ = count_params(model)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()          # paper Eq.(16): UNWEIGHTED CE
    unique_scr = np.unique(s_tr)
    scr_indices = {s: np.where(s_tr == s)[0] for s in unique_scr}
    n_per_scr = batch_size // len(unique_scr) + 1

    for ep in range(epochs):
        model.train()
        batch_idx = []
        for s in unique_scr:
            si = scr_indices[s]
            batch_idx.extend(np.random.choice(si, min(n_per_scr, len(si)), replace=False).tolist())
        batch_idx = np.array(batch_idx[:batch_size])
        np.random.shuffle(batch_idx)
        t = process_stat(X_tr[batch_idx], preprocessor)
        t = [x.to(device) for x in t]
        lb = torch.tensor(y_tr[batch_idx], dtype=torch.long, device=device)
        opt.zero_grad()
        lo = model(*t)
        loss = crit(lo.permute(0, 2, 1).reshape(-1, 2), lb.reshape(-1))
        loss.backward()
        opt.step()
    return model, n_params


def calibrate_from_train(model, preprocessor, X_tr, y_tr, batch_size, device,
                         pfa_levels=PFA_LEVELS):
    """Paper Eq.(15): sort o(0) of ALL training-set clutter cells.

    N_c = total training clutter cells (incl. clutter cells of injected-target
    samples); h = o_sorted[ceil(α_f·N_c)−1]. No separate target-free set.
    """
    model.eval()
    cc_list = []
    with torch.no_grad():
        for b in range(0, len(X_tr), batch_size):
            t = process_stat(X_tr[b:b + batch_size], preprocessor)
            lo = model(*[x.to(device) for x in t])
            cs = torch.softmax(lo, 1)[:, 0].cpu().numpy()       # [b, N]
            lb = y_tr[b:b + batch_size]
            cc_list.append(cs[lb == 0])                          # clutter cells only
    cc_all = np.concatenate(cc_list)
    sc_ = np.sort(cc_all)
    Nc = len(sc_)
    return {p: float(sc_[max(0, min(int(np.ceil(p * Nc)) - 1, Nc - 1))]) for p in pfa_levels}, Nc


def evaluate(model, preprocessor, test_data, scr_list, thresholds, batch_size, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    psc = {}
    with torch.no_grad():
        for scr in scr_list:
            d = test_data[scr]
            Xs, ys = d['X'], d['y'].flatten()
            csl = []
            for b0 in range(0, len(Xs), batch_size):             # batch for N=2224 GAT
                t = process_stat(Xs[b0:b0 + batch_size], preprocessor)
                lo = model(*[x.to(device) for x in t])
                csl.append(torch.softmax(lo, 1)[:, 0].cpu().numpy().flatten())
            psc[scr] = (np.concatenate(csl), ys)

    all_cs = np.concatenate([v[0] for v in psc.values()])
    all_lb = np.concatenate([v[1] for v in psc.values()])
    auc = float(roc_auc_score(all_lb, 1.0 - all_cs))

    results = {}
    for pfa in PFA_LEVELS:
        h = thresholds[pfa]
        pd_per_scr = {}
        for scr in scr_list:
            cs, lbs = psc[scr]
            det = (cs <= h).astype(float)
            TP = ((det == 1) & (lbs == 1)).sum()
            FN = ((det == 0) & (lbs == 1)).sum()
            pd_per_scr[scr] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        low_scrs = list(range(-24, -13, 2))
        pdL = float(np.mean([pd_per_scr[s] for s in low_scrs]))
        results[pfa] = {'pdL': pdL, 'pd_per_scr': pd_per_scr}
    return auc, results


def main():
    parser = argparse.ArgumentParser(description='Train GDCM-E4 model')
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--detector', type=str, default='gdcm_a0',
                        help='rdj | gdcm_a0 | gdcm_g0 | gdcm_g1 | gdcm_g2 | gdcm_g3 | gdcm_g4')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--output', type=str, default='checkpoints/gdcm_best.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"GDCM-E4 | detector={args.detector} | device={device} | seeds={seeds} | epochs={args.epochs}")
    print(f"Range dimension: FULL (no cropping) | target at range cell {TARGET_RANGE}")

    train_raw = auto_load_mat(args.train)          # FULL range dimension
    test_raw = auto_load_mat(args.test)
    N_full = train_raw.shape[1]
    print(f"Train: {train_raw.shape}  Test: {test_raw.shape}  (N_full={N_full})")

    train_scr = list(range(-12, 15, 2))
    test_scr = list(range(-24, 15, 2))
    X_tr, y_tr, s_tr = build_dataset(train_raw, train_scr)
    test_data = build_test_data(test_raw, test_scr)
    print(f"Train {X_tr.shape[0]} samples | Test {sum(len(v['X']) for v in test_data.values())} samples")

    preprocessor = E4D2Preprocessor().fit(X_tr)
    print(f"P99={preprocessor.P99:.1f}")

    all_results = []
    all_models = {}
    for seed in seeds:
        t0 = time.time()
        model, n_params = train_one_seed(seed, preprocessor, X_tr, y_tr, s_tr,
                                         device, args.epochs, args.batch, args.detector)
        thresholds, Nc = calibrate_from_train(model, preprocessor, X_tr, y_tr, args.batch, device)
        auc, multi_pfa = evaluate(model, preprocessor, test_data, test_scr, thresholds,
                                  args.batch, device)
        all_models[seed] = model
        all_results.append({'seed': seed, 'auc': auc,
                            'pdL': multi_pfa[0.001]['pdL'],
                            **{f'pdL_pfa{int(10000*p)}': v['pdL'] for p, v in multi_pfa.items()},
                            'pd_scr': {str(s): v['pd_per_scr'] for s, v in multi_pfa.items()}})
        print(f"  Seed {seed}: AUC={auc:.4f}  PdL@1e-3={multi_pfa[0.001]['pdL']:.4f}  (N_c={Nc:,})  ({time.time()-t0:.0f}s)")

    aucs = [r['auc'] for r in all_results]
    print(f"\nAUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  |  Params: {n_params:,}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({'model_state_dict': all_models[seeds[int(np.argmax(aucs))]].state_dict(),
                'preprocessor_P99': preprocessor.P99,
                'detector': args.detector,
                'N': int(N_full),
                'auc_mean': float(np.mean(aucs)),
                'all_results': all_results}, args.output)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
