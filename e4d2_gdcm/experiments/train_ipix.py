"""Train ST-GNN and E4-D2 on IPIX Dartmouth dataset with identical protocol.

Usage:
    python -m e4d2.experiments.train_ipix --model stgnn --epochs 100 --max-train 20000
    python -m e4d2.experiments.train_ipix --model both   --seeds 42,123,456
"""
import argparse, sys, os, time, json, glob

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from e4d2_gdcm.models.stgnn_backbone import STFE, TFE, STGNNBackbone
from e4d2_gdcm.models.e4d2 import E4D2, D0Detector
from e4d2_gdcm.utils.preprocess import E4D2Preprocessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════
#  ST-GNN (2ch) model for IPIX
# ═══════════════════════════════════════════════

class STGNN2ch(nn.Module):
    """ST-GNN baseline: 2-channel I/Q input."""
    def __init__(self):
        super().__init__()
        self.ft1 = nn.Conv2d(2, 32, (1, 3), padding=(0, 1))
        self.ft2 = nn.Conv2d(32, 64, (1, 3), padding=(0, 1))
        self.ftr = nn.ReLU()
        self.s1 = STFE(64, 128)
        self.t1 = TFE(128, 256)
        self.s2 = STFE(256, 512)
        self.t2 = TFE(512, 1024)
        self.det = nn.Sequential(nn.Conv1d(1024, 512, 3, padding=1),
                                 nn.ReLU(), nn.Conv1d(512, 2, 1))

    def forward(self, x):
        f = self.ftr(self.ft2(self.ftr(self.ft1(x))))
        xl = [f[:, :, p, :] for p in range(f.size(2))]
        s1v = self.s1(xl)
        t1v = self.t1(s1v)
        t1l = [t1v[:, :, p, :] for p in range(t1v.size(2))]
        s2v = self.s2(t1l)
        t2v = self.t2(s2v)
        return self.det(t2v.squeeze(2))


# ═══════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════

def load_ipix_splits(processed_dir):
    train_files = sorted(glob.glob(f"{processed_dir}/*__train.npz"))
    test_files = sorted(glob.glob(f"{processed_dir}/*__test.npz"))

    Xt, yt = [], []
    for f in train_files:
        d = np.load(f); Xt.append(d['E']); yt.append(d['y_range'])
    X_train = np.concatenate(Xt, axis=0)
    y_train = np.concatenate(yt, axis=0).astype(np.int64)

    Xs, ys = [], []
    for f in test_files:
        d = np.load(f); Xs.append(d['E']); ys.append(d['y_range'])
    X_test = np.concatenate(Xs, axis=0)
    y_test = np.concatenate(ys, axis=0).astype(np.int64)

    print(f"Train: {X_train.shape[0]} win, Test: {X_test.shape[0]} win, "
          f"P={X_train.shape[1]}, N={X_train.shape[2]}")
    return X_train, y_train, X_test, y_test


def subsample(X, y, max_samples):
    if len(X) <= max_samples:
        return X, y
    idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
    return X[idx], y[idx].copy()


def calibrate_far(clutter_scores, pfa):
    if len(clutter_scores) == 0:
        return 0.5
    s = np.sort(clutter_scores)
    idx = max(0, int(np.ceil(pfa * len(s))) - 1)
    return float(s[min(idx, len(s) - 1)])


# ═══════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════

def run_experiment(X_train_full, y_train_full, X_test, y_test,
                   epochs, batch_size, seeds, pfas, max_train, model_type):
    P, N = X_train_full.shape[1], X_train_full.shape[2]
    X_train, y_train = subsample(X_train_full, y_train_full, max_train)
    n_train = len(X_train)
    n_batches = n_train // batch_size

    # Prepare tensors
    X_stgnn = torch.tensor(
        np.stack([X_train.real, X_train.imag], axis=1), dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    # E4D2 preprocessor
    n_fit = min(5000, n_train)
    prep = E4D2Preprocessor()
    prep.fit(X_train[:n_fit])
    e4d2_inp = prep.transform(X_train)
    e4d2_tensors = {
        'x_main': torch.tensor(e4d2_inp['x_main']),
        'x_phase_cell': torch.tensor(e4d2_inp['x_phase_cell']),
        'x_amp_map': torch.tensor(e4d2_inp['x_amp_map']),
        'x_amp_cell': torch.tensor(e4d2_inp['x_amp_cell']),
        'x_global': torch.tensor(e4d2_inp['x_global']),
    }

    # Calibration indices
    n_cal = min(500, n_train)
    cal_idx = np.where(y_train[:n_cal].sum(axis=1) == 0)[0][:n_cal]
    if len(cal_idx) < 10:
        cal_idx = np.arange(n_cal)

    summary = {'stgnn': {}, 'e4d2': {}}
    pfas_list = list(pfas)
    B_ev = 512

    for mkey in (['stgnn'] if model_type in ('stgnn', 'both') else []):
        seed_metrics = []
        for sd in seeds:
            torch.manual_seed(sd); np.random.seed(sd)
            m = STGNN2ch().to(device)
            if sd == seeds[0]:
                npar_sg = sum(p.numel() for p in m.parameters())
            opt = optim.Adam(m.parameters(), lr=0.001)
            crit = nn.CrossEntropyLoss()

            for _ in range(epochs):
                m.train()
                perm = torch.randperm(n_train)
                for b in range(n_batches):
                    idx = perm[b * batch_size:(b + 1) * batch_size]
                    x = X_stgnn[idx].to(device)
                    lb = y_t[idx].to(device)
                    opt.zero_grad()
                    lo = m(x)
                    loss = crit(lo.permute(0, 2, 1).reshape(-1, 2), lb.reshape(-1))
                    loss.backward(); opt.step()

            # Calibrate
            m.eval()
            cs = []
            with torch.no_grad():
                for i in range(0, len(cal_idx), 128):
                    ci = cal_idx[i:i+128]
                    cp = torch.softmax(m(X_stgnn[ci].to(device)), 1)[:, 0, :].cpu().numpy()
                    for j in range(cp.shape[0]):
                        mask = y_train[cal_idx[i+j]] == 0
                        cs.extend(cp[j][mask].tolist())
            ths = {pfa: calibrate_far(cs, pfa) for pfa in pfas}

            # Evaluate
            all_cp, all_lb = [], []
            with torch.no_grad():
                for i in range(0, len(X_test), B_ev):
                    end = min(i + B_ev, len(X_test))
                    x = torch.tensor(np.stack([X_test[i:end].real, X_test[i:end].imag], axis=1),
                                    dtype=torch.float32, device=device)
                    cp = torch.softmax(m(x), 1)[:, 0, :].cpu().numpy().ravel()
                    all_cp.append(cp); all_lb.append(y_test[i:end].ravel())
            cp = np.concatenate(all_cp); lb = np.concatenate(all_lb)
            auc = float(roc_auc_score(lb, 1.0 - cp))
            metrics = {'auc': auc}
            for pfa in pfas_list:
                pred = (cp <= ths[pfa]).astype(int)
                tp_n = int(((pred == 1) & (lb == 1)).sum())
                fn_n = int(((pred == 0) & (lb == 1)).sum())
                fp_n = int(((pred == 1) & (lb == 0)).sum())
                tn_n = int(((pred == 0) & (lb == 0)).sum())
                metrics[pfa] = {
                    'pd': tp_n / (tp_n + fn_n + 1e-10),
                    'actual_pfa': fp_n / (fp_n + tn_n + 1e-10),
                }
            seed_metrics.append(metrics)

        aucs = [sm['auc'] for sm in seed_metrics]
        summary['stgnn'] = {
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'params': npar_sg,
        }
        for pfa in pfas_list:
            pds = [sm[pfa]['pd'] for sm in seed_metrics]
            summary['stgnn'][f'pd_{pfa}_mean'] = float(np.mean(pds))
            summary['stgnn'][f'pd_{pfa}_std'] = float(np.std(pds))

    for mkey in (['e4d2'] if model_type in ('e4d2', 'both') else []):
        seed_metrics = []
        for sd in seeds:
            torch.manual_seed(sd); np.random.seed(sd)
            m = E4D2().to(device)
            if sd == seeds[0]:
                npar_ed = sum(p.numel() for p in m.parameters())
            opt = optim.Adam(m.parameters(), lr=0.001)
            crit = nn.CrossEntropyLoss()

            for _ in range(epochs):
                m.train()
                perm = torch.randperm(n_train)
                for b in range(n_batches):
                    idx = perm[b * batch_size:(b + 1) * batch_size]
                    inp = {k: v[idx].to(device) for k, v in e4d2_tensors.items()}
                    lb = y_t[idx].to(device)
                    opt.zero_grad()
                    lo = m(**inp)
                    loss = crit(lo.permute(0, 2, 1).reshape(-1, 2), lb.reshape(-1))
                    loss.backward(); opt.step()

            # Calibrate
            m.eval()
            cs = []
            with torch.no_grad():
                for i in range(0, len(cal_idx), 128):
                    ci = cal_idx[i:i+128]
                    inp = {k: v[ci].to(device) for k, v in e4d2_tensors.items()}
                    cp = torch.softmax(m(**inp), 1)[:, 0, :].cpu().numpy()
                    for j in range(cp.shape[0]):
                        mask = y_train[cal_idx[i+j]] == 0
                        cs.extend(cp[j][mask].tolist())
            ths = {pfa: calibrate_far(cs, pfa) for pfa in pfas}

            # Evaluate
            e4d2_test = prep.transform(X_test)
            all_cp, all_lb = [], []
            with torch.no_grad():
                for i in range(0, len(X_test), B_ev):
                    end = min(i + B_ev, len(X_test))
                    inp = {k: torch.tensor(e4d2_test[k][i:end], device=device)
                           for k in ['x_main', 'x_phase_cell', 'x_amp_map',
                                     'x_amp_cell', 'x_global']}
                    cp = torch.softmax(m(**inp), 1)[:, 0, :].cpu().numpy().ravel()
                    all_cp.append(cp); all_lb.append(y_test[i:end].ravel())
            cp = np.concatenate(all_cp); lb = np.concatenate(all_lb)
            auc = float(roc_auc_score(lb, 1.0 - cp))
            metrics = {'auc': auc}
            for pfa in pfas_list:
                pred = (cp <= ths[pfa]).astype(int)
                tp_n = int(((pred == 1) & (lb == 1)).sum())
                fn_n = int(((pred == 0) & (lb == 1)).sum())
                fp_n = int(((pred == 1) & (lb == 0)).sum())
                tn_n = int(((pred == 0) & (lb == 0)).sum())
                metrics[pfa] = {
                    'pd': tp_n / (tp_n + fn_n + 1e-10),
                    'actual_pfa': fp_n / (fp_n + tn_n + 1e-10),
                }
            seed_metrics.append(metrics)

        aucs = [sm['auc'] for sm in seed_metrics]
        summary['e4d2'] = {
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'params': npar_ed,
        }
        for pfa in pfas_list:
            pds = [sm[pfa]['pd'] for sm in seed_metrics]
            summary['e4d2'][f'pd_{pfa}_mean'] = float(np.mean(pds))
            summary['e4d2'][f'pd_{pfa}_std'] = float(np.std(pds))

    return summary


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train on IPIX Dartmouth')
    parser.add_argument('--model', choices=['stgnn', 'e4d2', 'both'], default='both')
    parser.add_argument('--data-dir',
                        default='ipix_dartmouth/processed/window4_stride4_primary')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--max-train', type=int, default=20000)
    parser.add_argument('--output', default='checkpoints/ipix_results.json')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    PFA = (0.0001, 0.001, 0.01)

    print(f"IPIX Dartmouth: model={args.model}, epochs={args.epochs}, "
          f"seeds={seeds}, max_train={args.max_train}")

    X_train, y_train, X_test, y_test = load_ipix_splits(args.data_dir)
    results = run_experiment(X_train, y_train, X_test, y_test,
                             args.epochs, args.batch_size, seeds, PFA,
                             args.max_train, args.model)

    output = {
        'config': vars(args),
        'train_total': int(len(X_train)), 'test_total': int(len(X_test)),
        'P': int(X_train.shape[1]), 'N': int(X_train.shape[2]),
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")

    s = results.get('stgnn', {})
    e = results.get('e4d2', {})
    print(f"\n{'Metric':<20} {'ST-GNN':>14} {'E4-D2':>14}")
    print('-' * 48)
    if s:
        print(f"AUC               {s['auc_mean']:>10.4f}±{s['auc_std']:.4f}  "
              f"{e.get('auc_mean',0):>10.4f}±{e.get('auc_std',0):.4f}")
        for pfa in PFA:
            print(f"Pd@PFA={pfa}    {s[f'pd_{pfa}_mean']:>10.4f}±{s[f'pd_{pfa}_std']:.4f}  "
                  f"{e.get(f'pd_{pfa}_mean',0):>10.4f}±{e.get(f'pd_{pfa}_std',0):.4f}")


if __name__ == '__main__':
    main()
