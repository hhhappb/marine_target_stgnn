"""SE (selected E4 slots + GDCM-A0) on SDRDSP — protocol v1.1 (Pd-SCR).

⚠️ 协议适配配置：SE 固定包为 P=8 / N=14（IPIX）。本脚本为 v1.1（SDRDSP）
重新实例化 **同一个 selected 时空槽位 + GDCM-A0** 于 **P=4 / N=256**，
即 "v1.1 适配配置"，不是 SE 固定 P=8/N=14 原样包的直接运行。

协议 v1.1（与 ST-GNN Fig.9 / e4d2_gdcm 完全一致）:
  * 数据: SDRDSP 20210106155330 (train) / 20210106155432 (test),
    距离窗 [1955, 2211) = 256 bins, 目标 idx 128
  * 目标注入 (Eq.17): SCR 训练 -12..+14 step2 / 测试 -24..+14 step2,
    M=20 参考单元, amp = sqrt(clutter_power * 10^(SCR/10)),
    φ0 ~ U(-π,π) 每样本随机（v1.1 强制）
  * 训练: 200 epochs × 3 seeds (42/123/456), batch 24, Adam lr=1e-3, 普通 CE
  * FAR: Eq.15 — 训练集全部杂波单元 o(0) 升序, h = o_sorted[ceil(α_f·Nc)]
  * 输出: 三档 PFA {1e-4, 1e-3, 1e-2} 的完整 Pd-SCR 表 + AUC + PdL

Usage (from /tmp/ST-GNN/SE):
    PYTHONPATH=/tmp/ST-GNN/SE python -m experiments.train_sdrdsp \
        --train /tmp/ST-GNN/data/20210106155330_01_staring.mat \
        --test  /tmp/ST-GNN/data/20210106155432_01_staring.mat \
        --epochs 200 --seeds 42,123,456 --batch 24 \
        --output /tmp/ST-GNN/SE/checkpoints/se_v11.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                    # /tmp/ST-GNN/SE
for _p in (_PKG, "/tmp/ST-GNN"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selected_e4.e4_backbone import E4Backbone       # noqa: E402
from selected_e4.detector import GdcmDetector        # noqa: E402
from selected_e4.preprocess import E4D2Preprocessor  # noqa: E402

# ── v1.1 SDRDSP constants ──
P = 4
N_RANGE = 256
STEP = P
PRT = 1.0 / 1600
WAVELENGTH = 0.03
TARGET_RANGE = N_RANGE // 2
POSITIONS_PER_SAMPLE = 5
PFA_LEVELS = [0.0001, 0.001, 0.01]
PHI0_RNG_SEED = 777
SB = 1955

_SLOTS = json.loads(
    Path(_PKG, "selected_e4", "config.json").read_text())["spatiotemporal"]


class SEModelV11(nn.Module):
    """SE 的 selected 时空槽位 + GDCM-A0，按 v1.1 实例化为 P=4/N=256。"""

    def __init__(self):
        super().__init__()
        self.backbone = E4Backbone(P=P, spatiotemporal=_SLOTS)
        self.detector = GdcmDetector(1216, variant="a0")

    def forward(self, x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global):
        return self.detector(self.backbone(
            x_main, x_phase_cell, x_amp_map, x_amp_cell, x_global))


def count_params(model):
    n = sum(p.numel() for p in model.parameters())
    return n, f"{n:,}"


def load_crop(mat_path):
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    key = [k for k in mat if not k.startswith("__")][0]
    data = mat[key].astype(np.complex128)
    if data.shape[1] == N_RANGE:
        return data
    return data[:, SB:SB + N_RANGE]


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
    """Eq.17 injection with per-sample random φ0 ~ U(-π,π)."""
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
            seg, labels = inject_target(seg, gen_positions(st), gen_speed(st),
                                        scr, phi0_rng)
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
        result[scr] = {"X": np.array(Xs, dtype=np.complex64),
                       "y": np.array(ys, dtype=np.int32)}
    return result


def process_stat(cplx_batch, preprocessor):
    d = preprocessor.transform(cplx_batch)
    return (torch.tensor(d["x_main"], dtype=torch.float32),
            torch.tensor(d["x_phase_cell"], dtype=torch.float32),
            torch.tensor(d["x_amp_map"], dtype=torch.float32),
            torch.tensor(d["x_amp_cell"], dtype=torch.float32),
            torch.tensor(d["x_global"], dtype=torch.float32))


def train_one_seed(seed, preprocessor, X_tr, y_tr, s_tr, device, epochs, batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SEModelV11().to(device)
    n_params, _ = count_params(model)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()          # paper Eq.16: unweighted CE
    unique_scr = np.unique(s_tr)
    scr_indices = {s: np.where(s_tr == s)[0] for s in unique_scr}
    n_per_scr = batch_size // len(unique_scr) + 1

    for _ in range(epochs):
        model.train()
        batch_idx = []
        for s in unique_scr:
            si = scr_indices[s]
            batch_idx.extend(np.random.choice(si, min(n_per_scr, len(si)),
                                              replace=False).tolist())
        batch_idx = np.array(batch_idx[:batch_size])
        np.random.shuffle(batch_idx)
        t = [x.to(device) for x in process_stat(X_tr[batch_idx], preprocessor)]
        lb = torch.tensor(y_tr[batch_idx], dtype=torch.long, device=device)
        opt.zero_grad()
        lo = model(*t)
        loss = crit(lo.permute(0, 2, 1).reshape(-1, 2), lb.reshape(-1))
        loss.backward()
        opt.step()
    return model, n_params


def calibrate_from_train(model, preprocessor, X_tr, y_tr, device, batch_size=24):
    """Eq.15: sort o(0) of ALL training clutter cells."""
    model.eval()
    cc_list = []
    with torch.no_grad():
        for b in range(0, len(X_tr), batch_size):
            t = [x.to(device) for x in process_stat(X_tr[b:b + batch_size], preprocessor)]
            lo = model(*t)
            cs = torch.softmax(lo, 1)[:, 0].cpu().numpy()       # [b, N]
            lb = y_tr[b:b + batch_size]
            cc_list.append(cs[lb == 0])
    cc_all = np.concatenate(cc_list)
    sorted_cal = np.sort(cc_all)
    Nc = len(sorted_cal)
    thresholds = {pfa: float(sorted_cal[max(0, min(int(np.ceil(pfa * Nc)) - 1, Nc - 1))])
                  for pfa in PFA_LEVELS}
    return thresholds, Nc


def evaluate(model, preprocessor, test_data, scr_list, thresholds, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    psc = {}
    with torch.no_grad():
        for scr in scr_list:
            d = test_data[scr]
            t = [x.to(device) for x in process_stat(d["X"], preprocessor)]
            lo = model(*t)
            cs = torch.softmax(lo, 1)[:, 0].cpu().numpy().flatten()
            psc[scr] = (cs, d["y"].flatten())

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
        results[pfa] = {"pdL": pdL, "pd_per_scr": pd_per_scr,
                        "threshold": float(h)}
    return auc, results


def main():
    parser = argparse.ArgumentParser(description="SE (selected E4 slots) on SDRDSP v1.1")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--output", type=str, default="/tmp/ST-GNN/SE/checkpoints/se_v11.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train", type=int, default=None, help="debug only")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"SE v1.1 (P=4/N=256, selected slots + GDCM-A0) | device={device} "
          f"| seeds={seeds} | epochs={args.epochs} | batch={args.batch}", flush=True)

    train_raw = load_crop(args.train)
    test_raw = load_crop(args.test)
    train_scr = list(range(-12, 15, 2))
    test_scr = list(range(-24, 15, 2))
    X_tr, y_tr, s_tr = build_dataset(train_raw, train_scr)
    test_data = build_test_data(test_raw, test_scr)
    if args.max_train:
        keep = np.random.RandomState(0).choice(len(X_tr), args.max_train, replace=False)
        X_tr, y_tr, s_tr = X_tr[keep], y_tr[keep], s_tr[keep]
    n_clutter = int((y_tr == 0).sum())
    print(f"Train {X_tr.shape[0]} samples | Test "
          f"{sum(len(v['X']) for v in test_data.values())} | Clutter cells {n_clutter}",
          flush=True)

    preprocessor = E4D2Preprocessor().fit(X_tr)

    all_results, aucs = [], []
    t_all = time.time()
    for seed in seeds:
        t0 = time.time()
        model, n_params = train_one_seed(seed, preprocessor, X_tr, y_tr, s_tr,
                                         device, args.epochs, args.batch)
        thresholds, Nc = calibrate_from_train(model, preprocessor, X_tr, y_tr,
                                              device, batch_size=args.batch)
        auc, multi = evaluate(model, preprocessor, test_data, test_scr,
                              thresholds, device)
        aucs.append(auc)
        all_results.append({
            "seed": seed, "auc": auc,
            **{f"pdL_pfa{int(10000*p)}": v["pdL"] for p, v in multi.items()},
            **{f"threshold_pfa{int(10000*p)}": v["threshold"] for p, v in multi.items()},
            "cal_Nc": int(Nc),
            "pd_scr": {str(s): v["pd_per_scr"] for s, v in multi.items()},
        })
        print(f"  Seed {seed}: AUC={auc:.4f}  "
              f"PdL@1e-3={multi[0.001]['pdL']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "protocol": "v1.1",
        "model": "SE (selected E4 slots + GDCM-A0) — v1.1 adapted config P=4/N=256",
        "note": "SE 固定包为 P=8/N=14；本结果为 v1.1(SDRDSP) 适配配置，"
                "槽位与检测头不变，仅按 v1.1 重实例化为 P=4/N=256。",
        "epochs": args.epochs, "seeds": seeds, "batch": args.batch,
        "params": int(n_params), "N_c": int(n_clutter),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "results": all_results,
        "total_seconds": round(time.time() - t_all, 1),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nAUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | Params: {n_params:,}")
    for p in PFA_LEVELS:
        v = np.mean([r[f"pdL_pfa{int(10000*p)}"] for r in all_results])
        print(f"  PdL @ PFA={p}: {v:.4f}")
    print(f"Saved: {args.output}  |  {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
