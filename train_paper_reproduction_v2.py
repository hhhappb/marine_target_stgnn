"""
ST-GNN 论文复现训练脚本 v2
修复：使用全局归一化，保留不同SCR间的幅度差异
从checkpoint继续训练
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import json

from models.st_gnn import STGNNDetector


class PaperStrictDataset(Dataset):
    def __init__(self, data_dir, P=4, N=256, train=True, norm_stats=None):
        self.data_dir = data_dir
        self.P = P
        self.N = N
        self.train = train
        self.norm_stats = norm_stats
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        if self.train:
            npz_path = os.path.join(self.data_dir, 'train.npz')
            npz_data = np.load(npz_path)
            self.data = npz_data['X']
            self.labels = npz_data['y']
        else:
            npz_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('test_scr_')])
            all_X, all_y = [], []
            for f in npz_files:
                npz_data = np.load(os.path.join(self.data_dir, f))
                all_X.append(npz_data['X'])
                all_y.append(npz_data['y'])
            self.data = np.concatenate(all_X, axis=0)
            self.labels = np.concatenate(all_y, axis=0)

    def compute_norm_stats(self):
        all_real = self.data[:, 0, :, :].reshape(-1)
        all_imag = self.data[:, 1, :, :].reshape(-1)
        return {
            'real_mean': all_real.mean().item(),
            'real_std': all_real.std().item(),
            'imag_mean': all_imag.mean().item(),
            'imag_std': all_imag.std().item(),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        E_real = torch.tensor(X[0], dtype=torch.float32)
        E_imag = torch.tensor(X[1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.norm_stats is not None:
            E_real = (E_real - self.norm_stats['real_mean']) / (self.norm_stats['real_std'] + 1e-8)
            E_imag = (E_imag - self.norm_stats['imag_mean']) / (self.norm_stats['imag_std'] + 1e-8)

        return E_real, E_imag, label


class PerSCRTestDataset(Dataset):
    def __init__(self, npz_path, P=4, N=256, norm_stats=None):
        npz_data = np.load(npz_path)
        self.data = npz_data['X']
        self.labels = npz_data['y']
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        E_real = torch.tensor(X[0], dtype=torch.float32)
        E_imag = torch.tensor(X[1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.norm_stats is not None:
            E_real = (E_real - self.norm_stats['real_mean']) / (self.norm_stats['real_std'] + 1e-8)
            E_imag = (E_imag - self.norm_stats['imag_mean']) / (self.norm_stats['imag_std'] + 1e-8)

        return E_real, E_imag, label


def evaluate_model(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for E_real, E_imag, labels in loader:
            E = torch.complex(E_real.to(device), E_imag.to(device))
            _, logits, _, _ = model(E, return_features=True)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def compute_far_threshold(clutter_probs, target_pfa):
    o = np.sort(clutter_probs)
    N_c = len(o)
    threshold_idx = int(np.ceil(target_pfa * N_c)) - 1
    threshold_idx = max(0, min(threshold_idx, N_c - 1))
    return o[threshold_idx]


def evaluate_per_scr(model, data_dir, device, norm_stats, pfa_values=[0.0001, 0.001, 0.01]):
    test_files = sorted([f for f in os.listdir(data_dir) if f.startswith('test_scr_')])
    results = {}

    model.eval()
    for f in test_files:
        scr_str = f.replace('test_scr_', '').replace('.npz', '')
        scr = int(scr_str)
        ds = PerSCRTestDataset(os.path.join(data_dir, f), norm_stats=norm_stats)
        loader = DataLoader(ds, batch_size=50, shuffle=False)
        logits, labels = evaluate_model(model, loader, device)

        probs = torch.softmax(logits, dim=1)
        o0 = probs[:, 0, :].numpy()
        labels_np = labels.numpy()
        results[scr] = {'o0': o0, 'labels': labels_np}

    all_clutter_o0 = np.concatenate([results[scr]['o0'][results[scr]['labels'] == 0] for scr in sorted(results.keys())])

    scr_pd = {}
    for pfa in pfa_values:
        threshold = compute_far_threshold(all_clutter_o0, pfa)
        scr_pd[pfa] = {}
        for scr in sorted(results.keys()):
            o0 = results[scr]['o0']
            labels = results[scr]['labels']
            decisions = (o0 <= threshold).astype(np.float32)

            TP = ((decisions == 1) & (labels == 1)).sum()
            FN = ((decisions == 0) & (labels == 1)).sum()
            FP = ((decisions == 1) & (labels == 0)).sum()
            TN = ((decisions == 0) & (labels == 0)).sum()

            pd_val = TP / (TP + FN) if (TP + FN) > 0 else 0
            pf_val = FP / (FP + TN) if (FP + TN) > 0 else 0
            scr_pd[pfa][scr] = {'PD': pd_val, 'PF': pf_val}

    return scr_pd


def train():
    data_dir = './data/paper_strict_256'
    device = torch.device('cuda')
    P, N = 4, 256
    epochs = 500
    batch_size = 128
    lr = 0.001

    print("=" * 60)
    print("ST-GNN 论文复现训练 v2 (全局归一化)")
    print("=" * 60)
    print(f"设备: {device} ({torch.cuda.get_device_name(0)})")
    print(f"训练轮数: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"学习率: {lr} (AdamW)")
    print("=" * 60)

    train_dataset = PaperStrictDataset(data_dir, P=P, N=N, train=True)
    norm_stats = train_dataset.compute_norm_stats()
    print(f"\n全局归一化统计:")
    print(f"  Real: mean={norm_stats['real_mean']:.2f}, std={norm_stats['real_std']:.2f}")
    print(f"  Imag: mean={norm_stats['imag_mean']:.2f}, std={norm_stats['imag_std']:.2f}")

    train_dataset.norm_stats = norm_stats
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"\n训练集: {len(train_dataset)} 样本")
    target_ratio = train_dataset.labels.mean()
    print(f"目标比例: {target_ratio:.4f} ({target_ratio*100:.2f}%)")

    # Try to load previous best model
    model = STGNNDetector(P=P, N=N).to(device)
    ckpt_path = './checkpoints/best_model_paper.pth'
    if os.path.exists(ckpt_path):
        print(f"\n从checkpoint加载: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("\n从头开始训练")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([0.5305, 8.6847], device=device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    best_loss = float('inf')
    start_time = time.time()

    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for E_real, E_imag, labels in train_loader:
            E = torch.complex(E_real.to(device), E_imag.to(device))
            labels = labels.to(device)

            optimizer.zero_grad()
            _, logits, _, _ = model(E, return_features=True)
            loss = criterion(logits.view(-1, 2), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), './checkpoints/best_model_paper_v2.pth')

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr_now:.6f} | Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳 Loss: {best_loss:.6f}")

    torch.save(model.state_dict(), './checkpoints/final_model_paper_v2.pth')

    print("\n" + "=" * 60)
    print("按SCR评估检测性能")
    print("=" * 60)

    best_state = torch.load('./checkpoints/best_model_paper_v2.pth')
    ckpt_copy = {k.replace('norm_stats.', ''): v for k, v in best_state.items()}
    model.load_state_dict(best_state)
    model.eval()

    pfa_values = [0.0001, 0.001, 0.01]
    scr_results = evaluate_per_scr(model, data_dir, device, norm_stats, pfa_values)

    print("\nPd-SCR Results:")
    header = f"{'SCR(dB)':>8}"
    for pfa in pfa_values:
        header += f" {'Pfa='+str(pfa):>12}"
    print(header)
    print("-" * 60)

    for scr in sorted(scr_results[pfa_values[0]].keys()):
        row = f"{scr:8d}"
        for pfa in pfa_values:
            row += f" {scr_results[pfa][scr]['PD']:12.4f}"
        print(row)

    with open('./checkpoints/pd_scr_results_v2.json', 'w') as f:
        json.dump(scr_results, f, indent=2)

    return model, scr_results


if __name__ == '__main__':
    model, results = train()
