"""
ST-GNN 复现 - 500 epochs 快速训练
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.optim as optim, numpy as np, json, time
from models.st_gnn import STGNNDetector

device = torch.device('cuda')
data_dir = './data/paper_strict_256'
P, N, B, EPOCHS = 4, 256, 128, 500

print(f"ST-GNN | GPU: {torch.cuda.get_device_name(0)} | Epochs: {EPOCHS}", flush=True)

d = np.load(os.path.join(data_dir, 'train.npz'))
X_train, y_train = d['X'], d['y']
print(f"Train: {len(X_train)} samples, target ratio={y_train.mean():.4f}", flush=True)

nr = X_train[:, 0].reshape(-1); ni = X_train[:, 1].reshape(-1)
norm = (nr.mean(), nr.std(), ni.mean(), ni.std())
del nr, ni
Xr = np.clip((X_train[:, 0] - norm[0]) / (norm[1]+1e-8), -5, 5)
Xi = np.clip((X_train[:, 1] - norm[2]) / (norm[3]+1e-8), -5, 5)

model = STGNNDetector(P=P, N=N).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=250, T_mult=2, eta_min=1e-5)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5305, 8.6847], device=device))

best_loss = float('inf')
ns, idx = len(X_train), np.arange(len(X_train))
t0 = time.time()

for ep in range(EPOCHS):
    model.train()
    np.random.shuffle(idx)
    ls, nb = 0.0, 0
    for start in range(0, ns, B):
        bidx = idx[start:start+B]
        r = torch.tensor(Xr[bidx], dtype=torch.float32, device=device)
        im = torch.tensor(Xi[bidx], dtype=torch.float32, device=device)
        labs = torch.tensor(y_train[bidx], dtype=torch.long, device=device)
        opt.zero_grad()
        _, logits, _, _ = model(torch.complex(r, im), return_features=True)
        loss = criterion(logits.view(-1, 2), labs.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ls += loss.item(); nb += 1
    scheduler.step()
    al = ls / nb
    if al < best_loss:
        best_loss = al
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(model.state_dict(), './checkpoints/best_model_repro.pth')
    if (ep+1) % 100 == 0 or ep == 0:
        elapsed = time.time() - t0
        print(f"Ep {ep+1:3d}/{EPOCHS} | Loss: {al:.6f} | {elapsed:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\nDone! {elapsed/60:.1f}min | Best loss: {best_loss:.6f}", flush=True)

model.load_state_dict(torch.load('./checkpoints/best_model_repro.pth'))
model.eval()

# Evaluate
test_files = sorted([f for f in os.listdir(data_dir) if f.startswith('test_scr_')])
scr_data = {}
for f in test_files:
    scr = int(f.replace('test_scr_','').replace('.npz',''))
    d = np.load(os.path.join(data_dir, f))
    r = torch.clamp((torch.tensor(d['X'][:, 0], dtype=torch.float32, device=device) - norm[0]) / (norm[1]+1e-8), -5, 5)
    im = torch.clamp((torch.tensor(d['X'][:, 1], dtype=torch.float32, device=device) - norm[2]) / (norm[3]+1e-8), -5, 5)
    with torch.no_grad():
        _, logits, _, _ = model(torch.complex(r, im), return_features=True)
    scr_data[scr] = {'o0': torch.softmax(logits, 1)[:, 0].cpu().numpy(), 'labels': d['y']}

all_clutter = np.concatenate([scr_data[s]['o0'][scr_data[s]['labels']==0] for s in sorted(scr_data)])
results = {}
for pfa in [0.0001, 0.001, 0.01]:
    o = np.sort(all_clutter)
    th = o[max(0, min(int(np.ceil(pfa * len(o)))-1, len(o)-1))]
    results[pfa] = {}
    print(f"\nPfa={pfa}: th={th:.6f}", flush=True)
    print(f"  {'SCR':>6}  {'PD':>8}  {'PF':>10}", flush=True)
    for scr in sorted(scr_data):
        det = (scr_data[scr]['o0'] <= th).astype(float)
        labs = scr_data[scr]['labels']
        TP = ((det==1)&(labs==1)).sum(); FN = ((det==0)&(labs==1)).sum()
        FP = ((det==1)&(labs==0)).sum(); TN = ((det==0)&(labs==0)).sum()
        results[pfa][scr] = {'PD': float(TP/(TP+FN) if (TP+FN)>0 else 0), 'PF': float(FP/(FP+TN) if (FP+TN)>0 else 0)}
        print(f"  {scr:6d}  {results[pfa][scr]['PD']:8.4f}  {results[pfa][scr]['PF']:10.6f}", flush=True)

torch.save(model.state_dict(), './checkpoints/final_model_repro.pth')
with open('./checkpoints/pd_scr_repro.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nDone!", flush=True)
