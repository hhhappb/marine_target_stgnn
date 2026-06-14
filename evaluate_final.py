"""
ST-GNN 论文复现 - 最终评估脚本
生成 Pd-SCR 曲线和完整指标报告
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.st_gnn import STGNNDetector

device = torch.device('cuda')
data_dir = './data/paper_strict_256'
P, N = 4, 256

print("="*60)
print("ST-GNN 论文复现 - 最终评估")
print("="*60)

# Load model
model = STGNNDetector(P=P, N=N).to(device)
ckpt = torch.load('./checkpoints/best_model_repro.pth')
incompatible = model.load_state_dict(ckpt, strict=False)
if incompatible.missing_keys:
    raise RuntimeError(f"Missing checkpoint keys: {incompatible.missing_keys}")
if incompatible.unexpected_keys:
    print(f"Ignored extra checkpoint keys: {incompatible.unexpected_keys}")
model.eval()
print(f"模型加载成功 | Params: {sum(p.numel() for p in model.parameters()):,}")

# Evaluate per SCR
test_files = sorted([f for f in os.listdir(data_dir) if f.startswith('test_scr_')], 
                     key=lambda x: int(x.replace('test_scr_','').replace('.npz','')))

scr_data = {}
for f in test_files:
    scr = int(f.replace('test_scr_','').replace('.npz',''))
    d = np.load(os.path.join(data_dir, f))
    Xt, yt = d['X'], d['y']
    
    r = torch.tensor(Xt[:, 0], dtype=torch.float32, device=device)
    im = torch.tensor(Xt[:, 1], dtype=torch.float32, device=device)
    r = (r - r.mean((1,2), keepdim=True)) / (r.std((1,2), keepdim=True) + 1e-8)
    im = (im - im.mean((1,2), keepdim=True)) / (im.std((1,2), keepdim=True) + 1e-8)
    
    with torch.no_grad():
        _, logits, _, _ = model(torch.complex(r, im), return_features=True)
    scr_data[scr] = {
        'o0': torch.softmax(logits, 1)[:, 0].cpu().numpy().flatten(),
        'o1': torch.softmax(logits, 1)[:, 1].cpu().numpy().flatten(),
        'labels': yt.flatten()
    }

# Compute FAR thresholds from ALL clutter samples
all_clutter_o0 = np.concatenate([scr_data[s]['o0'][scr_data[s]['labels']==0] for s in sorted(scr_data)])
print(f"\n杂波样本总数: {len(all_clutter_o0)}")
print(f"杂波o(0)分布: min={all_clutter_o0.min():.8f}, max={all_clutter_o0.max():.6f}, "
      f"mean={all_clutter_o0.mean():.4f}, std={all_clutter_o0.std():.4f}")

# Evaluate at 3 Pfa values
pfa_values = [0.0001, 0.001, 0.01]
results = {}

for pfa in pfa_values:
    o = np.sort(all_clutter_o0)
    idx = int(np.ceil(pfa * len(o))) - 1
    th = o[max(0, min(idx, len(o)-1))]
    results[pfa] = {'threshold': float(th), 'data': {}}
    
    print(f"\n{'='*60}")
    print(f"Pfa = {pfa} | Threshold h = {th:.8f}")
    print(f"{'='*60}")
    print(f"  {'SCR(dB)':>8}  {'PD':>8}  {'PF':>10}  {'TP':>6}  {'FN':>6}  {'FP':>8}  {'TN':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}")
    
    for scr in sorted(scr_data):
        o0 = scr_data[scr]['o0']
        labs = scr_data[scr]['labels']
        det = (o0 <= th).astype(float)
        
        TP = ((det==1)&(labs==1)).sum()
        FN = ((det==0)&(labs==1)).sum()
        FP = ((det==1)&(labs==0)).sum()
        TN = ((det==0)&(labs==0)).sum()
        
        pd = TP/(TP+FN) if (TP+FN)>0 else 0
        pf = FP/(FP+TN) if (FP+TN)>0 else 0
        
        results[pfa]['data'][scr] = {'PD': float(pd), 'PF': float(pf)}
        print(f"  {scr:8d}  {pd:8.4f}  {pf:10.6f}  {TP:6d}  {FN:6d}  {FP:8d}  {TN:8d}")

# Save results
with open('./checkpoints/pd_scr_results_final.json', 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

# ======== Plot Pd-SCR curves ========
plt.figure(figsize=(10, 7))
colors = ['#2196F3', '#FF9800', '#4CAF50']
markers = ['o', 's', '^']

for i, pfa in enumerate(pfa_values):
    scrs = sorted(results[pfa]['data'].keys())
    pds = [results[pfa]['data'][s]['PD'] for s in scrs]
    plt.plot(scrs, pds, color=colors[i], marker=markers[i], markersize=6,
             linewidth=2, label=f'ST-GNN Pfa={pfa}')

plt.xlabel('SCR (dB)', fontsize=14)
plt.ylabel('Detection Probability $P_d$', fontsize=14)
plt.title('ST-GNN Detection Performance (Pd vs SCR)', fontsize=16)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('./pd_scr_curve_stgnn.png', dpi=150)
print("\n图表已保存: ./pd_scr_curve_stgnn.png")

# ======== Summary ========
print("\n" + "="*60)
print("评估摘要")
print("="*60)
for pfa in pfa_values:
    pds = [results[pfa]['data'][s]['PD'] for s in sorted(results[pfa]['data'])]
    pfs = [results[pfa]['data'][s]['PF'] for s in sorted(results[pfa]['data'])]
    print(f"Pfa={pfa}: avg PD={np.mean(pds):.4f}, max PD={np.max(pds):.4f}, "
          f"avg PF={np.mean(pfs):.6f}, target PF={pfa}")

print("\n评估完成!")
