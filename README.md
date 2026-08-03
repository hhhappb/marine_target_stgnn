# marine_target_stgnn

论文代码仓库。

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Structure

- `paper_modules/experiments/train.py`: unified training and evaluation entry point
- `paper_modules/experiments/auto_experiment.py`: suite runner for comparable experiments
- `paper_modules/experiments/leakage_probe.py`: diagnostic probe for range-position leakage
- `models/`: frozen ST-GNN baseline model implementations
- `paper_modules/models/`: configurable experimental model modules
- `paper_modules/datasets/`: dataset registry and dataset adapters
- `data/`, `datasets/`: prepared and raw datasets
- `checkpoints/`: saved model weights
- `logs/training/`: training and evaluation run logs

## Commands

IPIX 预处理默认使用 McMaster 官方 `ipixload.m` 的 `auto` 分支，逐距离单元处理并保留 double 精度：

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_ipix.py
```

输出固定写入 `datasets/ipix_dartmouth/processed/window4_stride4_related_official_ipixload_auto_double`。
脚本拒绝覆盖已存在的输出目录。官方源码地址为
`https://soma.ece.mcmaster.ca/ipix/dartmouth/mfiles/ipixload.m`，固化 SHA-256 为
`40f5499eeb0d1b1d7e158658bdcfddb31a18ef01eea4023e7da8ca0ba21bd517`。
训练接口通过 `dataset.expected_processing_mode: official_ipixload_auto` 校验数据来源，
并仅在模型输入边界将 complex128 转成现有模型契约所需的 complex64/float32。

IPIX module smoke:

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\real_imag_sfe_replacement_original_sfe.yaml --epochs 1 --max-train-windows 64 --max-test-windows-per-file 5 --no-progress --log-interval 1
```

生成 SDRDSP Fig.9 v2 数据（`N=256` 局部裁剪，论文第 2083 单元按一基编号处理）：

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_sdrdsp.py --raw-dir datasets\sdrdsp\raw
```

运行 SDRDSP v2 完整数据 smoke，并用独立 run_id 保存结果：

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --configs paper_modules\configs\repro_original_stgnn_sdrdsp_strict256_v2.yaml --seeds 42 --epochs 2 --target-pfa 0.001 --name sdrdsp_v2_full_smoke --stop-on-failure
```

