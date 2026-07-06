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

IPIX module smoke:

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\real_imag_sfe_replacement_original_sfe.yaml --epochs 1 --max-train-windows 64 --max-test-windows-per-file 5 --no-progress --log-interval 1
```

SDRDSP SCR reproduction smoke:

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\train.py --config paper_modules\configs\repro_original_stgnn_scr256.yaml --epochs 2 --no-progress --log-interval 10
```

IPIX Fig.7 hard-point per-file protocol check:

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_b.yaml
```

Validate the hard-point configs before launching runs:

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_b.yaml --validate-only
```

IPIX Fig.7 hard-point per-file run with train-time range-roll augmentation:

```powershell
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_shift_aug_b.yaml --validate-only
.\.venv\Scripts\python.exe paper_modules\experiments\auto_experiment.py --suite paper_modules\configs\suites\per_file_fig7_hardpoints_shift_aug_b.yaml
```
