# marine_target_stgnn

论文代码仓库。

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Structure

- `train.py`, `run_train.py`, `train_paper_reproduction_v2.py`: training entry points
- `evaluate_final.py`: evaluation entry point
- `configs/ipix_stgnn.yaml`: default IPIX training configuration
- `models/`, `paper_model/`: model implementations
- `data/`, `datasets/`: prepared and raw datasets
- `checkpoints/`: saved model weights
- `logs/training/`: training and evaluation run logs
