# IPIX Dartmouth preprocessing

Raw files live in `datasets/ipix_dartmouth/raw`.

The preprocessing script follows the Dartmouth IPIX MATLAB workflow:

1. Read NetCDF classic `.cdf` files and extract `adc_data`.
2. Select one polarization from `hh`, `hv`, `vv`, `vh`.
3. Apply official `auto` preprocessing per range cell:
   remove I/Q means, normalize I/Q standard deviations, and correct I/Q phase imbalance.
4. Mark target-related cells from the official Dartmouth primary/secondary table in `labels.json`.
5. Split sweeps by time: first 60% train, last 40% test.
6. Segment each split into fixed-length complex windows.

Example:

```powershell
python scripts/preprocess_ipix.py --dry-run
python scripts/preprocess_ipix.py --window 4 --stride 4
```

The current `paper_model.STGNNDetector` collapses the temporal axis with two stride-2 temporal layers, so `--window 4` is the compatible default.
