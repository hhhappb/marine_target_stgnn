"""SE experiment scripts.

Usage (run from ``/tmp/ST-GNN/SE`` with this directory on ``PYTHONPATH``)::

    # IPIX v2.x (window8_stride4_related, primary_strict, SE P=8)
    python -m experiments.train_ipix --conditions 19931109_191449_starea:hh \\
        --epochs 60 --output checkpoints/se_ipix.json

    # v2.1 / PAX-L1 quick-8 screening (label06-hh ... ) and full 56-point cohort
    python -m experiments.run_v21_ipix --conditions label06:hh --epochs 1
    python -m experiments.run_v21_ipix --all-56
"""
