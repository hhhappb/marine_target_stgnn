"""E4-D2 experiment scripts.

Usage:
    # SDRDSP dataset (with SCR injection):
    python -m e4d2.experiments.train_sdrdsp \\
        --train data/20210106155330_01_staring.mat \\
        --test  data/20210106155432_01_staring.mat \\
        --epochs 200 --seeds 42,123,456

    # IPIX Dartmouth dataset (real targets):
    python -m e4d2.experiments.train_ipix \\
        --data-dir ipix_dartmouth/processed/window4_stride4_primary \\
        --model both --epochs 200

Note: CVOCA ablation and detector head optimization experiments
(ablation.py, detector_heads.py) were removed from this package as they
depend on root-level scripts that predate the e4d2 reorganization.
See REMOVED.md in the project root for details.
"""
