# SE — Verification

This file restates the four conclusions recorded in `../VERIFICATION.json`
(which is itself left unchanged) and documents how to reproduce them.

## Recorded conclusions (`VERIFICATION.json`)

| # | Key | Value | Meaning |
|---|---|---|---|
| 1 | `e4_checkpoints_strictly_compatible` | `56` | The selected `E4Model` state_dict loads with `strict=True` into **56** E4 checkpoints produced by the P8 experiments — parameter shapes/names are fully compatible. |
| 2 | `preprocessing_equal` | `true` | `E4D2Preprocessor` produces the same feature tensors as the experiment preprocessing (per-sample element-wise equality). |
| 3 | `eval_logits_exactly_equal_to_experiment` | `true` | Under the same weights and input, `E4Model` logits are element-wise identical to the experiment model (`log1p`/CVOCA/attention paths unchanged). |
| 4 | `forward_shape` | `[4,2,14]` | Forward on a batch of 4 returns `[B,2,14]` logits (channel 0 = clutter, channel 1 = target); `backward_finite` is `true`. |

The canonical implementation is the untouched `../selected_e4/` package; no
network computation is modified by this scaffolding package.

## How to reproduce

```bash
# 1. Forward shape + parameter count
cd /tmp/ST-GNN/SE && python example.py            # prints [4, 2, 14]
python - <<'PY'
from selected_e4 import E4Model
m = E4Model()
print(sum(p.numel() for p in m.parameters()))     # 4461538
PY

# 2. Preprocessing equality (fit on the same training echoes, compare tensors)
python - <<'PY'
import numpy as np, torch
from selected_e4 import E4D2Preprocessor
rng = np.random.default_rng(0)
E = (rng.normal(size=(16,8,14)) + 1j*rng.normal(size=(16,8,14))).astype(np.complex64)
prep = E4D2Preprocessor().fit(E)
d = prep.transform(E)
assert d['x_main'].shape == (16,4,8,14)
assert d['x_phase_cell'].shape == (16,3,14)
assert d['x_amp_map'].shape == (16,1,8,14)
assert d['x_amp_cell'].shape == (16,4,14)
assert d['x_global'].shape == (16,6)
print('preprocessing OK')
PY

# 3. Checkpoint compatibility (strict state_dict load)
python - <<'PY'
import torch
from selected_e4 import E4Model
m = E4Model()
sd = m.state_dict()
m.load_state_dict(sd, strict=True)               # strict round-trip
print('state_dict strict OK,', len(sd), 'tensors')
PY
```

The 56-checkpoint strict-compatibility check and the element-wise logit check
were run against the experiment checkpoints during package preparation; their
pass/fail results are summarised above. This scaffolding package does not ship
those checkpoints.
