import numpy as np
import torch
from selected_e4 import E4Model, E4D2Preprocessor

# Shape-only example. Real use: fit only on processed training echoes.
rng = np.random.default_rng(42)
echoes = (rng.normal(size=(4,8,14))+1j*rng.normal(size=(4,8,14))).astype(np.complex64)
prep = E4D2Preprocessor().fit(echoes)
inputs = {name:torch.from_numpy(value) for name,value in prep.transform(echoes).items()}
model = E4Model().eval()
with torch.no_grad():
    logits = model(**inputs)
print(logits.shape)  # [4,2,14]
