import torch
from torch.utils.data import Dataset
import numpy as np
import os


class RadarDataset(Dataset):
    def __init__(self, data_dir, P=10, N=256, train=True):
        self.data_dir = data_dir
        self.P = P
        self.N = N
        self.train = train
        self.data = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        if self.train:
            npz_path = os.path.join(self.data_dir, 'train_data.npz')
        else:
            npz_path = os.path.join(self.data_dir, 'test_data.npz')

        if os.path.exists(npz_path):
            npz_data = np.load(npz_path)
            self.data = npz_data['data']
            self.labels = npz_data['labels']
        else:
            raise FileNotFoundError(f"Data file not found in {self.data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        E = self.data[idx]

        if E.dtype != np.complex64 and E.dtype != np.complex128:
            E = E['real'] + 1j * E['imag']

        actual_pulses = E.shape[0]

        if actual_pulses > self.P:
            start_idx = np.random.randint(0, actual_pulses - self.P)
            E = E[start_idx:start_idx + self.P]
        elif actual_pulses < self.P:
            padding_needed = self.P - actual_pulses
            E = np.vstack([E, np.zeros((padding_needed, self.N), dtype=np.complex64)])

        E_real = torch.tensor(E.real, dtype=torch.float32)
        E_imag = torch.tensor(E.imag, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return E_real, E_imag, label


class RadarDataProcessor:
    def __init__(self, P=10, N=256):
        self.P = P
        self.N = N

    def process_raw_data(self, raw_data):
        if raw_data.dtype != np.complex64 and raw_data.dtype != np.complex128:
            raw_data = raw_data['real'] + 1j * raw_data['imag']

        E_real = torch.tensor(raw_data.real, dtype=torch.float32)
        E_imag = torch.tensor(raw_data.imag, dtype=torch.float32)

        return E_real, E_imag

    def create_adjacency_matrix(self, N):
        adj = np.zeros((N, N))
        for i in range(N):
            adj[i, i] = 1
            if i > 0:
                adj[i, i - 1] = 1
            if i < N - 1:
                adj[i, i + 1] = 1
        return adj
