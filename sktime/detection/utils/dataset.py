"""Sliding reconstruction windows for already preprocessed time series."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ReconDataset(Dataset):
    """Window already preprocessed values and optional named aligned sequences.

    X is converted to float32 with shape (n, d). Additional sequences must have
    the same first-axis length as X; their remaining axes and dtypes are retained.
    Samples are x_window alone, or (x_window, extras) when extras are supplied.
    No normalization, imputation or masking of X is performed here.
    """

    def __init__(self, X, win_len, **aligned):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError("ReconDataset requires shape (n,) or (n, d)")
        if win_len < 1 or len(values) < win_len:
            raise ValueError("win_len must be positive and not exceed sequence length")

        self.values = torch.tensor(values, dtype=torch.float32)
        self.win_len = win_len
        self.aligned = {}
        for name, sequence in aligned.items():
            sequence = np.asarray(sequence)
            if sequence.ndim == 0 or len(sequence) != len(values):
                raise ValueError(f"{name} must have the same first-axis length as X")
            self.aligned[name] = torch.tensor(sequence)

    def __len__(self):
        return len(self.values) - self.win_len + 1

    def __getitem__(self, index):
        window = self.values[index:index + self.win_len]
        if not self.aligned:
            return window
        return window, {
            name: sequence[index:index + self.win_len]
            for name, sequence in self.aligned.items()
        }


class ForecastDataset(Dataset):
    """Return history (W, d) and future (P, d), without normalization."""

    def __init__(self, X, win_len, pred_len):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError("ForecastDataset requires shape (n,) or (n, d)")
        if win_len < 1 or pred_len < 1 or len(values) < win_len + pred_len:
            raise ValueError("Positive history and prediction windows must fit the sequence")
        data = torch.tensor(values, dtype=torch.float32)
        indices = np.arange(len(values) - win_len - pred_len + 1)
        self.samples = torch.stack([data[i:i + win_len] for i in indices])
        self.targets = torch.stack([
            data[i + win_len:i + win_len + pred_len] for i in indices
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]
