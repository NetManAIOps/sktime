"""NumPy windows for already preprocessed univariate detector inputs."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def sliding_windows(X, win_len):
    """Return stride-one windows (n - win_len + 1, win_len), retaining dtype.

    X is a univariate series, already checked by BaseDetector. This function
    does not impute, scale or convert values to Torch tensors.
    """
    if win_len < 1:
        raise ValueError("win_len must be >= 1")
    return sliding_window_view(np.asarray(X).reshape(-1), win_len)
