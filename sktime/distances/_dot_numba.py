"""Isolated numba imports for dot (cosine) distance."""

__author__ = ["chrisholder", "TonyBagnall"]

import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True, fastmath=True)
def _numba_dot_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Dot-distance (1 - cosine similarity) compiled to no_python.

    Distance = 1 - cosine_similarity(x, y), where cosine_similarity in [-1, 1].
    Resulting distance is in [0, 2].

    Parameters
    ----------
    x: np.ndarray (2d array shape (d, m))
        First time series.
    y: np.ndarray (2d array shape (d, m))
        Second time series.
    Returns
    -------
    float
        Dot distance (1 - cosine similarity) between x and y.
    """
    dot_val = 0.0
    norm_x_sq = 0.0
    norm_y_sq = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dot_val += x[i, j] * y[i, j]
            norm_x_sq += x[i, j] * x[i, j]
            norm_y_sq += y[i, j] * y[i, j]
    eps = 1e-12
    norm_x = np.sqrt(norm_x_sq) + eps
    norm_y = np.sqrt(norm_y_sq) + eps
    cosine = dot_val / (norm_x * norm_y)
    if cosine > 1.0:
        cosine = 1.0
    elif cosine < -1.0:
        cosine = -1.0
    return 1.0 - cosine
