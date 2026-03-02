"""Isolated numba imports for SBD distance."""

import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True, fastmath=True)
def _local_squared_norm(x: np.ndarray) -> float:
    """Squared l2 norm for a 1D series."""
    norm_sq = 0.0
    for i in range(x.shape[0]):
        norm_sq += x[i] * x[i]
    return norm_sq


@njit(cache=True, fastmath=True)
def _local_max_ncc(x: np.ndarray, y: np.ndarray) -> float:
    """Maximum normalized cross-correlation for equal-length 1D series."""
    m = x.shape[0]
    denom = np.sqrt(_local_squared_norm(x) * _local_squared_norm(y))

    if denom == 0.0:
        return 0.0

    max_ncc = -np.inf

    # lag in [-(m-1), ..., (m-1)]
    for lag in range(-(m - 1), m):
        corr = 0.0
        for t in range(m):
            y_idx = t - lag
            if 0 <= y_idx < m:
                corr += x[t] * y[y_idx]

        ncc = corr / denom
        if ncc > max_ncc:
            max_ncc = ncc

    return max_ncc


@njit(cache=True, fastmath=True)
def _local_ncc_profile(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Full normalized cross-correlation profile for equal-length 1D series."""
    m = x.shape[0]
    denom = np.sqrt(_local_squared_norm(x) * _local_squared_norm(y))

    profile = np.zeros(2 * m - 1)

    if denom == 0.0:
        return profile

    idx = 0
    for lag in range(-(m - 1), m):
        corr = 0.0
        for t in range(m):
            y_idx = t - lag
            if 0 <= y_idx < m:
                corr += x[t] * y[y_idx]

        profile[idx] = corr / denom
        idx += 1

    return profile


@njit(cache=True, fastmath=True)
def _numba_ncc_profile(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Average NCC profile across dimensions for 2D time series input."""
    n_dims = x.shape[0]
    profile_len = 2 * x.shape[1] - 1
    profile_sum = np.zeros(profile_len)

    for i in range(n_dims):
        profile_sum += _local_ncc_profile(x[i], y[i])

    return profile_sum / n_dims


@njit(cache=True, fastmath=True)
def _numba_sbd_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Shape-based distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series, shape [n_dims, n_timepoints].
    y: np.ndarray (2d array)
        Second time series, shape [n_dims, n_timepoints].

    Returns
    -------
    float
        SBD distance between x and y, defined as 1 - max NCC.
    """
    n_dims = x.shape[0]
    ncc_sum = 0.0

    for i in range(n_dims):
        ncc_sum += _local_max_ncc(x[i], y[i])

    mean_max_ncc = ncc_sum / n_dims
    return 1.0 - mean_max_ncc
