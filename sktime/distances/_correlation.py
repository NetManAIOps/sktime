import numpy as np

from sktime.distances._numba_utils import to_numba_timeseries


def _validate_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    if _x.shape != _y.shape:
        raise ValueError("x and y must have the same shape.")
    return _x, _y


def pearson_correlation(
    x: np.ndarray, y: np.ndarray, eps: float = 1e-12
) -> float:
    _x, _y = _validate_pair(x, y)
    x_centered = _x - _x.mean(axis=1, keepdims=True)
    y_centered = _y - _y.mean(axis=1, keepdims=True)
    denom = np.sqrt((x_centered**2).sum(axis=1) * (y_centered**2).sum(axis=1))
    corr = (x_centered * y_centered).sum(axis=1) / (denom + eps)
    corr = np.clip(corr, -1.0, 1.0)
    return float(corr.mean())


def dcor1_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    corr = pearson_correlation(x, y, eps=eps)
    return float(np.sqrt(2.0 * (1.0 - corr)))


def dcor2_distance(
    x: np.ndarray, y: np.ndarray, beta: float = 1.0, eps: float = 1e-12
) -> float:
    corr = pearson_correlation(x, y, eps=eps)
    ratio = (1.0 - corr) / (1.0 + corr + eps)
    return float(np.sqrt(ratio) ** beta)


def cort_coefficient(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    _x, _y = _validate_pair(x, y)
    dx = np.diff(_x, axis=1)
    dy = np.diff(_y, axis=1)
    denom = np.sqrt((dx**2).sum(axis=1) * (dy**2).sum(axis=1))
    corr = (dx * dy).sum(axis=1) / (denom + eps)
    corr = np.clip(corr, -1.0, 1.0)
    return float(corr.mean())
