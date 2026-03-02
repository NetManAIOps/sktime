"""SMETS distance for multivariate time series."""

from __future__ import annotations

__author__ = ["GitHub Copilot"]

from collections.abc import Callable
from typing import Any

import numpy as np

from sktime.distances._ddtw import _DdtwDistance
from sktime.distances._dtw import _DtwDistance
from sktime.distances._edr import _EdrDistance
from sktime.distances._erp import _ErpDistance
from sktime.distances._euclidean import _EuclideanDistance
from sktime.distances._lcss import _LcssDistance
from sktime.distances._msm import _MsmDistance
from sktime.distances._sbd import _SbdDistance
from sktime.distances._squared import _SquaredDistance
from sktime.distances._twe import _TweDistance
from sktime.distances._wddtw import _WddtwDistance
from sktime.distances._wdtw import _WdtwDistance
from sktime.distances.base import NumbaDistance
from sktime.utils.numba.njit import njit


_UTS_METRIC_INSTANCES = {
    "euclidean": _EuclideanDistance(),
    "ed": _EuclideanDistance(),
    "euclid": _EuclideanDistance(),
    "pythagorean": _EuclideanDistance(),
    "linear": _EuclideanDistance(),
    "squared": _SquaredDistance(),
    "dtw": _DtwDistance(),
    "ddtw": _DdtwDistance(),
    "wdtw": _WdtwDistance(),
    "wddtw": _WddtwDistance(),
    "lcss": _LcssDistance(),
    "edr": _EdrDistance(),
    "erp": _ErpDistance(),
    "msm": _MsmDistance(),
    "twe": _TweDistance(),
    "sbd": _SbdDistance(),
}


def _resolve_uts_callable(
    uts_metric: str | Callable | NumbaDistance,
    x_1d_2d: np.ndarray,
    y_1d_2d: np.ndarray,
    uts_metric_kwargs: dict[str, Any],
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Resolve inner UTS distance callable from string/callable/NumbaDistance."""
    if isinstance(uts_metric, NumbaDistance):
        return uts_metric.distance_factory(x_1d_2d, y_1d_2d, **uts_metric_kwargs)

    if isinstance(uts_metric, str):
        metric_key = uts_metric.lower()
        if metric_key == "smets":
            raise ValueError("uts_metric='smets' is not allowed to avoid recursion.")

        if metric_key not in _UTS_METRIC_INSTANCES:
            allowed = sorted(set(_UTS_METRIC_INSTANCES.keys()))
            raise ValueError(
                "Unknown uts_metric: "
                f"{uts_metric}. Supported values are {allowed}, "
                "or a NumbaDistance/callable."
            )

        metric_inst = _UTS_METRIC_INSTANCES[metric_key]
        return metric_inst.distance_factory(x_1d_2d, y_1d_2d, **uts_metric_kwargs)

    if callable(uts_metric):
        return uts_metric

    raise ValueError(
        "uts_metric must be a string, a callable, or an instance of NumbaDistance."
    )


@njit(cache=True)
def _entropy_hist_1d(x: np.ndarray, bins: int = 20) -> float:
    """Approximate entropy of a 1D series using a histogram."""
    n = x.shape[0]
    if n == 0:
        return 0.0

    x_min = x[0]
    x_max = x[0]
    for i in range(1, n):
        if x[i] < x_min:
            x_min = x[i]
        if x[i] > x_max:
            x_max = x[i]

    hist = np.zeros(bins)

    if x_max == x_min:
        hist[0] = n
    else:
        scale = bins / (x_max - x_min)
        for i in range(n):
            idx = int((x[i] - x_min) * scale)
            if idx >= bins:
                idx = bins - 1
            if idx < 0:
                idx = 0
            hist[idx] += 1.0

    total = np.sum(hist)
    if total == 0.0:
        return 0.0

    entropy = 0.0
    for i in range(bins):
        if hist[i] > 0.0:
            p = hist[i] / total
            entropy -= p * np.log(p)

    return entropy


@njit(cache=True)
def _smets_core(
    x: np.ndarray,
    y: np.ndarray,
    uts_callable: Callable[[np.ndarray, np.ndarray], float],
    p_norm: float,
    entropy_bins: int,
) -> float:
    """Core SMETS distance between two 2D MTS arrays [n_dims, n_timepoints]."""
    d1 = x.shape[0]
    d2 = y.shape[0]
    m = x.shape[1]

    dist_matrix = np.zeros((d1, d2))

    for i in range(d1):
        xi = x[i : i + 1, :]
        for j in range(d2):
            yj = y[j : j + 1, :]
            dist_matrix[i, j] = uts_callable(xi, yj)

    used_i = np.zeros(d1, dtype=np.int64)
    used_j = np.zeros(d2, dtype=np.int64)
    num_pairs = d1 if d1 < d2 else d2
    matched_dists = np.zeros(num_pairs)

    for k in range(num_pairs):
        best_val = np.inf
        best_i = -1
        best_j = -1

        for i in range(d1):
            if used_i[i] == 1:
                continue
            for j in range(d2):
                if used_j[j] == 1:
                    continue
                curr = dist_matrix[i, j]
                if curr < best_val:
                    best_val = curr
                    best_i = i
                    best_j = j

        used_i[best_i] = 1
        used_j[best_j] = 1
        matched_dists[k] = best_val

    norm_sum = 0.0
    for k in range(num_pairs):
        norm_sum += np.abs(matched_dists[k]) ** p_norm
    norm_d = norm_sum ** (1.0 / p_norm) if num_pairs > 0 else 0.0

    entropy_sum = 0.0
    unmatched_count = 0

    for i in range(d1):
        if used_i[i] == 0:
            entropy_sum += _entropy_hist_1d(x[i], bins=entropy_bins)
            unmatched_count += 1

    for j in range(d2):
        if used_j[j] == 0:
            entropy_sum += _entropy_hist_1d(y[j], bins=entropy_bins)
            unmatched_count += 1

    EP = entropy_sum / unmatched_count if unmatched_count > 0 else 0.0

    d_max = d1 if d1 > d2 else d2
    d_min = d2 if d1 > d2 else d1
    P = (d_max - d_min) / (d_max + d_min) if (d_max + d_min) > 0 else 0.0

    return np.sqrt((norm_d + EP) ** 2 + P**2)


class _SmetsDistance(NumbaDistance):
    """SMETS distance for multivariate time series.

    SMETS combines:
    1) greedy matched UTS distances with L_p aggregation,
    2) entropy penalty for unmatched dimensions,
    3) dimensional difference penalty.
    """

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p_norm: float = 2.0,
        entropy_bins: int = 20,
        uts_metric: str | Callable | NumbaDistance = "euclidean",
        uts_metric_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Create a (numba-compatible) SMETS distance callable."""
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                "SMETS currently requires equal length time series, "
                f"but found lengths {x.shape[1]} and {y.shape[1]}."
            )

        if p_norm <= 0:
            raise ValueError(f"p_norm must be > 0, but found {p_norm}.")

        if entropy_bins <= 0:
            raise ValueError(
                f"entropy_bins must be a positive integer, but found {entropy_bins}."
            )

        if uts_metric_kwargs is None:
            uts_metric_kwargs = {}

        # If user puts inner metric kwargs at top level, merge as fallback.
        if kwargs:
            merged = dict(uts_metric_kwargs)
            merged.update(kwargs)
            uts_metric_kwargs = merged

        x1 = x[0:1, :]
        y1 = y[0:1, :]
        uts_callable = _resolve_uts_callable(uts_metric, x1, y1, uts_metric_kwargs)

        def _distance(xi: np.ndarray, yi: np.ndarray) -> float:
            return _smets_core(
                xi,
                yi,
                uts_callable=uts_callable,
                p_norm=float(p_norm),
                entropy_bins=int(entropy_bins),
            )

        return _distance