"""Granger causality distance."""

__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any

import numpy as np
import pandas as pd

from sktime.distances.base import DistanceCallable, NumbaDistance


def _granger_distance_python(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> float:
    """Compute Granger-causality based distance using statsmodels.

    Uses grangercausalitytests from statsmodels. Tests both directions (x→y, y→x)
    and returns distance = 1 - min(p_value). Lower p-value indicates stronger
    Granger causality, hence smaller distance.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series, shape (d, m).
    y: np.ndarray (2d array)
        Second time series, shape (d, m).
    max_lag: int
        Maximum lag for Granger test.

    Returns
    -------
    float
        Granger distance in [0, 1].
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("statsmodels")
    from statsmodels.tsa.stattools import grangercausalitytests

    x_flat = x.ravel()
    y_flat = y.ravel()
    n = len(x_flat)
    if n < max_lag + 2:
        max_lag = max(1, n // 2)

    min_p = 1.0
    for dep, indep in [(y_flat, x_flat), (x_flat, y_flat)]:
        gc_data = pd.DataFrame({"dep": dep, "indep": indep})
        try:
            result = grangercausalitytests(
                gc_data[["dep", "indep"]], maxlag=max_lag, verbose=False
            )
            for lag, res in result.items():
                p_val = res[0]["ssr_ftest"][1]
                if np.isfinite(p_val) and p_val < min_p:
                    min_p = p_val
        except (ValueError, KeyError, Exception):
            pass
    return float(1.0 - min_p)


class _GrangerDistance(NumbaDistance):
    """Granger causality based distance between two time series.

    Uses statsmodels grangercausalitytests. Distance = 1 - min(p_value) over
    both directions and all lags. Not Numba-compiled (statsmodels is Python).
    """

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, max_lag: int = 5, **kwargs: Any
    ) -> DistanceCallable:
        """Create a Granger distance callable.

        Returns a Python callable (statsmodels cannot be Numba-compiled).

        Parameters
        ----------
        x: np.ndarray (1d or 2d array)
            First time series.
        y: np.ndarray (1d or 2d array)
            Second time series.
        max_lag: int, default=5
            Maximum lag for Granger causality test.
        kwargs: Any
            Extra kwargs (ignored).

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Granger distance callable.
        """

        def _granger_callable(_x: np.ndarray, _y: np.ndarray) -> float:
            return _granger_distance_python(_x, _y, max_lag=max_lag)

        return _granger_callable
