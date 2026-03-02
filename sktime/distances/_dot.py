"""Dot distance (1 - cosine similarity)."""

__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any

import numpy as np

from sktime.distances.base import DistanceCallable, NumbaDistance


class _DotDistance(NumbaDistance):
    """Dot distance between two time series.

    Defined as 1 - cosine_similarity(x, y).
    """

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled dot distance callable.

        Parameters
        ----------
        x: np.ndarray (1d or 2d array)
            First time series.
        y: np.ndarray (1d or 2d array)
            Second time series.
        kwargs: Any
            Extra kwargs (ignored).

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled dot distance callable.
        """
        from sktime.distances._dot_numba import _numba_dot_distance

        return _numba_dot_distance
