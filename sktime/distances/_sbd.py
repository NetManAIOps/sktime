"""Shape-based distance (SBD)."""

__author__ = ["GitHub Copilot"]

from typing import Any

import numpy as np

from sktime.distances.base import NumbaDistance


class _SbdDistance(NumbaDistance):
    """Shape-based distance between two time series.

    SBD is defined as ``1 - max(NCC)`` where NCC is normalized cross-correlation.
    """

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ):
        """Create a no_python compiled SBD distance callable."""
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "SBD requires the same number of dimensions in x and y, "
                f"but found {x.shape[0]} and {y.shape[0]}."
            )

        if x.shape[1] != y.shape[1]:
            raise ValueError(
                "SBD requires equal length time series, "
                f"but found lengths {x.shape[1]} and {y.shape[1]}."
            )

        from sktime.distances._sbd_numba import _numba_sbd_distance

        return _numba_sbd_distance
