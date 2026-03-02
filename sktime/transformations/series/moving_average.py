#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""K-order Moving Average smoothing transformer for time series."""

__author__ = ["wannabtl"]
__all__ = ["MovingAverageTransformer"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


def _moving_average_1d(x: np.ndarray, window_length: int, mode: str) -> np.ndarray:
    """Apply K-order moving average to a 1D array.

    Parameters
    ----------
    x : np.ndarray
        1D array to smooth
    window_length : int
        Window size (K) for the moving average
    mode : str
        One of "valid", "same", "full" - same as np.convolve mode

    Returns
    -------
    np.ndarray
        Smoothed 1D array
    """
    kernel = np.ones(window_length) / window_length
    return np.convolve(x, kernel, mode=mode)


class MovingAverageTransformer(BaseTransformer):
    """K-order Moving Average smoothing for time series.

    Applies a simple moving average (SMA) filter to reduce noise and emphasize
    trends. Uses convolution with a uniform kernel of size K.

    Parameters
    ----------
    window_length : int
        Window size (K) for the moving average. Must be >= 2.
    mode : str, default="valid"
        Convolution mode, determines output length:
        - "valid": output length = len(X) - window_length + 1 (no padding)
        - "same": output length = len(X) (same as input, with padding)
        - "full": output length = len(X) + window_length - 1

    Examples
    --------
    >>> from sktime.transformations.series.moving_average import MovingAverageTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = MovingAverageTransformer(window_length=7)
    >>> y_smoothed = transformer.fit_transform(y)
    """

    _tags = {
        "authors": ["wannabtl"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "transform-returns-same-time-index": False,
        "capability:multivariate": True,
        "capability:inverse_transform": False,
    }

    def __init__(self, window_length, mode="valid"):
        if window_length < 2:
            raise ValueError(
                f"window_length must be >= 2, got {window_length}"
            )
        if mode not in ("valid", "same", "full"):
            raise ValueError(
                f"mode must be one of 'valid', 'same', 'full', got '{mode}'"
            )
        self.window_length = window_length
        self.mode = mode
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X by applying moving average smoothing.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Time series to smooth
        y : ignored

        Returns
        -------
        Xt : pd.Series or pd.DataFrame
            Smoothed time series
        """
        if isinstance(X, pd.Series):
            x_arr = X.values.astype(float)
            smoothed = _moving_average_1d(x_arr, self.window_length, self.mode)
            # Build new index for output
            new_index = self._get_output_index(X.index, len(smoothed))
            return pd.Series(smoothed, index=new_index, name=X.name)
        else:
            # pd.DataFrame - apply to each column
            result = {}
            for col in X.columns:
                x_arr = X[col].values.astype(float)
                smoothed = _moving_average_1d(x_arr, self.window_length, self.mode)
                result[col] = smoothed
            new_len = len(next(iter(result.values())))
            new_index = self._get_output_index(X.index, new_len)
            return pd.DataFrame(result, index=new_index)

    def _get_output_index(self, original_index, new_len):
        """Create output index based on mode and original index."""
        n = len(original_index)
        k = self.window_length
        if self.mode == "valid":
            # Center-aligned: output[i] = avg of input[i:i+k], maps to index i+(k-1)//2
            start_offset = (k - 1) // 2
            return original_index[start_offset : start_offset + new_len]
        elif self.mode == "same":
            return original_index[:new_len]
        else:  # full - output longer than input
            if new_len <= n:
                return original_index[:new_len]
            # Use RangeIndex when output is longer (full mode)
            return pd.RangeIndex(0, new_len)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"window_length": 7, "mode": "valid"},
            {"window_length": 5, "mode": "same"},
        ]
