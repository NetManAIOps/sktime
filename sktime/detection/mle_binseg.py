"""Binary Segmentation with MLE cost for change point detection."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["sktime developers"]
__all__ = ["MLEBinSeg"]


class MLEBinSeg(BaseDetector):
    """Binary Segmentation with MLE (Maximum Likelihood Estimation) cost.

    Recursively splits the time series at the point that yields the greatest
    reduction in total Gaussian log-likelihood, penalised by ``beta``.
    Splitting stops when no candidate split reduces the cost below zero.

    Parameters
    ----------
    beta : float, default=40
        Penalty added to each candidate split cost.
        Larger values yield fewer change points.

    Attributes
    ----------
    changepoints_ : list of int
        Detected change point indices after fitting.

    Notes
    -----
    The cost function used is the Gaussian MLE:

    .. math::
        C(y_{s:t}) = (t - s) \\log \\hat\\sigma^2_{s:t}

    A split at ``tau`` is accepted when
    ``C(y_{s:tau}) + C(y_{tau:t}) - C(y_{s:t}) + beta < 0``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.mle_binseg import MLEBinSeg
    >>> X = pd.Series([1, 1, 1, 1, 5, 5, 5, 5])
    >>> model = MLEBinSeg(beta=1)
    >>> model.fit_predict(X)
       ilocs
    0      4
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "fit_is_empty": True,
        "capability:multivariate": False,
        "X_inner_mtype": "pd.Series",
    }

    def __init__(self, beta=40):
        self.beta = beta
        super().__init__()

    @staticmethod
    def _likelihood_mle(y):
        """Gaussian MLE cost of a segment.

        Parameters
        ----------
        y : array-like
            Segment values.

        Returns
        -------
        float
            ``n * log(variance)`` of the segment,
            ``inf`` when variance is zero,
            ``0`` for empty segments.
        """
        n = len(y)
        if n == 0:
            return 0.0
        variance = np.var(y)
        if variance == 0:
            return float("inf")
        return n * np.log(variance)

    def _binseg_recurse(self, y, s, t, beta, changepoints):
        """Recursively find change points via binary segmentation.

        Parameters
        ----------
        y : np.ndarray
            Full time series array.
        s : int
            Start index of current segment (inclusive).
        t : int
            End index of current segment (exclusive).
        beta : float
            Penalty term.
        changepoints : list of int
            Accumulator for detected change points (mutated in place).
        """
        if t - s <= 1:
            return

        best_tau = None
        min_cost = float("inf")

        for tau in range(s + 1, t):
            cost = (
                self._likelihood_mle(y[s:tau])
                + self._likelihood_mle(y[tau:t])
                - self._likelihood_mle(y[s:t])
                + beta
            )
            if cost < min_cost:
                min_cost = cost
                best_tau = tau

        if min_cost < 0:
            changepoints.append(best_tau)
            self._binseg_recurse(y, s, best_tau, beta, changepoints)
            self._binseg_recurse(y, best_tau, t, beta, changepoints)

    def _predict(self, X):
        """Detect change points in *X*.

        Parameters
        ----------
        X : pd.Series
            Univariate time series.

        Returns
        -------
        pd.Series of int
            Sorted iloc indices of detected change points.
        """
        y = X.values
        changepoints = []
        self._binseg_recurse(y, 0, len(y), self.beta, changepoints)
        changepoints.sort()
        return pd.Series(changepoints, dtype="int64")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params0 = {"beta": 40}
        params1 = {"beta": 10}
        return [params0, params1]
