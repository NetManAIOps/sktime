"""Optimal Partitioning (dynamic programming) for change point detection."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["sktime developers"]
__all__ = ["OptimalPartitioning"]


class OptimalPartitioning(BaseDetector):
    """Optimal Partitioning with MLE (Maximum Likelihood Estimation) cost.

    Uses dynamic programming to find the globally optimal set of change
    points that minimises the total penalised Gaussian log-likelihood.

    Parameters
    ----------
    beta : float, default=40
        Penalty added for each change point.
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

    The recurrence is
    ``Q(t) = min_{0 <= tau < t} [ Q(tau) + C(y_{tau:t}) + beta ]``
    with ``Q(0) = -beta``.

    References
    ----------
    .. [1] Jackson, B., et al. "An algorithm for optimal partitioning of
           data on an interval." IEEE Signal Processing Letters, 2005.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.optimal_partitioning import OptimalPartitioning
    >>> X = pd.Series([1, 1, 1, 1, 5, 5, 5, 5])
    >>> model = OptimalPartitioning(beta=1)
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

    def _optimal_partitioning(self, y, beta):
        """Run the DP recurrence over *y*.

        Parameters
        ----------
        y : np.ndarray
            Full time series array.
        beta : float
            Penalty term.

        Returns
        -------
        list of int
            Change point indices (may include 0 as the first segment start).
        """
        n = len(y)
        Q = np.full(n + 1, float("inf"))
        Q[0] = -beta
        cp = {0: []}

        for t in range(1, n + 1):
            for tau in range(t):
                cost = Q[tau] + self._likelihood_mle(y[tau:t]) + beta
                if cost < Q[t]:
                    Q[t] = cost
                    cp[t] = cp[tau] + [tau]

        return cp[n]

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
        changepoints = self._optimal_partitioning(y, self.beta)
        if changepoints and changepoints[0] == 0:
            changepoints = changepoints[1:]
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
