"""Interface module to scipy pairwise similarity kernels.

Interface module exposing correlation-style similarities as pairwise kernels.
"""

__author__ = ["fkiraly", "GitHub Copilot"]

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from sktime.dists_kernels.base import BasePairwiseTransformer


class ScipyKernel(BasePairwiseTransformer):
    """Interface to scipy similarity kernels.

    Computes pairwise similarities using scipy.stats correlation functions.

    Parameters
    ----------
    metric : str or callable, default="pearson"
        Similarity metric to use.

        If str, one of:
        - "pearson" (Pearson correlation)
        - "kendalltau" (Kendall's tau)
        - "spearman" (Spearman rank correlation)

        If callable, expected signature is
        ``callable(x_1d: np.ndarray, y_1d: np.ndarray, **metric_kwargs) -> float``
        or a scipy-style result where the first element/"statistic" is the score.
    colalign : str, one of {"intersect", "force-align", "none"}, default="intersect"
        Controls column alignment if X and X2 are pd.DataFrame.

        if ``intersect``, similarity is computed on columns occurring in both X and X2,
            other columns are discarded; column ordering in X2 is copied from X
        if ``force-align``, raises an error if the set of columns in X, X2 differs;
            column ordering in X2 is copied from X
        if ``none``, X and X2 are passed through unmodified (no columns are aligned)
    var_weights : 1D np.ndarray of float or None, default=None
        Weight/scaling vector applied to variables in X/X2
        before similarity computation. i-th column is multiplied by var_weights[i].
    metric_kwargs : dict, optional, default=None
        Additional keyword arguments passed to metric callable.
    nan_to_num : float or None, default=0.0
        Replacement for NaN metric values (e.g., constant vectors in Pearson).
        If None, NaN values are preserved.
    """

    _tags = {
        "authors": ["fkiraly", "GitHub Copilot"],
        "symmetric": True,
        "pwtrafo_type": "kernel",
    }

    def __init__(
        self,
        metric="pearson",
        colalign="intersect",
        var_weights=None,
        metric_kwargs=None,
        nan_to_num=0.0,
    ):
        self.metric = metric
        self.colalign = colalign
        self.var_weights = var_weights
        self.metric_kwargs = metric_kwargs
        self.nan_to_num = nan_to_num

        super().__init__()

    def _transform(self, X, X2=None):
        """Compute kernel/similarity matrix.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape [n, d]
        X2 : pd.DataFrame or np.ndarray of shape [m, d], optional

        Returns
        -------
        kernmat : np.ndarray of shape [n, m]
            (i,j)-th entry contains similarity/kernel between X[i] and X2[j].
        """
        metric = self.metric
        var_weights = self.var_weights
        metric_kwargs = self.metric_kwargs

        if metric_kwargs is None:
            metric_kwargs = {}

        if isinstance(X, pd.DataFrame) and isinstance(X2, pd.DataFrame):
            if self.colalign == "intersect":
                common_cols = X.columns.intersection(X2.columns)
                X = X[common_cols]
                X2 = X2[common_cols]
                X2 = X2[X.columns]
            elif self.colalign == "force-align":
                if not X.columns.equals(X2.columns):
                    raise ValueError("X and X2 have different columns")
                X2 = X2[X.columns]
            elif self.colalign == "none":
                pass
            else:
                raise ValueError("colalign must be one of intersect, force-align, none")

        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes("number").to_numpy(dtype="float")

        if isinstance(X2, pd.DataFrame):
            X2 = X2.select_dtypes("number").to_numpy(dtype="float")

        if np.ndim(var_weights) == 1:
            if len(var_weights) == X.shape[1] == X2.shape[1]:
                X = var_weights * X
                X2 = var_weights * X2
            else:
                raise ValueError(
                    "weights vector length must be equal to X and X2 number of columns"
                )

        metric_callable = self._resolve_metric_callable(metric)

        n, m = X.shape[0], X2.shape[0]
        kernmat = np.empty((n, m), dtype=float)

        for i in range(n):
            for j in range(m):
                val = metric_callable(X[i], X2[j], **metric_kwargs)
                val = self._to_statistic(val)
                kernmat[i, j] = val

        if self.nan_to_num is not None:
            kernmat = np.nan_to_num(kernmat, nan=self.nan_to_num)

        return kernmat

    @staticmethod
    def _resolve_metric_callable(metric):
        if callable(metric):
            return metric

        metric_map = {
            "pearson": pearsonr,
            "kendalltau": kendalltau,
            "spearman": spearmanr,
        }

        if metric not in metric_map:
            raise ValueError(
                "metric must be one of pearson, kendalltau, spearman, or a callable"
            )

        return metric_map[metric]

    @staticmethod
    def _to_statistic(result):
        if hasattr(result, "statistic"):
            return float(result.statistic)
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"metric": "kendalltau"}
        params3 = {"metric": "spearman", "colalign": "force-align"}

        return [params1, params2, params3]
