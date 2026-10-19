"""Sliding-window One-Class SVM retaining the DevAD scoring convention."""

import numpy as np
from sklearn.svm import OneClassSVM

from sktime.detection.base import BaseDetector
from sktime.detection.utils.window import sliding_windows


class SubOCSVM(BaseDetector):
    """Fit One-Class SVM to overlapping windows of a preprocessed series.

    Standardize the series with an upstream transformer before windowing.
    Each window's negative decision value is assigned to its last point;
    the first win_len - 1 positions are padded with zero.
    No subsampling or approximate solver is used. Validation data and extra
    fit arguments are unused; there is no internal validation-based selection.
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "np.ndarray",
        "capability:multivariate": False,
        "capability:missing_values": False,
        "python_dependencies": "scikit-learn",
    }

    def __init__(
        self,
        win_len,
        kernel="rbf",
        degree=3,
        gamma="auto",
        coef0=0.0,
        tol=1e-3,
        nu=0.5,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):
        self.win_len = win_len
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        super().__init__()

    def _fit(self, X, y=None, **kwargs):
        windows = sliding_windows(X, self.win_len)
        self.backend_ = OneClassSVM(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            nu=self.nu,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
        )
        self.backend_.fit(windows)

    def _transform_scores(self, X, **kwargs):
        windows = sliding_windows(X, self.win_len)
        scores = -self.backend_.decision_function(windows)
        return np.pad(np.asarray(scores, dtype=float), (self.win_len - 1, 0))

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"win_len": 4}
