"""Sliding-window novelty LOF retaining the DevAD scoring convention."""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from sktime.detection.base import BaseDetector
from sktime.detection.utils.window import sliding_windows


class SubLOF(BaseDetector):
    """Fit one novelty LOF to overlapping windows of a preprocessed series.

    Standardize the series with an upstream transformer before windowing.
    Each window's negative decision value is assigned to its last point;
    the first win_len - 1 positions are padded with zero.
    Unlike the former interval-based LOF, this does not fit separate models to
    non-overlapping time intervals. Validation data and extra fit arguments
    are unused; no validation-based model selection is performed internally.
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
        self, win_len, n_neighbors=20, algorithm="auto", metric="minkowski", n_jobs=4
    ):
        self.win_len = win_len
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y=None, **kwargs):
        windows = sliding_windows(X, self.win_len)
        self.backend_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            metric=self.metric,
            n_jobs=self.n_jobs,
            novelty=True,
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
        return {"win_len": 4, "n_neighbors": 2}
