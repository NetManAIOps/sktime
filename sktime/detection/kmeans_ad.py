"""Sliding-window K-Means distance detector retaining the DevAD convention."""

import numpy as np
from sklearn.cluster import KMeans

from sktime.detection.base import BaseDetector
from sktime.detection.utils.window import sliding_windows


class KMeansAD(BaseDetector):
    """Score preprocessed windows by distance to their assigned cluster center.

    Standardize the series with an upstream transformer before windowing.
    Each window's Euclidean distance is assigned to its last point;
    the first win_len - 1 positions are padded with zero.
    Validation data and extra fit arguments are unused; there is no internal
    validation-based model selection. seed controls sklearn's random_state.
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "np.ndarray",
        "capability:multivariate": False,
        "capability:missing_values": False,
        "python_dependencies": "scikit-learn",
        "property:randomness": "stochastic",
    }

    def __init__(self, win_len, k=20, seed=0):
        self.win_len = win_len
        self.k = k
        self.seed = seed
        super().__init__()

    def _fit(self, X, y=None, **kwargs):
        windows = sliding_windows(X, self.win_len)
        self.backend_ = KMeans(n_clusters=self.k, random_state=self.seed)
        self.backend_.fit(windows)

    def _transform_scores(self, X, **kwargs):
        windows = sliding_windows(X, self.win_len).astype(
            self.backend_.cluster_centers_.dtype, copy=False
        )
        clusters = self.backend_.predict(windows)
        scores = np.linalg.norm(windows - self.backend_.cluster_centers_[clusters], axis=1)
        return np.pad(np.asarray(scores, dtype=float), (self.win_len - 1, 0))

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"win_len": 4, "k": 2}
