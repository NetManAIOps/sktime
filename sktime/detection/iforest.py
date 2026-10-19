"""Windowed Isolation Forest retaining the DevAD scoring convention."""

import numpy as np
from sklearn.ensemble import IsolationForest

from sktime.detection.base import BaseDetector
from sktime.detection.utils.window import sliding_windows


class IForest(BaseDetector):
    """Fit Isolation Forest to overlapping windows of an imputed series.

    Missing and infinite values must be handled by upstream transformers.
    Optional normalization is performed after windowing, independently for
    each lag feature. Statistics are fitted on training windows and reused
    for scoring; an additional upstream scaler is generally unnecessary.
    Each window's negative decision value is assigned to its last point;
    the first win_len - 1 positions are padded with zero.
    Validation data and extra fit arguments are unused. seed controls the
    sklearn random_state; there is no internal validation-based selection.
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

    def __init__(
        self,
        win_len=1,
        n_estimators=100,
        max_samples="auto",
        contamination=0.1,
        max_features=1.0,
        bootstrap=False,
        n_jobs=1,
        verbose=0,
        normalize=True,
        scale_mode="zscore",
        seed=0,
    ):
        self.win_len = win_len
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.normalize = normalize
        self.scale_mode = scale_mode
        self.seed = seed
        super().__init__()

    def _fit(self, X, y=None, **kwargs):
        self.scale_cfg_ = {}
        features = self._build_features(X, fit_scale=True)
        self.backend_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=self.verbose,
        )
        self.backend_.fit(features)

    def _build_features(self, X, fit_scale):
        # DevAD builds a contiguous window matrix before reducing each column.
        features = sliding_windows(X, self.win_len).copy()
        if not self.normalize:
            return features

        mode = str(self.scale_mode).strip().lower()
        if mode not in {"zscore", "minmax"}:
            raise ValueError(f"Unsupported scale_mode: {self.scale_mode}")

        if fit_scale:
            if mode == "minmax":
                self.scale_cfg_ = {
                    "scale_mode": mode,
                    "min": features.min(axis=0),
                    "max": features.max(axis=0),
                }
            else:
                mean = features.mean(axis=0)
                std = features.std(axis=0)
                self.scale_cfg_ = {
                    "scale_mode": mode,
                    "mean": mean,
                    "std": np.where(std == 0, 1.0, std),
                }

        if self.scale_cfg_["scale_mode"] == "minmax":
            data_min = self.scale_cfg_["min"]
            denom = self.scale_cfg_["max"] - data_min
            denom = np.where(denom == 0, 1.0, denom)
            return (features - data_min) / denom

        return (features - self.scale_cfg_["mean"]) / self.scale_cfg_["std"]

    def _transform_scores(self, X, **kwargs):
        features = self._build_features(X, fit_scale=False)
        scores = -self.backend_.decision_function(features)
        return np.pad(np.asarray(scores, dtype=float), (self.win_len - 1, 0))

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"win_len": 4, "n_estimators": 10}
