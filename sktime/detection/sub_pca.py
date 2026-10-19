"""Windowed PCA retaining DevAD's feature preprocessing and distance score."""

import numpy as np
from sklearn.decomposition import PCA

from sktime.detection.base import BaseDetector
from sktime.detection.utils.window import sliding_windows


class SubPCA(BaseDetector):
    """Fit PCA to standardized lag features of an imputed univariate series.

    Window columns are standardized using training statistics, then constant
    columns are removed using a training-fitted mask. Missing and infinite
    values must be handled upstream; another scaler is generally unnecessary.
    The original DevAD score sums distances to the selected trailing component
    vectors, optionally weighted by inverse explained-variance ratios. This is
    not a PCA reconstruction-error score.
    Each window's score is assigned to its last point; the first win_len - 1
    positions are padded with zero. Validation data and extra fit arguments
    are unused. seed controls randomized PCA solvers via random_state.
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
        win_len,
        n_components=None,
        n_selected_components=None,
        weighted=True,
        svd_solver="auto",
        seed=0,
    ):
        self.win_len = win_len
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.weighted = weighted
        self.svd_solver = svd_solver
        self.seed = seed
        super().__init__()

    def _fit(self, X, y=None, **kwargs):
        windows = self._build_windows(X)
        features = self._fit_transform_features(windows)
        self.backend_ = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.seed,
        )
        self.backend_.fit(features)
        self._resolve_n_selected_components()

    def _build_windows(self, X):
        win_len = int(self.win_len)
        if win_len < 2:
            raise ValueError("win_len must be >= 2")
        if not np.isfinite(X).all():
            raise ValueError("SubPCA input contains NaN or Inf")
        return sliding_windows(X, win_len).astype(np.float64, copy=True)

    def _fit_transform_features(self, windows):
        mean = windows.mean(axis=0)
        std = windows.std(axis=0)
        self.feature_mask_ = std > 0
        if not self.feature_mask_.any():
            raise ValueError("SubPCA requires at least one non-constant window feature")

        safe_std = np.where(self.feature_mask_, std, 1.0)
        self.scale_cfg_ = {"scale_mode": "zscore", "mean": mean, "std": safe_std}
        return ((windows - mean) / safe_std)[:, self.feature_mask_]

    def _transform_features(self, windows):
        mean = self.scale_cfg_["mean"]
        std = self.scale_cfg_["std"]
        return ((windows - mean) / std)[:, self.feature_mask_]

    def _resolve_n_selected_components(self):
        available = int(self.backend_.components_.shape[0])
        configured = self.n_selected_components
        selected = available if configured is None else int(configured)
        if not 1 <= selected <= available:
            raise ValueError(
                "n_selected_components must be between 1 and the number of fitted "
                f"components ({available}), got {selected}"
            )
        return selected

    def _score_features(self, features):
        selected = self._resolve_n_selected_components()
        components = self.backend_.components_[-selected:]
        if self.weighted:
            weights = self.backend_.explained_variance_ratio_[-selected:]
            weights = np.maximum(weights, np.finfo(np.float64).eps)
        else:
            weights = np.ones(selected, dtype=np.float64)

        squared_distances = (
            np.sum(features ** 2, axis=1, keepdims=True)
            + np.sum(components ** 2, axis=1)[None, :]
            - 2.0 * features @ components.T
        )
        distances = np.sqrt(np.maximum(squared_distances, 0.0))
        return np.sum(distances / weights[None, :], axis=1)

    def _transform_scores(self, X, **kwargs):
        windows = self._build_windows(X)
        features = self._transform_features(windows)
        scores = self._score_features(features)
        return np.pad(np.asarray(scores, dtype=float), (int(self.win_len) - 1, 0))

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"win_len": 4, "n_components": 2, "n_selected_components": 1}
