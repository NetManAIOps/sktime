from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA

from .Base import REQUIRED, BaseModel, DetectResult
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class SubPCA(BaseModel):
    HP = {
        "win_len": REQUIRED,
        "n_components": None,
        "n_selected_components": None,
        "weighted": True,
        "svd_solver": "auto",
    }

    STATEFUL_ATTRS = BaseModel.STATEFUL_ATTRS + ("feature_mask",)

    def __init__(self, config):
        super().__init__(config)
        self.feature_mask: np.ndarray | None = None

        self.backend = PCA(
            n_components=self.params["n_components"],
            svd_solver=self.params["svd_solver"],
            random_state=self.seed,
        )

    def _fit(
        self,
        x_train: np.ndarray,
        y: np.ndarray | None = None,
        reporter: TrainingReporter | None = None,
        **kwargs,
    ) -> None:
        reporter = reporter or NullTrainingReporter()
        reporter.begin_stage("Fitting")
        windows = self._build_windows(x_train)
        features = self._fit_transform_features(windows)
        self.backend.fit(features)
        self._resolve_n_selected_components()

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        windows = self._build_windows(x_test)
        features = self._transform_features(windows)
        scores = self._score_features(features)

        return DetectResult(
            scores=scores,
            start_pos=int(self.params["win_len"]) - 1,
            direction="higher",
        )

    def _build_windows(self, values: np.ndarray) -> np.ndarray:
        win_len = int(self.params["win_len"])
        if win_len < 2:
            raise ValueError("win_len must be >= 2")
        if len(values) < win_len:
            raise ValueError(
                f"Sequence length must be at least win_len={win_len}, got {len(values)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("SubPCA input contains NaN or Inf")

        return sliding_window_view(values, window_shape=win_len).astype(
            np.float64,
            copy=True,
        )

    def _fit_transform_features(self, windows: np.ndarray) -> np.ndarray:
        mean = windows.mean(axis=0)
        std = windows.std(axis=0)
        self.feature_mask = std > 0

        if not self.feature_mask.any():
            raise ValueError("SubPCA requires at least one non-constant window feature")

        safe_std = np.where(self.feature_mask, std, 1.0)
        self.scale_cfg = {
            "scale_mode": "zscore",
            "mean": mean,
            "std": safe_std,
        }
        return ((windows - mean) / safe_std)[:, self.feature_mask]

    def _transform_features(self, windows: np.ndarray) -> np.ndarray:
        if not self.scale_cfg or self.feature_mask is None:
            raise RuntimeError("SubPCA feature preprocessing is not fitted")

        mean = self.scale_cfg["mean"]
        std = self.scale_cfg["std"]
        if windows.shape[1] != len(mean):
            raise ValueError(
                "Input window length does not match the fitted SubPCA model"
            )

        return ((windows - mean) / std)[:, self.feature_mask]

    def _resolve_n_selected_components(self) -> int:
        available = int(self.backend.components_.shape[0])
        configured = self.params["n_selected_components"]
        selected = available if configured is None else int(configured)

        if not 1 <= selected <= available:
            raise ValueError(
                "n_selected_components must be between 1 and the number of fitted "
                f"components ({available}), got {selected}"
            )
        return selected

    def _score_features(self, features: np.ndarray) -> np.ndarray:
        selected = self._resolve_n_selected_components()
        components = self.backend.components_[-selected:]

        if bool(self.params["weighted"]):
            weights = self.backend.explained_variance_ratio_[-selected:]
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
