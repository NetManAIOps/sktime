from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .Base import BaseModel, DetectResult
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class IForest(BaseModel):
    HP = {
        "win_len": 1,
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": 0.1,
        "max_features": 1.0,
        "bootstrap": False,
        "n_jobs": 1,
        "verbose": 0,
        "normalize": True,
        "scale_mode": "zscore",
    }

    def __init__(self, config):
        super().__init__(config)
        self.scale_cfg: dict = {}

        self.backend = IsolationForest(
            n_estimators=int(self.params["n_estimators"]),
            max_samples=self.params["max_samples"],
            contamination=float(self.params["contamination"]),
            max_features=self.params["max_features"],
            bootstrap=bool(self.params["bootstrap"]),
            n_jobs=int(self.params["n_jobs"]),
            random_state=self.seed,
            verbose=int(self.params["verbose"]),
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
        values = self._impute(x_train)
        features = self._build_features(values, fit_scale=True)
        self.backend.fit(features)

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        values = self._impute(x_test)
        features = self._build_features(values, fit_scale=False)
        scores = -self.backend.decision_function(features)
        win_len = int(self.params["win_len"])
        return DetectResult(scores=scores, start_pos=win_len - 1)

    @staticmethod
    def _impute(values: np.ndarray) -> np.ndarray:
        return (
            pd.Series(values)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy()
        )

    def _build_features(self, values: np.ndarray, fit_scale: bool) -> np.ndarray:
        win_len = int(self.params["win_len"])
        if win_len < 1:
            raise ValueError("win_len must be >= 1")

        series = values.reshape(-1, 1)
        n_samples = series.shape[0]

        if win_len == 1:
            features = series
        else:
            if n_samples < win_len:
                raise ValueError("Sequence too short for the specified window length")
            windows = np.stack([series[i : i + win_len] for i in range(n_samples - win_len + 1)], axis=0)
            features = windows.reshape(windows.shape[0], -1)

        if not bool(self.params["normalize"]):
            return features

        scale_mode = str(self.params["scale_mode"]).strip().lower()
        if scale_mode not in {"zscore", "minmax"}:
            raise ValueError(f"Unsupported scale_mode: {scale_mode}")

        if fit_scale or not self.scale_cfg:
            if scale_mode == "minmax":
                data_min = features.min(axis=0)
                data_max = features.max(axis=0)
                self.scale_cfg = {"scale_mode": "minmax", "min": data_min, "max": data_max}
            else:
                mean = features.mean(axis=0)
                std = features.std(axis=0)
                std = np.where(std == 0, 1.0, std)
                self.scale_cfg = {"scale_mode": "zscore", "mean": mean, "std": std}

        cfg_mode = self.scale_cfg.get("scale_mode", scale_mode)
        if cfg_mode == "minmax":
            data_min = self.scale_cfg["min"]
            data_max = self.scale_cfg["max"]
            denom = data_max - data_min
            denom = np.where(denom == 0, 1.0, denom)
            features = (features - data_min) / denom
        else:
            mean = self.scale_cfg["mean"]
            std = self.scale_cfg["std"]
            std = np.where(std == 0, 1.0, std)
            features = (features - mean) / std

        return features
