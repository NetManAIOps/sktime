"""Windowed K-Means distance detector, following TAB's KMeans implementation.

Reference: decisionintelligence/TAB, 5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2.
DevAD assigns each window distance to its endpoint instead of averaging
overlapping window scores. Standardization is fitted on train only.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .Base import REQUIRED, BaseModel, DetectResult
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class KMeansAD(BaseModel):
    HP = {
        "win_len": REQUIRED,
        "k": 20,
    }

    STATEFUL_ATTRS = BaseModel.STATEFUL_ATTRS + ("scaler",)

    def __init__(self, config):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.backend = KMeans(n_clusters=self.params["k"], random_state=self.seed)

    def _fit(
        self,
        x_train: np.ndarray,
        y=None,
        reporter: TrainingReporter | None = None,
        **kwargs,
    ) -> None:
        reporter = reporter or NullTrainingReporter()
        reporter.begin_stage("Fitting")
        values = self.scaler.fit_transform(x_train.reshape(-1, 1)).ravel()
        windows = sliding_window_view(values, self.params["win_len"])
        self.backend.fit(windows)

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        values = self.scaler.transform(x_test.reshape(-1, 1)).ravel()
        windows = sliding_window_view(values, self.params["win_len"])
        clusters = self.backend.predict(windows)
        scores = np.linalg.norm(windows - self.backend.cluster_centers_[clusters], axis=1)
        return DetectResult(scores=scores, start_pos=self.params["win_len"] - 1)
