"""Windowed novelty LOF, following EasyTSAD SubLOF defaults.

DevAD fits only train and assigns each window's score to its endpoint.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .Base import REQUIRED, BaseModel, DetectResult
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class SubLOF(BaseModel):
    HP = {
        "win_len": REQUIRED,
        "n_neighbors": 20,
        "algorithm": "auto",
        "metric": "minkowski",
        "n_jobs": 4,
    }

    STATEFUL_ATTRS = BaseModel.STATEFUL_ATTRS + ("scaler",)

    def __init__(self, config):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.backend = LocalOutlierFactor(novelty=True, **{
            key: value for key, value in self.params.items() if key != "win_len"
        })

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
        return DetectResult(
            scores=-self.backend.decision_function(windows),
            start_pos=self.params["win_len"] - 1,
        )
