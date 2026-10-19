"""Windowed One-Class SVM backed by sklearn/LIBSVM.

Defaults follow EasyTSAD SubOCSVM (a13519014c0334a26b0e74480881db503e6a8e0b).
DevAD fits train only, and assigns each window's negative decision value to
its endpoint. No window subsampling or approximate solver is used.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .Base import REQUIRED, BaseModel, DetectResult
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class SubOCSVM(BaseModel):
    HP = {
        "win_len": REQUIRED,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "auto",
        "coef0": 0.0,
        "tol": 1e-3,
        "nu": 0.5,
        "shrinking": True,
        "cache_size": 200,
        "verbose": False,
        "max_iter": -1,
    }

    STATEFUL_ATTRS = BaseModel.STATEFUL_ATTRS + ("scaler",)

    def __init__(self, config):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.backend = OneClassSVM(**{
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
        # Normalize the series before windowing, not each lag independently.
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
