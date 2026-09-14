"""From-scratch plugin: direct pyod usage, no sktime knowledge required.

Demonstrates the minimal anomaly_detection contract: a class with
``fit_predict(X)`` returning a dense 0/1 label per point. Third-party
libraries (here pyod) are simply imported and used.
"""

TASK = "anomaly_detection"
NAME = "ecod-direct"
PARAMS = {"contamination": 0.02}

import numpy as np


class Algorithm:
    def __init__(self, contamination=0.02):
        self.contamination = contamination

    def fit_predict(self, X):
        from pyod.models.ecod import ECOD

        model = ECOD(contamination=self.contamination)
        return model.fit_predict(np.asarray(X, dtype=float))
