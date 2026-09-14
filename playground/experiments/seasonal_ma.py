"""From-scratch plugin: seasonal moving-average forecaster in pure numpy.

Demonstrates the minimal forecasting contract: ``fit(y)`` then
``predict(steps)`` returning exactly ``steps`` values.
"""

TASK = "forecasting"
NAME = "seasonal-moving-average"
PARAMS = {"window": 12, "seasonal_period": 12}

import numpy as np


class Algorithm:
    def __init__(self, window=12, seasonal_period=12):
        self.window = int(window)
        self.seasonal_period = int(seasonal_period)

    def fit(self, y):
        self._values = np.asarray(y, dtype=float)
        return self

    def predict(self, steps):
        values = self._values
        out = []
        for h in range(steps):
            season = values[-self.seasonal_period + (h % self.seasonal_period)]
            trend = values[-self.window:].mean()
            out.append(0.5 * season + 0.5 * trend)
            values = np.append(values, out[-1])
        return np.asarray(out)
