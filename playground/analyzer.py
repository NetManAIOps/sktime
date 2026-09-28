"""Parameter-estimation backend for `labts analyze`.

Fits a sktime ``param_est`` estimator (seasonality, stationarity, lag order)
on a catalog series and returns the estimated parameters as JSON. Unlike
``run`` there is no evaluation stage: the fitted parameters *are* the result.
"""

from __future__ import annotations

import math
import time

import numpy as np

from catalog import get_dataset, import_estimator_class
from runners import PlaygroundError

ANALYZERS: list[dict] = [
    {
        "id": "seasonality-acf",
        "name": "SeasonalityACF",
        "module": "sktime.param_est.seasonality.SeasonalityACF",
        "estimates": ["sp", "sp_significant"],
        "params": {},
        "enabled": True,
        "default": True,
    },
    {
        "id": "seasonality-periodogram",
        "name": "SeasonalityPeriodogram",
        "module": "sktime.param_est.seasonality.SeasonalityPeriodogram",
        "estimates": ["sp", "sp_significant"],
        "params": {},
        "enabled": True,
    },
    {
        "id": "stationarity-adf",
        "name": "StationarityADF",
        "module": "sktime.param_est.stationarity.StationarityADF",
        "estimates": ["stationary", "pvalue", "test_statistic", "used_lag"],
        "params": {},
        "enabled": True,
    },
    {
        "id": "stationarity-kpss",
        "name": "StationarityKPSS",
        "module": "sktime.param_est.stationarity.StationarityKPSS",
        "estimates": ["stationary", "pvalue", "test_statistic", "lags"],
        "params": {},
        "enabled": True,
    },
    {
        "id": "ar-lag-order",
        "name": "ARLagOrderSelector",
        "module": "sktime.param_est.lag.ARLagOrderSelector",
        "estimates": ["selected_lags", "ic_value"],
        "params": {"maxlag": 12},
        "enabled": True,
    },
]


def get_analyzer(analyzer_id: str | None) -> dict | None:
    if not analyzer_id:
        return ANALYZERS[0]
    return next((a for a in ANALYZERS if a["id"] == analyzer_id), None)


def _load_series(dataset: dict, log: list[str]):
    """Load a catalog series dataset (forecasting or anomaly_detection)."""
    if dataset.get("task") == "anomaly_detection":
        from runners import load_anomaly_series

        values, _labels = load_anomaly_series(dataset, {"id": "none", "name": "none"}, {}, log)
        return values
    if dataset.get("task") == "forecasting":
        from runners import _load_forecasting_series

        import pandas as pd

        return pd.Series(_load_forecasting_series(dataset, log)).dropna()
    raise PlaygroundError(
        f"Dataset `{dataset['id']}` is a {dataset.get('task')} dataset; "
        "`labts analyze` needs a series dataset (forecasting or anomaly_detection)."
    )


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def run_analysis(spec: dict) -> dict:
    """Fit a param_est estimator on a catalog series; return its estimates."""
    started = time.perf_counter()
    analyzer = get_analyzer(spec.get("analyzer_id") or spec.get("algorithm_id"))
    if analyzer is None:
        valid = ", ".join(a["id"] for a in ANALYZERS)
        raise PlaygroundError(f"Unknown analyzer `{spec.get('analyzer_id')}`. Valid ids: {valid}")
    dataset = get_dataset(spec.get("dataset_id") or "airline")
    if dataset is None or not dataset.get("enabled"):
        raise PlaygroundError(f"Dataset is not enabled: {spec.get('dataset_id')}")

    params = {**analyzer.get("params", {}), **(spec.get("params") or {})}
    klass = import_estimator_class(analyzer["module"])
    log = [
        f"Selected analyzer: {analyzer['name']}",
        f"Selected dataset: {dataset['name']}",
        f"Params: {params or 'defaults'}",
    ]
    y = _load_series(dataset, log)
    try:
        estimator = klass(**params)
        estimator.fit(y)
    except Exception as exc:
        raise PlaygroundError(
            f"Analyzer `{analyzer['name']}` failed: {type(exc).__name__}: {exc}"
        ) from exc

    fitted = estimator.get_fitted_params()
    estimates = {
        key: _jsonable(value)
        for key, value in fitted.items()
        if not key.startswith("X") and key != "X"
    }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    log.append(f"Finished in {elapsed_ms} ms")
    return {
        "status": "ok",
        "analyzer": {k: analyzer[k] for k in ("id", "name", "module")},
        "dataset": {k: dataset.get(k) for k in ("id", "name", "task")},
        "params": params,
        "series_length": int(len(y)),
        "estimates": estimates,
        "duration_ms": elapsed_ms,
        "log": log,
    }
