"""Unified metric registry for the TSBox Playground and the LabTS API.

One registry drives three consumers:

- ``catalog.build_catalog()`` / ``labts ls metrics`` — discovery metadata.
- ``run --metric <id>`` (repeatable) — extra metrics computed on top of the
  per-task default set (the default set is unchanged when no ``--metric`` is
  passed, keeping every existing result byte-compatible).
- ``labts evaluate --from run.json --metric <id>`` — re-scores a saved run
  without re-fitting, using the persisted evaluation payload (anomaly runs
  store continuous ``scores``; forecasting runs store actuals/predictions).

Registry entry metadata (JSON-safe):

- ``id``        snake_case identifier used on the CLI (``--metric pa_f1``).
- ``name``      display name used as the key in the result ``metrics`` dict.
- ``task``      playground task the metric applies to.
- ``requires``  subset of ``scores | labels | values | predictions``:
    - ``values``      ground-truth numeric values (held-out series/targets)
    - ``predictions`` model predictions (point forecasts, class/cluster ids,
      anomaly point flags, or reconstructed values for reconstruction metrics)
    - ``labels``      ground-truth labels (class/cluster/anomaly)
    - ``scores``      continuous anomaly scores
- ``context``   extra runner-provided inputs beyond ``requires``
  (``y_train`` for scaled forecasting errors, ``y_pred_benchmark`` for
  relative forecasting errors; both are always present in forecasting runs).
- ``direction`` ``higher`` or ``lower`` (better).
- ``default``   part of the task's default metric set (matches the historic
  playground output exactly).
- ``source``    ``sktime`` | ``sklearn`` | ``devad`` | ``playground``.
- ``available`` False when no runner can currently produce the required
  context (requesting it raises a clear error instead of failing obscurely).
"""

from __future__ import annotations

import numpy as np


class MetricError(ValueError):
    """Metric cannot be computed from the available run context."""


# ---------------------------------------------------------------------------
# compute helpers (lazy imports keep `catalog` fast)
# ---------------------------------------------------------------------------


def _values(ctx):
    values = ctx.get("values")
    if values is None:
        raise MetricError("metric needs ground-truth values, none in this run")
    return np.asarray(values, dtype=float).ravel()


def _predictions(ctx, dtype=float):
    predictions = ctx.get("predictions")
    if predictions is None:
        raise MetricError("metric needs model predictions, none in this run")
    return np.asarray(predictions, dtype=dtype).ravel()


def _labels(ctx):
    labels = ctx.get("labels")
    if labels is None:
        raise MetricError("metric needs ground-truth labels, none in this run")
    return np.asarray(labels).ravel()


def _scores(ctx):
    scores = ctx.get("scores")
    if scores is None:
        raise MetricError(
            "metric needs continuous anomaly scores; re-run with a detector "
            "that exposes scores (all playground anomaly runs save them under "
            "the `evaluation` payload)"
        )
    return np.asarray(scores, dtype=float).ravel()


def _forecasting(fn_name):
    def compute(ctx):
        from sktime.performance_metrics import forecasting as skm

        fn = getattr(skm, fn_name)
        kwargs = {}
        if "scaled" in fn_name:
            y_train = ctx.get("y_train")
            if y_train is None:
                raise MetricError(f"{fn_name} requires y_train context")
            kwargs["y_train"] = np.asarray(y_train, dtype=float).ravel()
        if fn_name in _RELATIVE_FNS:
            benchmark = ctx.get("y_pred_benchmark")
            if benchmark is None:
                raise MetricError(f"{fn_name} requires y_pred_benchmark context")
            kwargs["y_pred_benchmark"] = np.asarray(benchmark, dtype=float).ravel()
        return float(fn(_values(ctx), _predictions(ctx), **kwargs))

    return compute


_RELATIVE_FNS = {
    "mean_relative_absolute_error",
    "median_relative_absolute_error",
    "geometric_mean_relative_absolute_error",
    "geometric_mean_relative_squared_error",
    "relative_loss",
}


def _forecasting_variant(fn_name, **fixed):
    base = _forecasting(fn_name)

    def compute(ctx):
        from sktime.performance_metrics import forecasting as skm

        fn = getattr(skm, fn_name)
        return float(fn(_values(ctx), _predictions(ctx), **fixed))

    return compute


def _sklearn_classification(fn_name, **fixed):
    def compute(ctx):
        from sklearn import metrics as skm

        fn = getattr(skm, fn_name)
        return float(fn(_labels(ctx), _predictions(ctx), **fixed))

    return compute


def _sklearn_regression(fn_name, **fixed):
    def compute(ctx):
        from sklearn import metrics as skm

        fn = getattr(skm, fn_name)
        return float(fn(_values(ctx), _predictions(ctx), **fixed))

    return compute


def _sklearn_clustering(fn_name):
    def compute(ctx):
        from sklearn import metrics as skm

        fn = getattr(skm, fn_name)
        return float(fn(_labels(ctx), _predictions(ctx)))

    return compute


def _detection_points(metric_cls, **fixed):
    """Wrap sktime.performance_metrics.detection metrics (point-iloc format)."""

    def compute(ctx):
        import pandas as pd
        from sktime.performance_metrics import detection as skd

        y_true = _labels(ctx).astype(int)
        y_pred = _predictions(ctx, dtype=int)
        to_points = lambda y: pd.DataFrame({"ilocs": np.where(y != 0)[0]})  # noqa: E731
        metric = getattr(skd, metric_cls)(**fixed)
        return float(metric(to_points(y_true), to_points(y_pred)))

    return compute


def _devad(metric_name):
    """Wrap a DevAD ``base_metricor`` score-based metric."""

    def compute(ctx):
        from sktime.libs.devad.utils.evaluate import base_metricor

        return float(base_metricor()(metric_name, label=_labels(ctx).astype(int), score=_scores(ctx)))

    return compute


def _point_prf(kind):
    def compute(ctx):
        y_true = _labels(ctx).astype(int)
        y_pred = _predictions(ctx, dtype=int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if kind == "precision":
            return precision
        if kind == "recall":
            return recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return compute


def _not_available(reason):
    def compute(ctx):
        raise MetricError(reason)

    return compute


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_F = "forecasting"
_C = "classification"
_R = "regression"
_K = "clustering"
_A = "anomaly_detection"

_VP = ["values", "predictions"]
_LP = ["labels", "predictions"]
_LS = ["labels", "scores"]

METRIC_REGISTRY: list[dict] = [
    # --- forecasting (sktime.performance_metrics.forecasting) -------------
    {"id": "mae", "name": "MAE", "task": _F, "requires": _VP, "direction": "lower", "default": True, "source": "sktime"},
    {"id": "mse", "name": "MSE", "task": _F, "requires": _VP, "direction": "lower", "default": True, "source": "sktime"},
    {"id": "rmse", "name": "RMSE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "mape", "name": "MAPE", "task": _F, "requires": _VP, "direction": "lower", "default": True, "source": "sktime"},
    {"id": "smape", "name": "sMAPE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "medae", "name": "MedAE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "medse", "name": "MedSE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "medape", "name": "MedAPE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "mspe", "name": "MSPE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "medspe", "name": "MedSPE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "mase", "name": "MASE", "task": _F, "requires": _VP, "context": ["y_train"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "medase", "name": "MedASE", "task": _F, "requires": _VP, "context": ["y_train"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "msse", "name": "MSSE", "task": _F, "requires": _VP, "context": ["y_train"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "medsse", "name": "MedSSE", "task": _F, "requires": _VP, "context": ["y_train"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "gmae", "name": "GMAE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "gmse", "name": "GMSE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "mrelae", "name": "MRelAE", "task": _F, "requires": _VP, "context": ["y_pred_benchmark"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "medrelae", "name": "MedRelAE", "task": _F, "requires": _VP, "context": ["y_pred_benchmark"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "gmrelae", "name": "GMRelAE", "task": _F, "requires": _VP, "context": ["y_pred_benchmark"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "gmrelse", "name": "GMRelSE", "task": _F, "requires": _VP, "context": ["y_pred_benchmark"], "direction": "lower", "default": False, "source": "sktime"},
    {"id": "masyme", "name": "MAsymE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "mlinex", "name": "Linex", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "maape", "name": "MAAPE", "task": _F, "requires": _VP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "rell", "name": "Relative Loss", "task": _F, "requires": _VP, "context": ["y_pred_benchmark"], "direction": "lower", "default": False, "source": "sktime"},
    # --- classification (sklearn) ------------------------------------------
    {"id": "accuracy", "name": "Accuracy", "task": _C, "requires": _LP, "direction": "higher", "default": True, "source": "sklearn"},
    {"id": "balanced_accuracy", "name": "Balanced Accuracy", "task": _C, "requires": _LP, "direction": "higher", "default": False, "source": "sklearn"},
    {"id": "macro_f1", "name": "Macro F1", "task": _C, "requires": _LP, "direction": "higher", "default": True, "source": "sklearn"},
    {"id": "micro_f1", "name": "Micro F1", "task": _C, "requires": _LP, "direction": "higher", "default": False, "source": "sklearn"},
    {"id": "weighted_f1", "name": "Weighted F1", "task": _C, "requires": _LP, "direction": "higher", "default": False, "source": "sklearn"},
    # --- regression (sklearn) ----------------------------------------------
    {"id": "mae", "name": "MAE", "task": _R, "requires": _VP, "direction": "lower", "default": True, "source": "sklearn"},
    {"id": "mse", "name": "MSE", "task": _R, "requires": _VP, "direction": "lower", "default": False, "source": "sklearn"},
    {"id": "rmse", "name": "RMSE", "task": _R, "requires": _VP, "direction": "lower", "default": True, "source": "sklearn"},
    {"id": "medae", "name": "MedAE", "task": _R, "requires": _VP, "direction": "lower", "default": False, "source": "sklearn"},
    {"id": "mape", "name": "MAPE", "task": _R, "requires": _VP, "direction": "lower", "default": False, "source": "sklearn"},
    {"id": "r2", "name": "R²", "task": _R, "requires": _VP, "direction": "higher", "default": True, "source": "sklearn"},
    # --- clustering (sklearn) -----------------------------------------------
    {"id": "ari", "name": "ARI", "task": _K, "requires": _LP, "direction": "higher", "default": True, "source": "sklearn"},
    {"id": "nmi", "name": "NMI", "task": _K, "requires": _LP, "direction": "higher", "default": True, "source": "sklearn"},
    {"id": "homogeneity", "name": "Homogeneity", "task": _K, "requires": _LP, "direction": "higher", "default": False, "source": "sklearn"},
    {"id": "completeness", "name": "Completeness", "task": _K, "requires": _LP, "direction": "higher", "default": False, "source": "sklearn"},
    {"id": "v_measure", "name": "V-Measure", "task": _K, "requires": _LP, "direction": "higher", "default": False, "source": "sklearn"},
    # --- anomaly detection: point P/R/F1 (playground) -----------------------
    {"id": "precision", "name": "Precision", "task": _A, "requires": _LP, "direction": "higher", "default": True, "source": "playground"},
    {"id": "recall", "name": "Recall", "task": _A, "requires": _LP, "direction": "higher", "default": True, "source": "playground"},
    {"id": "f1", "name": "F1", "task": _A, "requires": _LP, "direction": "higher", "default": True, "source": "playground"},
    # --- anomaly detection: sktime.performance_metrics.detection ------------
    {"id": "windowed_f1", "name": "Windowed F1", "task": _A, "requires": _LP, "direction": "higher", "default": False, "source": "sktime"},
    {"id": "rand_index", "name": "Rand Index", "task": _A, "requires": _LP, "direction": "higher", "default": False, "source": "sktime"},
    {"id": "ts_auprc", "name": "TS-AUPRC", "task": _A, "requires": _LP, "direction": "higher", "default": False, "source": "sktime"},
    {"id": "directed_chamfer", "name": "Directed Chamfer", "task": _A, "requires": _LP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "directed_hausdorff", "name": "Directed Hausdorff", "task": _A, "requires": _LP, "direction": "lower", "default": False, "source": "sktime"},
    {"id": "detection_count", "name": "Detection Count", "task": _A, "requires": _LP, "direction": "lower", "default": False, "source": "sktime"},
    # --- anomaly detection: DevAD metricor (score-based) --------------------
    {"id": "auc_roc", "name": "AUC-ROC", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    {"id": "ap", "name": "AP", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    {"id": "point_f1", "name": "Point-F1", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    {"id": "pa_f1", "name": "PA-F1", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    {"id": "affiliation_f1", "name": "Affiliation-F1", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    {"id": "delay_f1", "name": "Delay-F1", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    {"id": "vus_pr", "name": "VUS-PR", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    {"id": "vus_roc", "name": "VUS-ROC", "task": _A, "requires": _LS, "direction": "higher", "default": False, "source": "devad"},
    # --- anomaly detection: DevAD reconstruction metrics (declared) ---------
    # These compare the input series against the detector's reconstructed
    # output; no playground runner produces reconstructions yet, so they are
    # advertised as unavailable and raise a clear error when requested.
    {
        "id": "predict_error", "name": "Predict-Error", "task": _A,
        "requires": ["values", "predictions"], "direction": "lower",
        "default": False, "source": "devad", "available": False,
        "unavailable_reason": "requires reconstructed values (detector output); no playground runner produces reconstructions yet",
    },
    {
        "id": "output_mae", "name": "Output-MAE", "task": _A,
        "requires": ["values", "predictions"], "direction": "lower",
        "default": False, "source": "devad", "available": False,
        "unavailable_reason": "requires reconstructed values (detector output); no playground runner produces reconstructions yet",
    },
]

_COMPUTERS = {
    # forecasting
    "mae": _forecasting("mean_absolute_error"),
    "mse": _forecasting("mean_squared_error"),
    "rmse": _forecasting_variant("mean_squared_error", square_root=True),
    "mape": _forecasting("mean_absolute_percentage_error"),
    "smape": _forecasting_variant("mean_absolute_percentage_error", symmetric=True),
    "medae": _forecasting("median_absolute_error"),
    "medse": _forecasting("median_squared_error"),
    "medape": _forecasting("median_absolute_percentage_error"),
    "mspe": _forecasting("mean_squared_percentage_error"),
    "medspe": _forecasting("median_squared_percentage_error"),
    "mase": _forecasting("mean_absolute_scaled_error"),
    "medase": _forecasting("median_absolute_scaled_error"),
    "msse": _forecasting("mean_squared_scaled_error"),
    "medsse": _forecasting("median_squared_scaled_error"),
    "gmae": _forecasting("geometric_mean_absolute_error"),
    "gmse": _forecasting("geometric_mean_squared_error"),
    "mrelae": _forecasting("mean_relative_absolute_error"),
    "medrelae": _forecasting("median_relative_absolute_error"),
    "gmrelae": _forecasting("geometric_mean_relative_absolute_error"),
    "gmrelse": _forecasting("geometric_mean_relative_squared_error"),
    "masyme": _forecasting("mean_asymmetric_error"),
    "mlinex": _forecasting("mean_linex_error"),
    "maape": _forecasting("mean_arctangent_absolute_percentage_error"),
    "rell": _forecasting("relative_loss"),
    # classification
    "accuracy": _sklearn_classification("accuracy_score"),
    "balanced_accuracy": _sklearn_classification("balanced_accuracy_score"),
    "macro_f1": _sklearn_classification("f1_score", average="macro"),
    "micro_f1": _sklearn_classification("f1_score", average="micro"),
    "weighted_f1": _sklearn_classification("f1_score", average="weighted"),
    # regression
    "r2": _sklearn_regression("r2_score"),
    # clustering
    "ari": _sklearn_clustering("adjusted_rand_score"),
    "nmi": _sklearn_clustering("normalized_mutual_info_score"),
    "homogeneity": _sklearn_clustering("homogeneity_score"),
    "completeness": _sklearn_clustering("completeness_score"),
    "v_measure": _sklearn_clustering("v_measure_score"),
    # anomaly: point P/R/F1 + detection metrics + DevAD score metrics
    "precision": _point_prf("precision"),
    "recall": _point_prf("recall"),
    "f1": _point_prf("f1"),
    "windowed_f1": _detection_points("WindowedF1Score"),
    "rand_index": _detection_points("RandIndex"),
    "ts_auprc": _detection_points("TimeSeriesAUPRC"),
    "directed_chamfer": _detection_points("DirectedChamfer"),
    "directed_hausdorff": _detection_points("DirectedHausdorff"),
    "detection_count": _detection_points("DetectionCount"),
    "auc_roc": _devad("AUC-ROC"),
    "ap": _devad("AP"),
    "point_f1": _devad("Point-F1"),
    "pa_f1": _devad("PA-F1"),
    "affiliation_f1": _devad("Affiliation-F1"),
    "delay_f1": _devad("Delay-F1"),
    "vus_pr": _devad("VUS-PR"),
    "vus_roc": _devad("VUS-ROC"),
    "predict_error": _not_available(
        "Predict-Error requires reconstructed values; no playground runner "
        "produces reconstructions yet"
    ),
    "output_mae": _not_available(
        "Output-MAE requires reconstructed values; no playground runner "
        "produces reconstructions yet"
    ),
}

# regression re-uses the forecasting point-error computers (same 1-D arrays)
for _rid in ("mae", "mse", "rmse", "medae", "mape"):
    _COMPUTERS[("regression", _rid)] = _COMPUTERS[_rid]


def _key(entry: dict, task: str | None = None):
    return entry["id"]


def all_metrics() -> list[dict]:
    """All registry entries (JSON-safe metadata, computers stripped)."""
    return [dict(entry) for entry in METRIC_REGISTRY]


def metrics_for_task(task: str) -> list[dict]:
    return [dict(entry) for entry in METRIC_REGISTRY if entry["task"] == task]


def default_metric_ids(task: str) -> list[str]:
    return [entry["id"] for entry in METRIC_REGISTRY if entry["task"] == task and entry.get("default")]


def get_metric(metric_id: str, task: str | None = None) -> dict | None:
    """Look up a metric by id; disambiguate cross-task id collisions by task."""
    matches = [entry for entry in METRIC_REGISTRY if entry["id"] == metric_id]
    if task is not None:
        for entry in matches:
            if entry["task"] == task:
                return dict(entry)
        return None
    return dict(matches[0]) if matches else None


def resolve_metrics(metric_ids: list[str] | None, task: str) -> list[dict]:
    """Validate a list of requested metric ids against the registry."""
    resolved = []
    for metric_id in metric_ids or []:
        entry = get_metric(metric_id, task)
        if entry is None:
            valid = sorted({e["id"] for e in METRIC_REGISTRY if e["task"] == task})
            raise MetricError(
                f"Unknown metric `{metric_id}` for task `{task}`. "
                f"Valid ids: {', '.join(valid)} (see `labts ls metrics --task {task}`)."
            )
        resolved.append(entry)
    return resolved


def compute_metric(entry: dict, task: str, context: dict) -> float:
    """Compute one registry metric from a run context.

    context keys: ``values``, ``predictions``, ``labels``, ``scores`` plus
    optional ``y_train`` / ``y_pred_benchmark`` (forecasting).
    """
    computer = _COMPUTERS.get((task, entry["id"])) or _COMPUTERS.get(entry["id"])
    if computer is None:
        raise MetricError(f"No computer registered for metric `{entry['id']}`.")
    try:
        return float(computer(context))
    except MetricError:
        raise
    except Exception as exc:
        raise MetricError(
            f"Metric `{entry['id']}` failed: {type(exc).__name__}: {exc}"
        ) from exc
