"""Public evaluation of continuous anomaly scores against binary point labels."""

import numpy as np

from sktime.performance_metrics.detection._anomaly_score import _AnomalyMetricCalculator


def evaluate_anomaly_scores(y_true, y_score, metric_spec: dict) -> dict[str, float]:
    """Compute the score-based TSAD metrics selected by a name/parameters mapping.

    Supported names: AUC-ROC, AP, Point-F1, PA-F1, Affiliation-F1, Delay-F1,
    VUS-PR and VUS-ROC. For example::

        {"VUS-PR": {"sliding_window": 100}, "Point-F1": {}}

    Larger scores must indicate more anomalous points. F1 metrics search for
    their best threshold on the evaluation labels; they do not calibrate a
    deployable threshold. The original DevAD formulas and defaults are kept:
    VUS uses maximum buffer 100 and 250 score thresholds, adjusted/affiliation
    F1 scans at most 600 thresholds, and Delay-F1 defaults to delay_k=7.
    No automatic scaling, inversion, cropping or point adjustment is performed
    before dispatch. Undefined metrics raise rather than silently becoming zero.
    This array/score API is separate from the sparse-event BaseDetectionMetric.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size < 2:
        raise ValueError(
            "Labels and scores must be equal-length 1D arrays with at least two points"
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Ground-truth labels must be binary 0/1")
    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores must be finite")
    if not isinstance(metric_spec, dict) or not metric_spec:
        raise ValueError(
            "metrics must be a nonempty mapping of metric names to parameter dictionaries"
        )

    calculator = _AnomalyMetricCalculator()
    labels = labels.astype(int)
    results = {}
    for name, params in metric_spec.items():
        entry = calculator.METRIC_MAP.get(name)
        if entry is None or entry["requires"] != ("label", "score"):
            raise ValueError(f"Unsupported anomaly score metric: {name}")
        if name in {"AUC-ROC", "VUS-PR", "VUS-ROC"} and np.unique(labels).size != 2:
            raise ValueError(f"{name} requires both normal and anomalous points")
        value = calculator(name, label=labels, score=scores, **params)
        if not np.isfinite(value):
            raise ValueError(f"{name} is undefined for these labels/scores")
        results[name] = float(value)
    return results
