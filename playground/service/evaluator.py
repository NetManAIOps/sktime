"""Independent evaluation of saved scores against test ground truth."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sktime.performance_metrics.detection import evaluate_anomaly_scores

from playground.service.data import load_test_labels, read_csv


def evaluate(
    experiment_spec: dict, scores_path: str | Path, output_dir: str | Path
) -> dict[str, float]:
    """Evaluate metrics requested in experiment_spec and save metrics.json.

    Example: metrics={"VUS-PR": {"sliding_window": 100}, "Point-F1": {}}.
    Ground truth comes from data.test's label column (or the registered CSV).
    Test labels and saved score indexes must match in the same order. All saved points, including
    detector warm-up zeros, participate in evaluation. Repeated evaluations
    replace metrics.json and the evaluation manifest in output_dir.
    """
    if experiment_spec["task"] != "anomaly_detection":
        raise NotImplementedError(
            "This service currently supports anomaly_detection only"
        )
    output_dir = Path(output_dir).expanduser().resolve()
    scores_path = Path(scores_path).expanduser().resolve()
    frame = read_csv(scores_path, float_precision="round_trip")
    results = compute_metrics(experiment_spec, frame["score"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_spec": experiment_spec,
        "scores_path": str(scores_path),
        "artifacts": {"metrics": "metrics.json"},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


def compute_metrics(experiment_spec: dict, scores: pd.Series) -> dict[str, float]:
    """Compute the requested metrics for scores aligned with the test labels."""
    labels = load_test_labels(experiment_spec["data"])
    if not scores.index.equals(labels.index):
        raise ValueError(
            "Test labels and saved scores must have identical indexes/order"
        )
    return evaluate_anomaly_scores(
        labels.to_numpy(), scores.to_numpy(), experiment_spec["metrics"]
    )
