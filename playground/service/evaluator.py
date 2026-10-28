"""Independent evaluation of saved scores against test ground truth."""

import json
from datetime import datetime, timezone
from pathlib import Path

from sktime.performance_metrics.detection import evaluate_anomaly_scores

from playground.service.data import load_test_labels, read_csv


def evaluate(
    experiment_spec: dict, prediction_dir: str | Path, output_dir: str | Path
) -> dict[str, float]:
    """Evaluate metrics requested in experiment_spec and save metrics.json.

    Example: metrics={"VUS-PR": {"sliding_window": 100}, "Point-F1": {}}.
    Ground truth comes from data.test's label column (or the registered CSV).
    Test labels and saved score indexes must match in the same order. No model
    is loaded or fitted; no scores are recomputed. All saved points, including
    detector warm-up zeros, participate in evaluation.
    """
    if experiment_spec["task"] != "anomaly_detection":
        raise NotImplementedError(
            "This service currently supports anomaly_detection only"
        )
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    prediction_dir = Path(prediction_dir).expanduser().resolve()
    frame = read_csv(prediction_dir / "scores.csv", float_precision="round_trip")
    labels = load_test_labels(experiment_spec["data"])
    if not frame.index.equals(labels.index):
        raise ValueError(
            "Test labels and saved scores must have identical indexes/order"
        )
    results = evaluate_anomaly_scores(
        labels.to_numpy(), frame["score"].to_numpy(), experiment_spec["metrics"]
    )

    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_spec": experiment_spec,
        "prediction_dir": str(prediction_dir),
        "artifacts": {"metrics": "metrics.json"},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results
