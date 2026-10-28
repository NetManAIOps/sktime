"""End-to-end experiment orchestration, reusing the individual services."""

import json
from datetime import datetime, timezone
from pathlib import Path

from playground.service.data import resolve_data_path
from playground.service.evaluator import evaluate
from playground.service.predictor import predict
from playground.service.trainer import train


def run(experiment_spec: dict, output_dir: str | Path) -> dict:
    """Train, save, reload, score and evaluate one anomaly-detection experiment.

    Layout: model/{model.zip,manifest.json}, prediction/{scores.csv,manifest.json},
    evaluation/{metrics.json,manifest.json}, plus this run's manifest.json.
    Explicit train/test CSV paths are required until the registered-dataset
    split protocol is defined. Failed runs keep completed stages for inspection.
    """
    if experiment_spec["task"] != "anomaly_detection":
        raise NotImplementedError(
            "This service currently supports anomaly_detection only"
        )
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    data = experiment_spec["data"]
    if "dataset_id" in data:
        raise ValueError(
            "run requires explicit train/test paths: registered dataset IDs "
            "do not yet provide an independent train/test split"
        )
    train_path = resolve_data_path(data, "train")
    test_path = resolve_data_path(data, "test")
    if train_path == test_path:
        raise ValueError("run requires separate train and test files")
    if not experiment_spec.get("metrics"):
        raise ValueError("experiment_spec.metrics must select at least one metric")

    model_dir = output_dir / "model"
    prediction_dir = output_dir / "prediction"
    evaluation_dir = output_dir / "evaluation"
    output_dir.mkdir(parents=True)
    train(experiment_spec, model_dir)
    predict(experiment_spec, model_dir, prediction_dir)
    results = evaluate(experiment_spec, prediction_dir, evaluation_dir)
    summary = {
        "model_dir": str(model_dir),
        "prediction_dir": str(prediction_dir),
        "evaluation_dir": str(evaluation_dir),
        "metrics": results,
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_spec": experiment_spec,
        **summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
