"""Train, score and evaluate a pipeline in a single experiment directory."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from playground.service.data import resolve_data_path
from playground.service.evaluator import compute_metrics
from playground.service.predictor import score_estimator
from playground.service.trainer import fit_estimator


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


@contextmanager
def _run_directory(output_dir: Path, experiment_spec: dict):
    """Reserve the directory for one run, accepting a matching prewritten spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".run.lock"
    lock.touch(exist_ok=False)
    try:
        spec_path = output_dir / "spec.json"
        for entry in output_dir.iterdir():
            if entry not in (spec_path, lock):
                raise FileExistsError(f"Experiment directory already contains {entry.name}: {output_dir}")
        if spec_path.exists():
            if json.loads(spec_path.read_text(encoding="utf-8")) != experiment_spec:
                raise ValueError(f"Existing spec.json differs from the requested experiment: {spec_path}")
        else:
            _write_json(spec_path, experiment_spec)
        yield
    finally:
        lock.unlink()


def run(experiment_spec: dict, output_dir: str | Path) -> dict:
    """Save one spec, fitted pipeline, score series, metric result and final manifest."""
    if experiment_spec["task"] != "anomaly_detection":
        raise NotImplementedError(
            "This service currently supports anomaly_detection only"
        )
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

    output_dir = Path(output_dir).expanduser().resolve()
    with _run_directory(output_dir, experiment_spec):
        estimator = fit_estimator(experiment_spec)
        estimator.save(output_dir / "model").close()

        scores = score_estimator(experiment_spec, estimator)
        scores.to_csv(output_dir / "scores.csv", index_label="timestamp", mode="x")

        metrics = compute_metrics(experiment_spec, scores)
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "manifest.json", {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "spec": "spec.json",
            "artifacts": {"model": "model.zip", "scores": "scores.csv", "metrics": "metrics.json"},
        })

    return {
        "spec_path": str(output_dir / "spec.json"),
        "model_path": str(output_dir / "model.zip"),
        "scores_path": str(output_dir / "scores.csv"),
        "metrics_path": str(output_dir / "metrics.json"),
        "metrics": metrics,
    }
