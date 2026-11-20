"""Inference using a saved model or complete preprocessing/detector pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sktime.base import load

from playground.service.data import load_test_data


def predict(
    experiment_spec: dict, model_path: str | Path, output_dir: str | Path
) -> pd.Series:
    """Load a fitted model, score test data and save scores.csv plus a manifest.

    Return a score Series indexed like the test input. No fit, thresholding,
    label-based decisions or evaluation occurs here. The saved model provides
    its own estimator and fit parameters.
    Scores are higher for more anomalous points. Existing warm-up zeros from
    detectors are kept as-is, without automatic trimming or normalization.
    """
    if experiment_spec["task"] != "anomaly_detection":
        raise NotImplementedError(
            "This service currently supports anomaly_detection only"
        )
    output_dir = Path(output_dir).expanduser().resolve()
    scores_path = output_dir / "scores.csv"
    if scores_path.exists():
        raise FileExistsError(f"Scores already exist: {scores_path}")

    model_path = Path(model_path).expanduser().resolve()
    estimator = load(model_path)
    scores = score_estimator(experiment_spec, estimator)

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_path, index_label="timestamp", mode="x")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_spec": experiment_spec,
        "model_path": str(model_path),
        "artifacts": {"scores": "scores.csv"},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return scores


def score_estimator(experiment_spec: dict, estimator) -> pd.Series:
    """Score the test series with a fitted detector and preserve its index."""
    X = load_test_data(experiment_spec["data"])
    values = np.asarray(estimator.transform_scores(X), dtype=float)
    if values.shape != (len(X),) or not np.isfinite(values).all():
        raise ValueError("Detector scores must be finite and have shape (n_test,)")
    return pd.Series(values, index=X.index, name="score")
