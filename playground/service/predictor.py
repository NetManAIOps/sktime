"""Inference using a saved model or complete preprocessing/detector pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sktime.base import load

from playground.service.data import load_test_data


def load_model(model_dir: str | Path):
    """Restore model.zip, including fitted preprocessing; only load trusted files."""
    model_dir = Path(model_dir).expanduser().resolve()
    return load(model_dir / "model.zip")


def predict(
    experiment_spec: dict, model_dir: str | Path, output_dir: str | Path
) -> pd.Series:
    """Load a fitted model, score test data and save scores.csv plus a manifest.

    Return a score Series indexed like the test input. No fit, thresholding,
    label-based decisions or evaluation occurs here. init_params/fit_params
    in experiment_spec do not replace parameters on the saved model.
    Scores are higher for more anomalous points. Existing warm-up zeros from
    detectors are kept as-is, without automatic trimming or normalization.
    """
    if experiment_spec["task"] != "anomaly_detection":
        raise NotImplementedError(
            "This service currently supports anomaly_detection only"
        )
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    estimator = load_model(model_dir)
    X = load_test_data(experiment_spec["data"])
    values = np.asarray(estimator.transform_scores(X), dtype=float)
    if values.shape != (len(X),) or not np.isfinite(values).all():
        raise ValueError("Detector scores must be finite and have shape (n_test,)")
    scores = pd.Series(values, index=X.index, name="score")

    output_dir.mkdir(parents=True)
    scores.to_csv(output_dir / "scores.csv", index_label="timestamp")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_spec": experiment_spec,
        "model_dir": str(Path(model_dir).expanduser().resolve()),
        "artifacts": {"scores": "scores.csv"},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return scores
