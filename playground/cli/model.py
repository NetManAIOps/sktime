"""Thin command wrappers for model lifecycle and experiment services."""

import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import typer

from playground.cli import report_error


def train(
    ctx: typer.Context,
    spec: Path = typer.Option(..., "--spec", help="Experiment JSON file."),
    output_dir: Path = typer.Option(..., "--output-dir", help="New model directory."),
) -> None:
    """Train and save a model or pipeline."""
    _execute(ctx, "train", spec, output_dir)


def predict(
    ctx: typer.Context,
    spec: Path = typer.Option(..., "--spec", help="Experiment JSON file."),
    model_dir: Path = typer.Option(
        ..., "--model-dir", help="Trusted saved model directory."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="New prediction directory."
    ),
) -> None:
    """Load a trusted model and save anomaly scores without refitting."""
    _execute(ctx, "predict", spec, output_dir, model_dir=model_dir)


def evaluate(
    ctx: typer.Context,
    spec: Path = typer.Option(..., "--spec", help="Experiment JSON file with metrics."),
    prediction_dir: Path = typer.Option(
        ..., "--prediction-dir", help="Directory containing saved scores.csv."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="New evaluation directory."
    ),
) -> None:
    """Evaluate saved scores against test labels without rerunning the model."""
    _execute(ctx, "evaluate", spec, output_dir, prediction_dir=prediction_dir)


def run(
    ctx: typer.Context,
    spec: Path = typer.Option(
        ..., "--spec", help="Experiment JSON with train/test/metrics."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="New experiment directory."
    ),
) -> None:
    """Train, predict and evaluate one experiment using test ground truth."""
    _execute(ctx, "run", spec, output_dir)


def _execute(
    ctx: typer.Context, action: str, spec_path: Path, output_dir: Path, **kwargs
) -> None:
    """Call a service and report artifact paths instead of fitted objects or scores."""
    try:
        with redirect_stdout(sys.stderr), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using `json`-module for json-handling\..*",
                category=UserWarning,
                module=r"gluonts\.json",
            )
            from playground import service

            experiment_spec = json.loads(
                spec_path.expanduser().read_text(encoding="utf-8")
            )

            # service
            result = getattr(service, action)(
                experiment_spec, output_dir=output_dir, **kwargs
            )

        output_dir = output_dir.expanduser().resolve()
        if action == "train":
            payload = {
                "model_dir": str(output_dir),
                "model_path": str(output_dir / "model.zip"),
            }
        elif action == "predict":
            payload = {
                "prediction_dir": str(output_dir),
                "scores_path": str(output_dir / "scores.csv"),
                "n_scores": len(result),
            }
        elif action == "evaluate":
            payload = {
                "evaluation_dir": str(output_dir),
                "metrics_path": str(output_dir / "metrics.json"),
                "metrics": result,
            }
        else:
            payload = {"run_dir": str(output_dir), **result}
        payload["manifest_path"] = str(output_dir / "manifest.json")
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    except Exception as exc:
        report_error(ctx, exc)
    typer.echo(encoded)
