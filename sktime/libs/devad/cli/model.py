from __future__ import annotations

import json
from pathlib import Path

import typer

from ..services.model_service import (
    get_model_info,
    list_families,
    model_detect,
    model_evaluate,
    train_model,
)
from .common import write, fail


model_app = typer.Typer(no_args_is_help=True)


@model_app.command("list")
def list_models() -> None:
    try:
        families = list_families()
    except Exception as exc:
        fail(exc)
    write({"Available model families": families})


@model_app.command("info")
def model_info(family: str = typer.Argument(...)) -> None:
    try:
        info = get_model_info(family)
    except Exception as exc:
        fail(exc)
    write(info)


@model_app.command("train")
def train(
    model_root: Path = typer.Option(..., "--model-root"),
    model_id: str = typer.Option(..., "--model-id"),
    family: str = typer.Option(..., "--family"),
    config_path: Path = typer.Option(..., "--config"),
    train_data: Path = typer.Option(..., "--train-data"),
    val_data: Path | None = typer.Option(None, "--val-data"),
    seed: int = typer.Option(2026, "--seed"),
    device: str = typer.Option("mps", "--device"),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress",
        help="Show training progress on stderr; the training log is always saved.",
    ),
) -> None:
    try:
        result = train_model(
            model_root=model_root,
            model_id=model_id,
            family=family,
            config_path=config_path,
            x_train=train_data,
            x_val=val_data,
            seed=seed,
            device=device,
            show_progress=show_progress,
        )
    except Exception as exc:
        fail(exc)

    write(result)


@model_app.command("detect")
def detect(
    model_dir: Path = typer.Option(..., "--model-dir"),
    input_data: Path = typer.Option(..., "--input-data"),
    result_dir: Path = typer.Option(..., "--result-dir"),
    device: str = typer.Option("mps", "--device"),
) -> None:
    try:
        result = model_detect(
            model_dir=model_dir,
            device=device,
            x_test=input_data,
            result_dir=result_dir,
        )
    except Exception as exc:
        fail(exc)

    write(result)


@model_app.command("evaluate")
def evaluate(
    model_dir: Path = typer.Option(..., "--model-dir"),
    input_data: Path = typer.Option(..., "--input-data"),
    label_data: Path = typer.Option(..., "--label-data"),
    result_dir: Path = typer.Option(..., "--result-dir"),
    metrics: list[str] = typer.Option(..., "--metric"),
    metric_params: str | None = typer.Option(None, "--metric-params"),
    device: str = typer.Option("mps", "--device"),
) -> None:
    try:
        params = json.loads(metric_params) if metric_params is not None else None
        result = model_evaluate(
            model_dir=model_dir,
            device=device,
            x_test=input_data,
            label=label_data,
            result_dir=result_dir,
            metrics=metrics,
            metric_params=params,
        )
    except Exception as exc:
        fail(exc)

    write(result)
