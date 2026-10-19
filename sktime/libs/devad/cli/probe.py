from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import sys

import typer

from ..services.probe_service import prepare_pseudo_anomaly, run_pseudo_anomaly
from .common import fail, write


probe_app = typer.Typer(no_args_is_help=True)
pseudo_anomaly_app = typer.Typer(no_args_is_help=True)
probe_app.add_typer(pseudo_anomaly_app, name="pseudo-anomaly")


@pseudo_anomaly_app.command("prepare")
def prepare(
    input_data: Path = typer.Option(..., "--input-data"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    anomaly_type: str = typer.Option(..., "--anomaly-type"),
    anomaly_rate: float = typer.Option(..., "--anomaly-rate"),
    length_range: tuple[int, int] = typer.Option(..., "--length-range"),
    strength_range: tuple[float, float] = typer.Option(..., "--strength-range"),
    seed: int = typer.Option(2026, "--seed"),
) -> None:
    try:
        result = prepare_pseudo_anomaly(
            input_data=input_data,
            output_dir=output_dir,
            anomaly_type=anomaly_type,
            anomaly_rate=anomaly_rate,
            length_range=length_range,
            strength_range=strength_range,
            seed=seed,
        )
    except Exception as exc:
        fail(exc)

    write(result)


@pseudo_anomaly_app.command("run")
def run(
    model_root: Path = typer.Option(..., "--model-root"),
    probe_dir: Path = typer.Option(..., "--probe-dir"),
    device: str = typer.Option("mps", "--device"),
) -> None:
    try:
        with redirect_stdout(sys.stderr):
            result = run_pseudo_anomaly(
                model_root=model_root,
                probe_dir=probe_dir,
                device=device,
            )
    except Exception as exc:
        fail(exc)

    # stdout
    write(result)
    if result["errors"]:
        raise typer.Exit(code=1)
