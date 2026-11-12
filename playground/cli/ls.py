"""Read-only component discovery commands."""

import json
import sys
import warnings
from contextlib import redirect_stdout

import typer

from playground.cli import report_error

app = typer.Typer(help="List components or inspect their parameters.", no_args_is_help=True)


@app.command()
def algorithms(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task", help="Task, e.g. anomaly_detection."),
    name: str | None = typer.Option(None, "--name", help="Exact algorithm class name."),
) -> None:
    """List algorithms or show one algorithm's description and constructor parameters."""
    show(ctx, "algorithms", task=task, name=name)


@app.command()
def preprocessors(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task", help="Task, e.g. anomaly_detection."),
    name: str | None = typer.Option(None, "--name", help="Exact transformer class name."),
) -> None:
    """List task preprocessors, or show one transformer's parameters."""
    show(ctx, "preprocessors", task=task, name=name)


def show(ctx: typer.Context, kind: str, **kwargs) -> None:
    """Keep stdout as JSON and send discovery output to stderr."""
    try:
        with redirect_stdout(sys.stderr), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using `json`-module for json-handling\..*",
                category=UserWarning,
                module=r"gluonts\.json",
            )
            from playground.service.catalog import list_algorithms, list_preprocessors

            query = {
                "algorithms": list_algorithms,
                "preprocessors": list_preprocessors,
            }[kind]
            result = query(**kwargs) # service
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    except Exception as exc:
        report_error(ctx, exc)
    typer.echo(payload)
