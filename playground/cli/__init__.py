"""Command-line interfaces for LibraryTS services."""

import json
import sys
import traceback

import typer


def report_error(ctx: typer.Context, exc: Exception) -> None:
    """Write the CLI error response and optionally its traceback."""
    if ctx.meta.get("debug", False):
        traceback.print_exception(exc, file=sys.stderr)
    typer.echo(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
    raise typer.Exit(code=1) from exc
