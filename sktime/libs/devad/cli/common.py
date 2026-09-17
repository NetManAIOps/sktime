import json
import typer
from typing import Any


def write(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False))


def fail(exc: Exception) -> None:
    write(
        {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        }
    )
    raise typer.Exit(code=1)