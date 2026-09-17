from __future__ import annotations

import typer

from .model import model_app
from .probe import probe_app


app = typer.Typer(name="devad", no_args_is_help=True)
app.add_typer(model_app, name="model")
app.add_typer(probe_app, name="probe")


if __name__ == "__main__":
    app()
