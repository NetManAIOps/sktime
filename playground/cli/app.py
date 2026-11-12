"""LibraryTS command-line entry point."""

import typer

from playground.cli import model
from playground.cli.ls import app as ls_app

app = typer.Typer(
    help="Discover time-series components and run experiments with LibraryTS.",
    no_args_is_help=True,
    add_completion=False,
)

# 在根 ctx 中写入 meta 字典 {"debug": debug}
@app.callback()
def configure(
    ctx: typer.Context,
    debug: bool = typer.Option(
        False, "--debug", help="Print exception tracebacks to stderr."
    ),
) -> None:
    """Configure CLI diagnostics."""
    ctx.meta["debug"] = debug


app.add_typer(ls_app, name="ls") # 注册命令组
app.command()(model.train)
app.command()(model.predict)
app.command()(model.evaluate)
app.command()(model.run)


if __name__ == "__main__":
    app()
