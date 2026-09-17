from __future__ import annotations

import sys
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.table import Table
from rich.text import Text


class TotalTimeElapsedColumn(ProgressColumn):
    def __init__(self, get_elapsed: Callable[[], float]):
        super().__init__()
        self.get_elapsed = get_elapsed

    def render(self, task: Task) -> Text:
        elapsed = max(0, int(self.get_elapsed()))
        return Text(str(timedelta(seconds=elapsed)), style="progress.elapsed")


class BatchProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if not task.fields.get("show_batches", False):
            return Text("")
        return Text(f"{int(task.completed)}/{int(task.total)} batches")


class NullTrainingReporter:
    def begin_stage(self, *args, **kwargs):
        pass

    def begin_epoch(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def end_epoch(self, *args, **kwargs):
        pass


class TrainingReporter:
    """Render training state on stderr and keep plain-text epoch logs."""

    DISPLAY_PRECISION = 2

    def __init__(self, info: dict, log_path: Path, show_progress: bool = True):
        self.info = info
        self.log_path = log_path
        self.show_progress = show_progress
        # Bind the terminal before redirecting Python's stdout/stderr to the log.
        self.console = Console(file=sys.stderr)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(overflow="fold")
        for name, value in info.items():
            table.add_row(Text(name), Text(str(value)))
        self.info_table = table
        self.progress = Progress(
            TextColumn("{task.description}"), BarColumn(), BatchProgressColumn(),
            TotalTimeElapsedColumn(self._training_elapsed),
            console=self.console, auto_refresh=False,
        )
        self.task = self.progress.add_task(
            "Preparing", total=None, start=False, show_batches=False,
        )
        self.status = "Preparing"
        self.metrics: dict[str, str] = {}
        self.stopped_early = False
        self.stage_active = False
        self.live = None
        self._stack = ExitStack()

    def __enter__(self):
        self.started_at = monotonic()
        try:
            self.log_file = self._stack.enter_context(
                self.log_path.open("w", encoding="utf-8", buffering=1)
            )
            self._stack.enter_context(redirect_stdout(self.log_file))
            self._stack.enter_context(redirect_stderr(self.log_file))
            self._write("Training info")
            for name, value in self.info.items():
                self._write(f"{name}: {value}")
            if self.show_progress:
                self.live = self._stack.enter_context(Live(
                    self._render(), console=self.console, refresh_per_second=10,
                    redirect_stdout=False, redirect_stderr=False,
                ))
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                self.status = "Interrupted" if isinstance(exc, KeyboardInterrupt) else "Failed"
                self._write(f"{self.status}: {exc_type.__name__}: {exc}")
                traceback.print_exception(exc_type, exc, tb, file=self.log_file)
                self.progress.stop_task(self.task)
                self._refresh(force=True)
        finally:
            self._stack.close()
        return False

    def begin_stage(self, description: str):
        if not hasattr(self, "training_started_at"):
            self.training_started_at = monotonic()
        self.stage_active = True
        self.status = description
        self.metrics = {}
        self.progress.reset(
            self.task, total=None, description=description, show_batches=False,
        )
        self._write(description)
        self._refresh(force=True)

    def begin_epoch(self, epoch: int, total_epochs: int, total_batches: int):
        if not hasattr(self, "training_started_at"):
            self.training_started_at = monotonic()
        self.epoch = epoch
        self.stage_active = False
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.current_step = 0
        self.loss_sum = 0.0
        self.status = "Training"
        previous_val_loss = self.metrics.get("Val loss", self.metrics.get("Last val loss"))
        best_epoch = self.metrics.get("Best epoch")
        patience = self.metrics.get("Patience")
        self.metrics = {"Train loss": "—"}
        if previous_val_loss is not None:
            self.metrics["Last val loss"] = previous_val_loss
        if best_epoch is not None:
            self.metrics["Best epoch"] = best_epoch
        if patience is not None:
            self.metrics["Patience"] = patience
        self.progress.reset(
            self.task, total=total_batches, description=f"Epoch {epoch}/{total_epochs}",
            show_batches=True,
        )
        self._refresh(force=True)

    def step(self, loss: float):
        self.current_step += 1
        self.loss_sum += float(loss)
        self.metrics["Train loss"] = self._format_loss(
            self.loss_sum / self.current_step
        )
        self.progress.update(self.task, completed=self.current_step)
        # Every step updates the state; terminal rendering is capped at 10 Hz.
        self._refresh()

    def end_epoch(
        self,
        val_loss: float | None = None,
        best_epoch: int | None = None,
        early_stopping=None,
    ):
        train_loss = self.loss_sum / self.current_step
        stopped_early = early_stopping is not None and early_stopping.early_stop
        early_stop_counter = (
            early_stopping.counter if early_stopping is not None else None
        )
        patience = early_stopping.patience if early_stopping is not None else None
        self.stopped_early = stopped_early
        self.status = "Early stopped" if stopped_early else "Epoch complete"
        self.metrics = {"Train loss": self._format_loss(train_loss)}
        if val_loss is not None:
            self.metrics["Val loss"] = self._format_loss(val_loss)
        if best_epoch is not None:
            self.metrics["Best epoch"] = str(best_epoch)
        if early_stop_counter is not None:
            self.metrics["Patience"] = f"{early_stop_counter}/{patience}"
        log_fields = [self.status, f"Train loss: {train_loss:.6f}"]
        if val_loss is not None:
            log_fields.append(f"Val loss: {val_loss:.6f}")
        if best_epoch is not None:
            log_fields.append(f"Best epoch: {best_epoch}")
        if early_stop_counter is not None:
            log_fields.append(f"Patience: {early_stop_counter}/{patience}")
        self._write(
            f"Epoch {self.epoch}/{self.total_epochs} | "
            f"batches={self.total_batches}/{self.total_batches} | "
            + " | ".join(log_fields)
        )
        self._refresh(force=True)

    def finish(self, best_epoch: int | None):
        self.status = "Early stopped" if self.stopped_early else "Completed"
        if best_epoch is not None:
            self.metrics["Best epoch"] = str(best_epoch)
        if self.stage_active:
            self.progress.update(self.task, total=1, completed=1)
        self.progress.stop_task(self.task)
        self._write(
            f"{self.status} | best_epoch={best_epoch} | "
            f"elapsed={monotonic() - self.started_at:.2f}s"
        )
        self._refresh(force=True)

    def _write(self, message: str):
        self.log_file.write(message + "\n")

    def _status_line(self):
        return " | ".join([self.status, *(f"{k}: {v}" for k, v in self.metrics.items())])

    def _format_loss(self, loss: float) -> str:
        return f"{loss:.{self.DISPLAY_PRECISION}f}"

    def _training_elapsed(self) -> float:
        if not hasattr(self, "training_started_at"):
            return 0.0
        return monotonic() - self.training_started_at

    def _render(self):
        return Panel(
            Group(self.info_table, Text(""), self.progress, Text(self._status_line())),
            title=Text("Training info", style="bold red"), title_align="left",
        )

    def _refresh(self, force: bool = False):
        if self.live is not None:
            self.live.update(self._render(), refresh=force)
