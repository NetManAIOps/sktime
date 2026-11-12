"""Early-stopping decisions without checkpoint or display responsibilities."""

from math import isfinite

__all__ = ["EarlyStopping"]


class EarlyStopping:
    """Track a validation metric and stop after consecutive non-improvements.

    Parameters
    ----------
    patience : int, default=7
        Number of consecutive non-improving updates before stopping. Must be
        positive; callers disable early stopping by not creating this object.
    delta : float, default=0.0
        With inclusive=False, an improvement must exceed this absolute margin.
        Equal values then do not count as improvements, even when delta is zero.
    mode : {"min", "max"}, default="min"
        Whether smaller or larger metric values are better.
    inclusive : bool, default=False
        Count improvements exactly at the delta boundary. Set True to preserve
        DevAD's comparison, including equal metrics when delta is zero.

    Notes
    -----
    Create a fresh instance for each training run. This utility only tracks
    metrics; the detector saves weights when update returns improved=True.
    """

    def __init__(self, patience=7, delta=0.0, mode="min", inclusive=False):
        if patience < 1:
            raise ValueError("patience must be positive")
        if not isfinite(delta) or delta < 0:
            raise ValueError("delta must be finite and non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.inclusive = inclusive
        self.best_metric = None
        self.counter = 0
        self.early_stop = False

    def update(self, metric):
        """Return whether one validation result improved the best metric.

        The first finite result is an improvement. Subsequent improvements
        reset the counter. Read early_stop for the stopping decision.
        Once stopped, further updates return False.
        """
        if self.early_stop:
            return False

        metric = float(metric)
        if not isfinite(metric):
            raise ValueError("validation metric must be finite")

        if self.best_metric is None:
            improved = True
        else:
            # Use DevAD's signed-score arithmetic to retain boundary behavior.
            score = -metric if self.mode == "min" else metric
            best_score = -self.best_metric if self.mode == "min" else self.best_metric
            boundary = best_score + self.delta
            improved = score >= boundary if self.inclusive else score > boundary

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

        return improved
