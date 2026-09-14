"""Drop-in experiment algorithms for the TSBox Sandbox Playground.

Every ``*.py`` file in this directory (except this one) is a self-contained
algorithm plugin, discovered by ``playground/catalog.py`` and runnable from
the web UI and the LabTS CLI as ``user-<filename>``.

Required top-level names
------------------------
TASK : str
    One of "forecasting", "classification", "regression", "clustering",
    "anomaly_detection".
Algorithm : class
    The algorithm implementation.

Optional top-level names
------------------------
NAME : str
    Display name (defaults to the file stem).
PARAMS : dict
    Default hyperparameters; numeric values become UI/CLI knobs.
FORKED_FROM : str
    Algorithm id this file was forked from (provenance, written by
    ``labts fork``).

Minimal method contract (per task)
----------------------------------
- forecasting:          ``fit(y: pd.Series)`` + ``predict(steps: int) -> array-like``
                        (return exactly ``steps`` values)
- classification:       ``fit(X, y)`` + ``predict(X)``
- regression:           ``fit(X, y)`` + ``predict(X)``
- clustering:           ``fit(X)`` + ``predict(X)``
- anomaly_detection:    ``fit_predict(X: pd.DataFrame) -> dense 0/1 array``
                        (or ``fit(X)`` + ``predict(X)``)

Classes that implement the sktime estimator interface (``get_params`` etc.)
are driven through the full sktime contract instead (e.g. ``predict(fh=...)``
for forecasters), so ``labts fork`` scaffolds can subclass any catalog
algorithm directly.

Workflow
--------
    labts.py fork registered-forecasting-TimesNetForecaster --name my_timesnet
    # edit playground/experiments/my_timesnet.py
    labts.py check playground/experiments/my_timesnet.py
    labts.py run --algorithm user-my-timesnet --task forecasting ...
"""

