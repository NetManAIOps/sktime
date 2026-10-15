"""User algorithm plugins: discovery, validation, and fork scaffolding.

User algorithms are single-file plugins in ``playground/experiments/`` (see
that package's docstring for the contract). They are discovered by
:func:`discover_user_algorithms` and merged into the catalog by
``catalog.build_catalog``; a broken plugin only disables itself, never the
whole catalog.

In addition to file-based plugins, this module hosts built-in user
algorithms: thin wrappers that expose an estimator the plugin mechanism
cannot express (e.g. scalar-only constructor params wrapping a nested
sklearn estimator). They share the ``user-`` id prefix and the catalog
merge path but are defined here so they can subclass a sktime base class —
a requirement of the `libraryts train` persistence contract — without
depending on the experiments directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from sktime.clustering.base import BaseClusterer

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
USER_ID_PREFIX = "user-"

_TASKS = ("forecasting", "classification", "regression", "clustering", "anomaly_detection")

# task -> (required methods, human-readable contract)
_CONTRACT = {
    "forecasting": (("fit", "predict"), "fit(y) + predict(steps)"),
    "classification": (("fit", "predict"), "fit(X, y) + predict(X)"),
    "regression": (("fit", "predict"), "fit(X, y) + predict(X)"),
    "clustering": (("fit", "predict"), "fit(X) + predict(X)"),
    "anomaly_detection": (("fit", "predict"), "fit(X) + predict(X), or fit_predict(X)"),
}


class PluginError(Exception):
    """Contract violation in a user plugin file."""


def user_algorithm_id(stem: str) -> str:
    return f"{USER_ID_PREFIX}{stem}"


def iter_plugin_files() -> list[Path]:
    if not EXPERIMENTS_DIR.is_dir():
        return []
    return sorted(
        path
        for path in EXPERIMENTS_DIR.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith(".")
    )


def _load_module(path: Path):
    """Load a plugin file as an importable module (experiments.<stem>)."""
    module_name = f"experiments.{path.stem}"
    # Reload if already imported so edits take effect within a process.
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_plugin(module, path: Path) -> dict:
    """Validate the plugin contract; return {task, name, params, klass}."""
    task = getattr(module, "TASK", None)
    if task not in _TASKS:
        raise PluginError(
            f"TASK must be one of {_TASKS}, got {task!r}."
        )
    klass = getattr(module, "Algorithm", None)
    if not isinstance(klass, type):
        raise PluginError("Plugin must define an `Algorithm` class.")
    required, readable = _CONTRACT[task]
    missing = [name for name in required if not callable(getattr(klass, name, None))]
    if missing and not (task == "anomaly_detection" and callable(getattr(klass, "fit_predict", None))):
        raise PluginError(
            f"`Algorithm` is missing method(s) {missing}; the {task} contract "
            f"is {readable}."
        )
    params = getattr(module, "PARAMS", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise PluginError(f"PARAMS must be a dict, got {type(params).__name__}.")
    name = getattr(module, "NAME", None) or path.stem
    return {"task": task, "name": str(name), "params": dict(params), "klass": klass}


def _plugin_cache_file() -> Path | None:
    from catalog import _disk_cache_dir

    cache_dir = _disk_cache_dir()
    return cache_dir / "user-plugins.json" if cache_dir else None


def _load_plugin_cache() -> dict:
    cache_file = _plugin_cache_file()
    try:
        return json.loads(cache_file.read_text(encoding="utf-8")) if cache_file else {}
    except (OSError, ValueError):
        return {}


def _save_plugin_cache(cache: dict) -> None:
    cache_file = _plugin_cache_file()
    if cache_file is None:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(f".tmp{__import__('os').getpid()}")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(cache_file)
    except OSError:
        pass


def discover_user_algorithms() -> list[dict]:
    """Discover experiment plugins; never raises on a broken file.

    Importing a plugin can pull in heavy dependencies (torch, tensorflow), so
    validated plugins are cached by (path, mtime, size, this file's mtime) and
    only re-imported when they change. Broken plugins are never cached.
    """
    entries = []
    cache = _load_plugin_cache()
    fresh_cache: dict = {}
    self_stamp = Path(__file__).stat().st_mtime_ns
    for path in iter_plugin_files():
        stat = path.stat()
        stamp = [stat.st_mtime_ns, stat.st_size, self_stamp]
        cached = cache.get(str(path))
        if cached and cached.get("stamp") == stamp:
            entries.append(cached["entry"])
            fresh_cache[str(path)] = cached
            continue
        base = {
            "id": user_algorithm_id(path.stem),
            "task": "all",
            "module": f"experiments.{path.stem}.Algorithm",
            "class_name": "Algorithm",
            "curated": False,
            "user": True,
            "plugin_file": str(path),
        }
        try:
            info = _validate_plugin(_load_module(path), path)
            base.update(
                name=info["name"],
                task=info["task"],
                enabled=True,
                params=info["params"],
            )
            if info["task"] == "anomaly_detection":
                base["subtype"] = "point_anomaly"
            forked_from = getattr(sys.modules[f"experiments.{path.stem}"], "FORKED_FROM", None)
            if forked_from:
                base["forked_from"] = forked_from
        except Exception as exc:
            base.update(
                name=path.stem,
                enabled=False,
                disabled_reason=f"{type(exc).__name__}: {exc}",
                params={},
            )
        entries.append(base)
        if base.get("enabled"):
            fresh_cache[str(path)] = {"stamp": stamp, "entry": base}
    if fresh_cache != cache:
        _save_plugin_cache(fresh_cache)
    entries.extend(_builtin_user_algorithms())
    return entries


def get_user_algorithm(algorithm_id: str) -> dict | None:
    if not algorithm_id or not algorithm_id.startswith(USER_ID_PREFIX):
        return None
    for entry in discover_user_algorithms():
        if entry["id"] == algorithm_id and entry.get("enabled"):
            return entry
    return None


# ----------------------------------------------------------------------
# built-in user algorithms
# ----------------------------------------------------------------------


class AgglomerativePanelClusterer(BaseClusterer):
    """Agglomerative (hierarchical) clustering of time series panels.

    Thin wrapper over sklearn's ``AgglomerativeClustering``: the panel is
    flattened to one vector per series (numpy3D ``(n, c, t) -> (n, c*t)``)
    and clustered with the given linkage and metric. Defaults are sklearn's
    own (ward linkage, Euclidean) — the configuration the Javed et al.
    time series clustering benchmark's published Agglomerative scores
    actually match (verified: 40/41 UCR datasets reproduce the paper CSV
    to machine precision; ``linkage='complete'`` as written in the
    benchmark's code/main.py:135 does not).

    The algorithm is transductive — sklearn's agglomerative clustering has
    no out-of-sample assignment — so ``predict`` returns the fitted
    ``labels_`` for the training panel and otherwise re-clusters the new
    panel with a fresh clone, the same contract as ``TimeSeriesDBSCAN``.
    """

    _tags = {
        "capability:out_of_sample": False,
        "capability:predict_proba": False,
        "capability:multivariate": True,
        "X_inner_mtype": "numpy3D",
    }

    def __init__(self, n_clusters=2, linkage="ward", metric="euclidean"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        from sklearn.cluster import AgglomerativeClustering

        self._X = X
        flat = X.reshape(X.shape[0], -1)
        self.agglomerative_ = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=self.linkage, metric=self.metric
        )
        self.labels_ = self.agglomerative_.fit_predict(flat)
        return self

    def _predict(self, X, y=None):
        from sktime.utils.warnings import warn

        if X is self._X:
            return self.labels_
        warn(
            "sklearn AgglomerativeClustering does not support different X "
            "in fit and predict, but a new X was passed in predict. "
            "Therefore, a clone of AgglomerativePanelClusterer will be fit, "
            "and results returned, without updating the state of the "
            "fitted estimator.",
            obj=self,
        )
        return self.clone().fit(X).labels_


def _builtin_user_algorithms() -> list[dict]:
    """Catalog entries for the built-in wrappers defined in this module."""
    return [
        {
            "id": "user-agglomerative",
            "name": "Agglomerative (Euclidean, ward linkage)",
            "task": "clustering",
            "module": "user_algos.AgglomerativePanelClusterer",
            "class_name": "AgglomerativePanelClusterer",
            "curated": False,
            "user": True,
            "builtin": True,
            "enabled": True,
            "params": {"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
        },
    ]


# ----------------------------------------------------------------------
# fork scaffolding
# ----------------------------------------------------------------------

_FORK_TEMPLATE = '''"""Fork of {algorithm_id} (`{module}`).

Edit this file freely, then:
    libraryts.py check playground/experiments/{stem}.py
    libraryts.py run --algorithm user-{stem} --task {task} ...

The original implementation lives in `{module}`; this subclass starts as an
exact copy of its behavior. Override methods, change defaults in PARAMS, or
replace the body entirely — anything matching the `{task}` plugin contract
(see playground/experiments/__init__.py) runs in the Playground.
"""

TASK = {task!r}
NAME = {name!r}
FORKED_FROM = {algorithm_id!r}
PARAMS = {params_repr}

from {module_name} import {class_name} as _Base


class Algorithm(_Base):
    """Fork of {class_name}; starts identical to the original."""

    pass
'''


def scaffold_fork(algorithm: dict, name: str | None = None) -> Path:
    """Write a subclass-based fork scaffold for a catalog algorithm."""
    stem = (name or algorithm["name"]).lower().replace(" ", "_")
    stem = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)
    target = EXPERIMENTS_DIR / f"{stem}.py"
    if target.exists():
        raise PluginError(f"Plugin file already exists: {target}")
    module_path = algorithm["module"]
    module_name, _, class_name = module_path.rpartition(".")
    params = algorithm.get("params") or {}
    content = _FORK_TEMPLATE.format(
        algorithm_id=algorithm["id"],
        module=module_path,
        stem=stem,
        task=algorithm["task"],
        name=name or algorithm["name"],
        params_repr=repr(params),
        module_name=module_name,
        class_name=class_name,
    )
    target.write_text(content, encoding="utf-8")
    return target


# ----------------------------------------------------------------------
# check: contract validation + tiny smoke run
# ----------------------------------------------------------------------

def check_plugin(path: Path) -> dict:
    """Validate a plugin file and run a tiny smoke experiment.

    Returns {"ok": bool, "checks": [str], "error": str | None}.
    """
    checks = []
    try:
        module = _load_module(path)
        info = _validate_plugin(module, path)
        checks.append(
            f"contract ok: TASK={info['task']}, PARAMS={list(info['params'])}"
        )
        klass = info["klass"]
        from catalog import EVAL_PARAMS

        eval_keys = EVAL_PARAMS.get(info["task"], set())
        instance = klass(**{k: v for k, v in info["params"].items() if k not in eval_keys})
        checks.append(f"constructed {klass.__name__} with default PARAMS")
        _smoke_run(info["task"], instance, checks)
        return {"ok": True, "checks": checks, "error": None}
    except Exception as exc:
        import traceback

        return {
            "ok": False,
            "checks": checks,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        }


def _is_sktime_estimator(instance) -> bool:
    return callable(getattr(instance, "get_params", None))


def _smoke_run(task: str, instance, checks: list[str]) -> None:
    import numpy as np

    if task == "forecasting":
        from sktime.datasets import load_airline

        y = load_airline()[:72]
        if _is_sktime_estimator(instance):
            fh = [1, 2, 3]
            instance.fit(y, fh=fh)
            pred = instance.predict(fh=fh)
        else:
            instance.fit(y)
            pred = instance.predict(3)
        arr = np.asarray(pred, dtype=float).ravel()
        if len(arr) != 3 or not np.isfinite(arr).all():
            raise PluginError(
                f"predict must return 3 finite values, got shape {arr.shape}."
            )
        checks.append("smoke ok: fit on 72 points, predicted 3 finite values")

    elif task == "anomaly_detection":
        import pandas as pd

        from catalog import REPO_ROOT

        frame = pd.read_csv(REPO_ROOT / "sktime/datasets/data/yahoo/yahoo.csv")
        X = frame["data"].astype(float).iloc[:500].to_frame("data")
        if _is_sktime_estimator(instance):
            # sktime detectors return sparse event ilocs from (fit_)predict;
            # transform() is the dense per-point label contract.
            out = instance.fit(X).transform(X)
        elif callable(getattr(instance, "fit_predict", None)):
            out = instance.fit_predict(X)
        else:
            instance.fit(X)
            out = instance.predict(X)
        arr = np.asarray(out).ravel()
        if len(arr) != len(X):
            raise PluginError(
                f"anomaly output length {len(arr)} != input length {len(X)}; "
                "the contract is a dense 0/1 label per point."
            )
        checks.append(f"smoke ok: {int((arr != 0).sum())} anomalies in 500 points")

    else:
        from sktime.datasets._single_problem_loaders import load_unit_test

        X_train, y_train = load_unit_test(split="train", return_X_y=True)
        X_test, _ = load_unit_test(split="test", return_X_y=True)
        if task in ("classification", "regression"):
            instance.fit(X_train, y_train)
        else:
            instance.fit(X_train)
        pred = np.asarray(instance.predict(X_test.iloc[:5])).ravel()
        if len(pred) != 5:
            raise PluginError(f"predict must return 5 rows, got {len(pred)}.")
        checks.append(f"smoke ok: fit on unit-test train, predicted 5 rows")
