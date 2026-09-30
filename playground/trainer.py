"""Persistent model training and prediction for playground models.

Two lifecycle forms share one API:

* ``labts run`` (no model id) is stateless: fit and predict happen in one
  call and the fitted model is discarded.
* ``labts train`` + ``labts run --model-id`` split the lifecycle for EVERY
  catalog algorithm/task (``predict``/``detect`` are aliases of the model
  path). The train-once/run-many workflow persists fitted models under
  ``playground/models/<model_id>/`` and reloads them later::

    labts train --algorithm registered-forecasting-SARIMAX --dataset airline \
        --model-id sarimax-v1 --param horizon=12
    labts run   --model-id sarimax-v1 --dataset airline

    labts train --algorithm registered-anomaly_detection-DevADFITSDetector \
        --dataset yahoo --model-id fits-v1 --param epochs=3
    labts run   --model-id fits-v1 --dataset yahoo --param threshold_quantile=0.99

Backend selection rule (`train(spec)`): an algorithm whose catalog module
starts with ``sktime.detection.adapters.devad.`` is trained through the
DevAD backend (`train_devad`; ``model.pt`` + DevAD manifest via
``sktime.libs.devad.services.model_service``). Every other algorithm is
fitted as a plain sktime estimator and persisted through the sktime
save/load backend in `persistence` (``model.zip`` + ``manifest.json``; the
manifest schema — task/algorithm_id/params/backend/created_at/spec plus
artifacts — is documented in ``playground/persistence.py``). Curated
catalog algorithms map to the same estimators as their `labts run`
counterparts, except the curated anomaly detector whose `run` pipeline
embeds ad-hoc detrending that is not part of the estimator and is
therefore rejected with a hint to use a registered detector instead.

Train/predict reproduces the one-shot `run` evaluation protocol per task:
rolling-origin forecasting (``eval_mode=rolling`` + split fractions, via
tslib ``predict_windows``), clustering ``fit_on=all`` (fused train+test),
causal discovery (fitted discoverer persisted; predict re-scores its graph
against the true DAG), and multi-series anomaly datasets (one fitted
detector per series under ``series/<NNNN>/model.zip``, manifest
``multiseries: true``; predict averages the per-series metrics exactly like
the one-shot multi-series runner).

`predict_estimator(model_id, dataset_id, params)` reloads a persisted
model and evaluates it on the dataset's holdout split, returning the same
result envelope as `labts run`, so `labts report --from` works on it
unchanged. For DevAD models it delegates to `detect_devad`; for sktime
models the holdout split is reproduced from the manifest's eval params,
overridable via `params` (e.g. ``{"horizon": 24}``). The fitted
preprocessor selected at train time is persisted alongside the model and
re-applied (transform only, never refit) at predict time.
"""

from __future__ import annotations

import datetime
import json
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

import persistence
from catalog import (
    EVAL_PARAMS,
    REPO_ROOT,
    get_dataset,
    get_enabled_algorithm,
    get_enabled_preprocessor,
    import_estimator_class,
    split_params,
)
from runners import (
    PlaygroundError,
    build_anomaly_result,
    generate_report,
    load_anomaly_series,
)

MODELS_ROOT = REPO_ROOT / "playground" / "models"
_ADAPTER_MODULE = "sktime.detection.adapters.devad."
# Constructor params of the DevAD adapter (everything else is a DevAD HP).
_ADAPTER_PARAMS = {
    "threshold_quantile",
    "win_len",
    "epochs",
    "batch_size",
    "seed",
    "device",
    "params",
}
# Curated-detector eval params (see catalog.EVAL_PARAMS); not DevAD HPs.
_EVAL_PARAMS = {"threshold", "window"}
# Default dataset per task when a train spec omits dataset_id (mirrors
# runners._normalize_spec).
_TASK_DEFAULT_DATASETS = {
    "forecasting": "airline",
    "classification": "unit-test",
    "regression": "covid-3month",
    "clustering": "unit-test-cl",
    "anomaly_detection": "yahoo",
    "causal": "causal-sachs",
}


def list_trained_models() -> list[dict]:
    """All persisted models under playground/models/ with a valid manifest."""
    rows = []
    if not MODELS_ROOT.is_dir():
        return rows
    for manifest_path in sorted(MODELS_ROOT.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        backend = persistence.detect_backend(manifest)
        rows.append(
            {
                "model_id": manifest_path.parent.name,
                "backend": backend,
                "family": manifest.get("family"),
                "task": manifest.get("task")
                or ("anomaly_detection" if backend == "devad" else None),
                "algorithm_id": manifest.get("algorithm_id"),
                "params": manifest.get("params"),
                "seed": manifest.get("seed"),
                "best_epoch": manifest.get("best_epoch"),
                "multiseries": bool(manifest.get("multiseries")),
                "created_at": manifest.get("created_at"),
                "model_dir": str(manifest_path.parent),
            }
        )
    return rows


def train(spec: dict) -> dict:
    """Train any catalog algorithm and persist the model (generic entry point).

    Routes to the DevAD backend (`train_devad`) or the sktime save/load
    backend (`persistence.save_sktime_model`) by the algorithm's catalog
    module — see the module docstring for the selection rule. Returns a
    dict with at least ``model_id``, ``model_dir``, and ``manifest``.
    """
    algorithm_id = spec.get("algorithm_id")
    if not algorithm_id:
        raise PlaygroundError(
            "`train` requires algorithm_id (see `labts.py ls algorithms`)."
        )
    declared_task = spec.get("task")
    algorithm = None
    if declared_task and "/" not in algorithm_id:
        # scope short-name resolution to the declared task first
        algorithm = get_enabled_algorithm(f"{declared_task}/{algorithm_id}")
    if algorithm is None:
        algorithm = get_enabled_algorithm(algorithm_id)
    if algorithm is None or not algorithm.get("enabled"):
        raise PlaygroundError(f"Algorithm is not enabled: {algorithm_id}")
    if declared_task and algorithm.get("task") != declared_task:
        raise PlaygroundError(
            f"Algorithm {algorithm_id} resolves to {algorithm['id']} "
            f"(task: {algorithm.get('task')}), not the declared task "
            f"{declared_task}. Drop --task or fix the algorithm name."
        )
    module = algorithm.get("module") or ""
    if module.startswith(_ADAPTER_MODULE):
        result = train_devad(spec)
        manifest_path = result.get("manifest_path")
        manifest = {}
        if manifest_path and Path(manifest_path).is_file():
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        # Uniform generic-contract view of the manifest; the on-disk DevAD
        # manifest written by model_service stays untouched.
        result["manifest"] = {
            **manifest,
            "backend": "devad",
            "task": "anomaly_detection",
            "algorithm_id": algorithm["id"],
            "created_at": manifest.get(
                "created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()
            ),
            "spec": {
                "task": "anomaly_detection",
                "dataset_id": result.get("dataset_id"),
                "algorithm_id": algorithm["id"],
                "preprocessor_id": spec.get("preprocessor_id") or "none",
                "params": spec.get("params") or {},
            },
        }
        result["backend"] = "devad"
        result["task"] = "anomaly_detection"
        return result
    return _train_sktime(spec, algorithm)


def predict_estimator(
    model_id: str, dataset_id: str | None = None, params: dict | None = None
) -> dict:
    """Reload a persisted model and evaluate it on a dataset's holdout split.

    Generic predict backend for `labts predict`; returns the same result
    envelope as `labts run`. DevAD models delegate to `detect_devad`
    (anomaly alias), sktime models are reloaded from ``model.zip`` and
    scored against a holdout reproduced from the manifest's eval params
    (`params` overrides them, e.g. ``{"horizon": 24}``).
    """
    started = time.perf_counter()
    if not model_id:
        raise PlaygroundError(
            "`predict` requires model_id (see `labts.py ls models`)."
        )
    persistence.validate_model_id(model_id)
    model_dir = MODELS_ROOT / model_id
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PlaygroundError(
            f"No trained model `{model_id}` under {MODELS_ROOT} "
            "(see `labts.py ls models`)."
        )
    manifest = persistence.read_manifest(model_dir)
    if persistence.detect_backend(manifest) == "devad":
        return detect_devad(
            {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "params": params or {},
            }
        )
    return _predict_sktime(
        model_id, model_dir, manifest, dataset_id, params or {}, started
    )


def _train_sktime(spec: dict, algorithm: dict) -> dict:
    """Fit a non-DevAD catalog algorithm and persist it via the sktime backend."""
    started = time.perf_counter()
    task = algorithm["task"]
    dataset_id = spec.get("dataset_id") or _TASK_DEFAULT_DATASETS.get(task, "airline")
    dataset = get_dataset(dataset_id)
    if dataset is None or not dataset.get("enabled"):
        raise PlaygroundError(f"Dataset is not enabled: {dataset_id}")
    if dataset.get("task") != task:
        raise PlaygroundError(
            f"Dataset `{dataset['id']}` is a {dataset.get('task')} dataset; "
            f"algorithm `{algorithm['id']}` is a {task} algorithm."
        )
    preprocessor = get_enabled_preprocessor(spec.get("preprocessor_id"))
    if preprocessor is None or not preprocessor.get("enabled"):
        raise PlaygroundError(
            f"Preprocessor is not enabled: {spec.get('preprocessor_id')}"
        )
    compat = preprocessor.get("compatible_tasks")
    if compat and task not in compat and "all" not in compat:
        raise PlaygroundError(
            f"Preprocessor `{preprocessor['name']}` is not compatible with task "
            f"`{task}`. Compatible tasks: {', '.join(compat)}."
        )

    params = spec.get("params") or {}
    eval_params, est_params = split_params(task, params)
    estimator = _build_train_estimator(algorithm, est_params)
    if not callable(getattr(estimator, "save", None)):
        raise PlaygroundError(
            f"Algorithm `{algorithm['id']}` does not implement the sktime "
            "save/load contract (no `save` method) and cannot be persisted. "
            "User plugins need to inherit from a sktime base class to work "
            "with `labts train`."
        )
    prep_est = _build_train_preprocessor(preprocessor, spec)

    log = [
        f"Selected task: {task}",
        f"Selected dataset: {dataset['name']}",
        f"Selected preprocessor: {preprocessor['name']}",
        f"Selected algorithm: {algorithm['name']} (sktime backend)",
        f"Estimator params: {est_params or 'defaults'}",
    ]
    try:
        fit_info, eval_params = _fit_sktime_estimator(
            estimator,
            prep_est,
            preprocessor.get("name", "preprocessor"),
            task,
            dataset,
            eval_params,
            log,
        )
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(f"Training failed: {type(exc).__name__}: {exc}") from exc

    model_id = spec.get("model_id") or f"{algorithm['name'].lower()}-{dataset['id']}"
    normalized_spec = {
        "task": task,
        "dataset_id": dataset["id"],
        "algorithm_id": algorithm["id"],
        "preprocessor_id": preprocessor.get("id", "none"),
        "params": params,
        "preprocessor_params": spec.get("preprocessor_params") or {},
    }
    multiseries = fit_info.pop("_multiseries", None)
    if multiseries is not None:
        manifest = persistence.save_sktime_multiseries_model(
            multiseries,
            models_root=MODELS_ROOT,
            model_id=model_id,
            task=task,
            algorithm=algorithm,
            est_params=est_params,
            eval_params=eval_params,
            spec=normalized_spec,
        )
        fit_info["series_models"] = int(len(multiseries))
    else:
        manifest = persistence.save_sktime_model(
            estimator,
            models_root=MODELS_ROOT,
            model_id=model_id,
            task=task,
            algorithm=algorithm,
            est_params=est_params,
            eval_params=eval_params,
            spec=normalized_spec,
            preprocessor=prep_est,
        )
    model_dir = MODELS_ROOT / model_id
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    log.append(f"Finished in {elapsed_ms} ms")
    return {
        "status": "ok",
        "backend": "sktime",
        "task": task,
        "model_id": model_id,
        "algorithm_id": algorithm["id"],
        "dataset_id": dataset["id"],
        **fit_info,
        "model_dir": str(model_dir),
        "manifest": manifest,
        "manifest_path": str(model_dir / "manifest.json"),
        "duration_ms": elapsed_ms,
        "log": log,
        "next_steps": [
            f"labts.py run --model-id {model_id} --dataset {dataset['id']}",
            "labts.py ls models",
        ],
    }


def _predict_sktime(
    model_id: str,
    model_dir: Path,
    manifest: dict,
    dataset_id: str | None,
    params: dict,
    started: float,
) -> dict:
    """Evaluate a persisted sktime model on a dataset's holdout split."""
    task = manifest.get("task")
    train_spec = manifest.get("spec") or {}
    dataset_id = dataset_id or train_spec.get("dataset_id")
    dataset = get_dataset(dataset_id) if dataset_id else None
    if dataset is None or not dataset.get("enabled"):
        raise PlaygroundError(f"Dataset is not enabled: {dataset_id}")
    if dataset.get("task") != task:
        raise PlaygroundError(
            f"Dataset `{dataset['id']}` is a {dataset.get('task')} dataset; "
            f"model `{model_id}` was trained for {task}."
        )
    preprocessor = get_enabled_preprocessor(train_spec.get("preprocessor_id"))
    if preprocessor is None:
        preprocessor = get_enabled_preprocessor("none")
    eval_params = {**(manifest.get("eval_params") or {}), **params}

    display_name = f"{manifest.get('algorithm_name') or model_id} (trained: {model_id})"
    log = [
        f"Selected dataset: {dataset['name']}",
        f"Selected preprocessor: {preprocessor['name']}",
        f"Selected model: {model_id} (sktime backend, algorithm: "
        f"{manifest.get('algorithm_id')}, created: {manifest.get('created_at')})",
    ]
    try:
        if manifest.get("multiseries"):
            models = persistence.load_sktime_series_models(model_dir, manifest)
            result, predict_meta = _predict_multiseries_payload(
                models, dataset, display_name, log
            )
        else:
            estimator = persistence.load_sktime_model(model_dir)
            prep_est = persistence.load_sktime_preprocessor(model_dir)
            result, predict_meta = _predict_sktime_payload(
                estimator,
                prep_est,
                preprocessor.get("name", "preprocessor"),
                task,
                dataset,
                eval_params,
                display_name,
                log,
            )
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(
            f"Prediction failed: {type(exc).__name__}: {exc}"
        ) from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    run_id = uuid.uuid4().hex[:12]
    spec_out = {
        "task": task,
        "dataset_id": dataset["id"],
        "algorithm_id": f"sktime-trained:{model_id}",
        "preprocessor_id": preprocessor.get("id", "none"),
        "params": params,
    }
    algorithm_entry = {
        "id": spec_out["algorithm_id"],
        "name": display_name,
        "task": task,
        "module": manifest.get("module"),
        "curated": False,
        "trained": True,
        "backend": "sktime",
        "model_dir": str(model_dir),
        "manifest": manifest,
    }
    result.update(
        {
            "run_id": run_id,
            "spec": spec_out,
            "task": task,
            "dataset": dataset,
            "algorithm": algorithm_entry,
            "preprocessor": preprocessor,
            "duration_ms": elapsed_ms,
            "log": log + [f"Finished in {elapsed_ms} ms"],
        }
    )
    result["code"] = _predict_script(model_id, task, dataset, predict_meta)
    result["report"] = generate_report(result)
    return result


def _predict_sktime_payload(
    estimator,
    prep_est,
    prep_name: str,
    task: str,
    dataset: dict,
    eval_params: dict,
    display_name: str,
    log: list[str],
) -> tuple[dict, dict]:
    """Run the loaded estimator on the dataset holdout; return (payload, meta)."""
    if task == "forecasting":
        import pandas as pd

        horizon = max(1, int(eval_params.get("horizon") or 12))
        if str(eval_params.get("eval_mode") or "single") == "rolling":
            from runners import (
                _load_forecasting_frame,
                _rolling_forecast_result,
                _rolling_split,
            )

            if prep_est is not None:
                raise PlaygroundError(
                    "rolling evaluation does not support preprocessors yet"
                )
            if not hasattr(estimator, "predict_windows"):
                raise PlaygroundError(
                    "This forecaster does not support rolling evaluation (needs "
                    "refit-free windowed prediction; tslib adapters provide it)."
                )
            y = _load_forecasting_frame(dataset, log)
            train_end, test_start, test_end = _rolling_split(len(y), horizon, eval_params)
            payload = _rolling_forecast_result(
                estimator,
                display_name,
                y,
                train_end,
                test_start,
                test_end,
                horizon,
                log,
            )
            payload.pop("_metric_context", None)
            return payload, {"horizon": horizon, "eval_mode": "rolling"}
        y_train, y_test, horizon = _forecast_split(dataset, horizon, log)
        if prep_est is not None:
            y_train = _transform_series(prep_est, prep_name, y_train, log)
            y_test = _transform_series(prep_est, prep_name, y_test, log)
        y_pred = estimator.predict(fh=list(range(1, len(y_test) + 1)))
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(np.asarray(y_pred, dtype=float).ravel())
        y_pred.index = y_test.index
        payload = _forecast_payload(
            y_train,
            y_test,
            y_pred,
            f"Forecasted {len(y_test)} steps with {display_name}.",
        )
        return payload, {"horizon": int(len(y_test))}
    if task == "clustering" and str(eval_params.get("fit_on") or "") == "all":
        from runners import _concat_panels

        X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
        if prep_est is not None:
            X_train = _transform_panel(prep_est, prep_name, X_train, log)
            X_test = _transform_panel(prep_est, prep_name, X_test, log)
        X_all = _concat_panels(X_train, X_test)
        y_all = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
        y_pred = getattr(estimator, "labels_", None)
        if y_pred is None:
            y_pred = estimator.predict(X_all)
        payload = _clustering_payload(
            y_all,
            y_pred,
            f"Clustered all {len(y_all)} series with {display_name}.",
        )
        return payload, {}
    if task in ("classification", "regression", "clustering"):
        _X_train, _y_train, X_test, y_test = _load_panel_xy(dataset, log)
        if prep_est is not None:
            X_test = _transform_panel(prep_est, prep_name, X_test, log)
        if task == "clustering" and not hasattr(estimator, "predict"):
            y_pred = estimator.fit_predict(X_test)
        else:
            y_pred = estimator.predict(X_test)
        if task == "classification":
            payload = _classification_payload(
                y_test,
                y_pred,
                f"Classified {len(y_test)} held-out time series with {display_name}.",
            )
        elif task == "regression":
            payload = _regression_payload(
                y_test,
                y_pred,
                f"Regressed {len(y_test)} held-out time series targets "
                f"with {display_name}.",
            )
        else:
            payload = _clustering_payload(
                y_test,
                y_pred,
                f"Clustered {len(y_test)} held-out time series with {display_name}.",
            )
        return payload, {}
    if task == "anomaly_detection":
        raw, y_true = _load_anomaly_frame(dataset, log)
        if prep_est is not None:
            raw = _transform_series(prep_est, prep_name, raw, log)
        sparse = estimator.predict(raw.to_frame("data"))
        arr = np.asarray(sparse).ravel() if sparse is not None else np.array([])
        if arr.size == len(raw) and arr.size > 0:
            pred_indices = np.where(arr != 0)[0]
        else:
            pred_indices = _extract_sparse_ilocs(sparse)
        return build_anomaly_result(raw, y_true, pred_indices, display_name), {}
    if task == "causal":
        from domain_runners import _causal_payload, _load_causal_dataset, graph_metrics

        X, true_edges = _load_causal_dataset(dataset, log)
        variable_names = [
            str(v) for v in getattr(estimator, "variable_names_", X.columns)
        ]
        metrics = graph_metrics(
            estimator.get_adjacency_matrix(), variable_names, true_edges
        )
        graph = _causal_payload(estimator, variable_names, true_edges)
        matched = {
            (edge["source"], edge["target"])
            for edge in graph["edges"]
            if edge.get("in_true_graph")
        }
        true_rows = [
            {
                "source": str(s),
                "target": str(t),
                "found": (str(s), str(t)) in matched
                or any(
                    e["source"] == str(t)
                    and e["target"] == str(s)
                    and e["type"] == "undirected"
                    for e in graph["edges"]
                ),
            }
            for s, t in true_edges
        ]
        payload = {
            "status": "ok",
            "metrics": metrics,
            "graph": graph,
            "series": {
                "kind": "causal_graph",
                "nodes": graph["variable_names"],
                "edges": graph["edges"],
                "meta": {
                    "graph_type": graph["graph_type"],
                    "n_vars": len(graph["variable_names"]),
                },
            },
            "tables": {"edges": graph["edges"][:200], "true_edges": true_rows},
            "summary": (
                f"Discovered {metrics['Edges']} edges ({metrics['True Edges']} true) "
                f"with {display_name}: SHD={metrics['SHD']}, "
                f"edge F1={metrics['Edge F1']:.3f}."
            ),
        }
        return payload, {}
    raise PlaygroundError(f"Unknown task in manifest: {task}")


def _predict_multiseries_payload(models: dict, dataset: dict, display_name: str, log: list[str]):
    """Evaluate per-series persisted detectors; dataset-level averaged metrics.

    Mirrors the one-shot `_run_anomaly_multiseries` protocol (TSB-UAD), but
    with no refit: each series is scored by its own persisted detector.
    """
    import pandas as pd
    from metrics import resolve_metrics
    from runners import (
        _anomaly_series_row,
        multiseries_anomaly_payload,
        resolve_series_dir,
    )

    series_dir = resolve_series_dir(dataset)
    files = sorted(series_dir.glob("*.out"))
    if not files:
        raise PlaygroundError(f"No .out series files under {series_dir}")
    auc_entry = resolve_metrics(["auc_roc"], "anomaly_detection")[0]
    per_series = []
    missing = 0
    for i, path in enumerate(files):
        detector = models.get(path.name)
        if detector is None:
            missing += 1
            continue
        frame = pd.read_csv(path, header=None, names=["data", "label"])
        raw = frame["data"].astype(float)
        y_true = frame["label"].astype(int).to_numpy()
        sparse = detector.predict(raw.to_frame("data"))
        per_series.append(
            _anomaly_series_row(detector, sparse, raw, y_true, auc_entry, path.name)
        )
        if (i + 1) % 50 == 0:
            log.append(f"Scored {i + 1}/{len(files)} series")
    if missing:
        log.append(f"Skipped {missing} series without a persisted model")
    if not per_series:
        raise PlaygroundError(
            "No persisted series models match this dataset's series files."
        )
    payload = multiseries_anomaly_payload(per_series, display_name, log)
    payload.pop("_metric_context", None)
    return payload, {"multiseries": True}


def _fit_sktime_estimator(
    estimator,
    prep_est,
    prep_name: str,
    task: str,
    dataset: dict,
    eval_params: dict,
    log: list[str],
) -> tuple[dict, dict]:
    """Fit estimator (+ optional preprocessor); return (fit_info, eval_params).

    Multi-series anomaly datasets return the per-series fitted detectors in
    ``fit_info["_multiseries"]`` (handled by `_train_sktime`, which persists
    them through `persistence.save_sktime_multiseries_model`).
    """
    if task == "forecasting":
        horizon = max(1, int(eval_params.get("horizon") or 12))
        if str(eval_params.get("eval_mode") or "single") == "rolling":
            from runners import _load_forecasting_frame, _rolling_split

            if prep_est is not None:
                raise PlaygroundError(
                    "rolling evaluation does not support preprocessors yet"
                )
            if not hasattr(estimator, "predict_windows"):
                raise PlaygroundError(
                    "This forecaster does not support rolling evaluation (needs "
                    "refit-free windowed prediction; tslib adapters provide it). "
                    "Use the default single-origin eval instead."
                )
            y = _load_forecasting_frame(dataset, log)
            train_end, _, _ = _rolling_split(len(y), horizon, eval_params)
            estimator.fit(y.iloc[:train_end])
            log.append(f"Rolling train: fit on {train_end} rows")
            return {"train_length": int(train_end)}, {**eval_params, "horizon": horizon}
        y_train, _y_test, horizon = _forecast_split(dataset, horizon, log)
        if prep_est is not None:
            y_train, _ = _fit_apply_series_preprocessor(
                prep_est, prep_name, y_train, None, log
            )
        estimator.fit(y_train)
        # Persist the post-adjustment horizon so predict reproduces the split.
        return {"train_length": int(len(y_train))}, {**eval_params, "horizon": horizon}
    if task in ("classification", "regression"):
        X_train, y_train, _X_test, _y_test = _load_panel_xy(dataset, log)
        if prep_est is not None:
            X_train, _ = _fit_apply_panel_preprocessor(
                prep_est, prep_name, X_train, None, log
            )
        estimator.fit(X_train, y_train)
        return {"train_instances": int(len(y_train))}, eval_params
    if task == "clustering":
        X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
        if prep_est is not None:
            X_train, X_test = _fit_apply_panel_preprocessor(
                prep_est, prep_name, X_train, X_test, log
            )
        if str(eval_params.get("fit_on") or "") == "all":
            # Unsupervised clustering papers evaluate on the fused train+test
            # set — mirror the `labts run` protocol exactly.
            from runners import _concat_panels

            X_all = _concat_panels(X_train, X_test)
            estimator.fit(X_all)
            log.append(f"fit_on=all: fused train+test into {len(X_all)} series")
            return {"train_instances": int(len(y_train) + len(y_test))}, eval_params
        estimator.fit(X_train)
        return {"train_instances": int(len(y_train))}, eval_params
    if task == "anomaly_detection":
        if dataset.get("series_dir"):
            import pandas as pd
            from runners import resolve_series_dir

            if prep_est is not None:
                raise PlaygroundError(
                    "multi-series training does not support preprocessors yet"
                )
            series_dir = resolve_series_dir(dataset)
            files = sorted(series_dir.glob("*.out"))
            if not files:
                raise PlaygroundError(f"No .out series files under {series_dir}")
            models = {}
            for i, path in enumerate(files):
                frame = pd.read_csv(path, header=None, names=["data", "label"])
                raw = frame["data"].astype(float)
                detector = (
                    estimator.clone() if hasattr(estimator, "clone") else estimator
                )
                detector.fit(raw.to_frame("data"))
                models[path.name] = detector
                if (i + 1) % 50 == 0:
                    log.append(f"Fitted {i + 1}/{len(files)} series")
            log.append(f"Fitted per-series detectors on {len(models)} series")
            return {
                "train_length": int(len(models)),
                "_multiseries": models,
            }, eval_params
        raw, _y_true = _load_anomaly_frame(dataset, log)
        if prep_est is not None:
            raw, _ = _fit_apply_series_preprocessor(prep_est, prep_name, raw, None, log)
        estimator.fit(raw.to_frame("data"))
        return {"train_length": int(len(raw))}, eval_params
    if task == "causal":
        from domain_runners import _load_causal_dataset, _subsample

        if prep_est is not None:
            raise PlaygroundError("causal discovery does not support preprocessors")
        X, _true_edges = _load_causal_dataset(dataset, log)
        X = _subsample(
            X,
            int(eval_params.get("max_samples") or 2000),
            int(eval_params.get("seed") or 7),
            log,
        )
        estimator.fit(X)
        return {"train_instances": int(len(X))}, eval_params
    raise PlaygroundError(f"Unknown task: {task}")


def _build_train_estimator(algorithm: dict, est_params: dict):
    """Instantiate the train-time estimator for a catalog algorithm.

    Mirrors the construction semantics of `labts run`: curated algorithms
    map to the same estimators as their curated runners; registered and
    user algorithms are constructed from their catalog module with params
    coerced to the advertised defaults' types.
    """
    if algorithm.get("curated"):
        return _build_curated_estimator(algorithm, est_params)
    return _build_module_estimator(algorithm, est_params)


def _build_module_estimator(algorithm: dict, est_params: dict):
    """Build an estimator from its catalog module path (run-equivalent)."""
    klass = import_estimator_class(algorithm["module"])
    defaults = algorithm.get("params") or {}
    if algorithm.get("user"):
        eval_keys = EVAL_PARAMS.get(algorithm.get("task"), set())
        est_params = {
            **{k: v for k, v in defaults.items() if k not in eval_keys},
            **est_params,
        }
    coerced = {}
    for key, value in est_params.items():
        default = defaults.get(key)
        if isinstance(default, bool):
            coerced[key] = bool(value)
        elif isinstance(default, int):
            coerced[key] = int(value)
        elif isinstance(default, float):
            coerced[key] = float(value)
        else:
            coerced[key] = value
    return klass(**coerced)


def _build_curated_estimator(algorithm: dict, est_params: dict):
    """Construct the estimator behind a curated catalog algorithm.

    Uses the same classes and defaults as the curated `labts run` runners.
    The curated anomaly detector is rejected: its `run` pipeline detrends
    and standardizes the series outside the estimator, so a persisted
    estimator alone could not reproduce it.
    """
    algorithm_id = algorithm["id"]
    if algorithm_id == "naive-seasonal-last":
        from sktime.forecasting.naive import NaiveForecaster

        seasonal_period = max(1, int(est_params.get("seasonal_period") or 12))
        return NaiveForecaster(strategy="last", sp=seasonal_period)
    if algorithm_id == "summary-random-forest":
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.feature_based import SummaryClassifier

        n_estimators = int(est_params.get("n_estimators") or 25)
        random_state = int(est_params.get("random_state") or 7)
        return SummaryClassifier(
            estimator=RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state
            ),
            random_state=random_state,
        )
    if algorithm_id == "summary-random-forest-regressor":
        from sklearn.ensemble import RandomForestRegressor
        from sktime.regression.compose import SklearnRegressorPipeline
        from sktime.transformations.series.summarize import SummaryTransformer

        n_estimators = int(est_params.get("n_estimators") or 25)
        random_state = int(est_params.get("random_state") or 7)
        return SklearnRegressorPipeline(
            regressor=RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state
            ),
            transformers=[SummaryTransformer()],
        )
    if algorithm_id == "ts-kmeans":
        from sktime.clustering.k_means import TimeSeriesKMeans

        return TimeSeriesKMeans(
            n_clusters=int(est_params.get("n_clusters") or 2),
            random_state=int(est_params.get("random_state") or 7),
        )
    if algorithm.get("task") == "causal":
        # Curated causal entries run their module estimator directly (same as
        # `run_causal`), so the module-based build is run-equivalent.
        return _build_module_estimator(algorithm, est_params)
    raise PlaygroundError(
        f"Curated algorithm `{algorithm_id}` has no persistent train pipeline: "
        "its `labts run` path embeds ad-hoc preprocessing that is not part of "
        "the estimator. Use a registered-* detector instead "
        "(see `labts.py ls algorithms --task anomaly_detection`)."
    )


def _build_train_preprocessor(preprocessor: dict, spec: dict):
    """Instantiate the train spec's preprocessor, or None for the identity."""
    if (
        not preprocessor
        or preprocessor.get("id") == "none"
        or not preprocessor.get("module")
    ):
        return None
    klass = import_estimator_class(preprocessor["module"])
    defaults = preprocessor.get("params") or {}
    params = spec.get("preprocessor_params") or {}
    coerced = {}
    for key, value in params.items():
        default = defaults.get(key)
        if isinstance(default, bool):
            coerced[key] = bool(value)
        elif isinstance(default, int):
            coerced[key] = int(value)
        elif isinstance(default, float):
            coerced[key] = float(value)
        else:
            coerced[key] = value
    return klass(**coerced)


def _load_forecasting_series(dataset: dict, log: list[str]):
    if dataset["source"] == "huggingface":
        from hf_data import load_hf_series

        y = load_hf_series(dataset["hf_config"])
        log.append(f"Loaded Hugging Face config {dataset['hf_config']} ({len(y)} rows)")
        return y
    from sktime.datasets import load_airline, load_lynx, load_shampoo_sales

    loaders = {
        "airline": load_airline,
        "shampoo-sales": load_shampoo_sales,
        "lynx": load_lynx,
    }
    y = loaders[dataset["id"]]()
    log.append(f"Loaded local forecasting dataset {dataset['name']} ({len(y)} rows)")
    return y


def _load_panel_xy(dataset: dict, log: list[str]):
    """Load a panel (X_train, y_train, X_test, y_test) for panel tasks."""
    if dataset.get("source") == "ucr_uea":
        from sktime.datasets import load_UCR_UEA_dataset

        name = dataset["ucr_name"]
        X_train, y_train = load_UCR_UEA_dataset(
            name=name, split="TRAIN", return_X_y=True
        )
        X_test, y_test = load_UCR_UEA_dataset(name=name, split="TEST", return_X_y=True)
        log.append(f"Loaded UCR/UEA {name} train={len(y_train)} test={len(y_test)}")
        return X_train, y_train, X_test, y_test

    loader_path = dataset.get("loader")
    if not loader_path:
        raise PlaygroundError(f"Dataset {dataset['id']} has no panel loader.")
    loader = import_estimator_class(loader_path)
    X_train, y_train = loader(split="train", return_X_y=True)
    X_test, y_test = loader(split="test", return_X_y=True)
    log.append(f"Loaded {dataset['name']} train={len(y_train)} test={len(y_test)}")
    return X_train, y_train, X_test, y_test


def _load_anomaly_frame(dataset: dict, log: list[str]):
    """Load (values, labels) of an anomaly_detection dataset, unpreprocessed."""
    import pandas as pd

    frame = pd.read_csv(REPO_ROOT / dataset["path"])
    y_true = frame["label"].astype(int).to_numpy()
    raw = frame["data"].astype(float)
    log.append(f"Loaded {dataset['name']} rows={len(raw)}")
    return raw, y_true


def _forecast_split(dataset: dict, horizon: int, log: list[str]):
    """Load the task series and split off a `horizon`-long holdout tail."""
    import pandas as pd

    y = pd.Series(_load_forecasting_series(dataset, log)).dropna()
    if (
        hasattr(y.index, "freq")
        and y.index.freq is None
        and not isinstance(y.index, pd.RangeIndex)
    ):
        y.index = pd.RangeIndex(start=0, stop=len(y), step=1)
        log.append("Normalized time index to RangeIndex for reproducible forecasting")
    horizon = max(1, int(horizon or 12))
    if len(y) <= horizon + 1:
        horizon = max(1, len(y) // 4)
        log.append(f"Adjusted horizon to {horizon} for short series")
    return y.iloc[:-horizon], y.iloc[-horizon:], horizon


def _coerce_series(value, index, name: str):
    import pandas as pd

    try:
        if isinstance(value, pd.Series):
            return value
        if isinstance(value, pd.DataFrame):
            if value.shape[1] < 1:
                raise ValueError("Preprocessor returned an empty DataFrame")
            return value.iloc[:, 0]
        arr = np.asarray(value).ravel()
        if len(arr) != len(index):
            raise ValueError(
                f"Preprocessor changed series length from {len(index)} to {len(arr)}"
            )
        return pd.Series(arr, index=index)
    except (ValueError, TypeError) as exc:
        raise PlaygroundError(
            f"Preprocessor `{name}` changed the series shape/length and cannot be used "
            f"in this pipeline: {exc}. Choose a length-preserving transformer."
        ) from exc


def _fit_apply_series_preprocessor(prep, name: str, y_train, y_test, log: list[str]):
    """Fit `prep` on y_train and transform train/test (length-preserving)."""
    try:
        if hasattr(prep, "fit_transform"):
            y_train_t = prep.fit_transform(y_train)
        else:
            y_train_t = prep.fit(y_train).transform(y_train)
        y_test_t = y_test if y_test is None else prep.transform(y_test)
    except Exception as exc:
        raise PlaygroundError(
            f"Preprocessor `{name}` cannot be applied to this univariate series: "
            f"{type(exc).__name__}: {exc}. Pick a series-to-series transformer "
            f"(e.g. Detrender, Deseasonalizer, BoxCox, Log, Imputer)."
        ) from exc
    y_train_t = _coerce_series(y_train_t, y_train.index, name)
    if y_test is not None:
        y_test_t = _coerce_series(y_test_t, y_test.index, name)
    log.append(f"Applied preprocessor: {name}")
    return y_train_t, y_test_t


def _fit_apply_panel_preprocessor(prep, name: str, X_train, X_test, log: list[str]):
    """Fit `prep` on X_train and transform train/test (instance-preserving)."""
    try:
        if hasattr(prep, "fit_transform"):
            X_train_t = prep.fit_transform(X_train)
        else:
            X_train_t = prep.fit(X_train).transform(X_train)
        X_test_t = X_test if X_test is None else prep.transform(X_test)
    except Exception as exc:
        raise PlaygroundError(
            f"Preprocessor `{name}` cannot be applied to this panel: "
            f"{type(exc).__name__}: {exc}. Pick a panel-to-panel transformer."
        ) from exc
    try:
        if len(X_train_t) != len(X_train) or (
            X_test is not None and len(X_test_t) != len(X_test)
        ):
            raise PlaygroundError(
                f"Preprocessor `{name}` changed the number of instances "
                f"({len(X_train)} -> {len(X_train_t)}); "
                "choose an instance-preserving transformer."
            )
    except TypeError:
        pass
    log.append(f"Applied preprocessor: {name}")
    return X_train_t, X_test_t


def _transform_series(prep, name: str, values, log: list[str]):
    """Predict-time series transform with the persisted (fitted) preprocessor."""
    try:
        transformed = prep.transform(values)
    except Exception as exc:
        raise PlaygroundError(
            f"Persisted preprocessor `{name}` cannot be applied to this series: "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    log.append(f"Applied persisted preprocessor: {name}")
    return _coerce_series(transformed, values.index, name)


def _transform_panel(prep, name: str, X, log: list[str]):
    """Predict-time panel transform with the persisted (fitted) preprocessor."""
    try:
        transformed = prep.transform(X)
    except Exception as exc:
        raise PlaygroundError(
            f"Persisted preprocessor `{name}` cannot be applied to this panel: "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    log.append(f"Applied persisted preprocessor: {name}")
    return transformed


def _extract_sparse_ilocs(sparse):
    import pandas as pd

    if sparse is None:
        return np.array([], dtype=int)
    if isinstance(sparse, pd.DataFrame):
        if "ilocs" in sparse.columns:
            values = sparse["ilocs"].to_list()
        else:
            values = sparse.iloc[:, 0].to_list()
    elif isinstance(sparse, pd.Series):
        values = sparse.to_list()
    else:
        values = list(sparse)
    clean = []
    for value in values:
        if hasattr(value, "left") and hasattr(value, "right"):
            clean.extend(range(int(value.left), int(value.right)))
        else:
            clean.append(int(value))
    return np.array(clean, dtype=int)


def _forecast_payload(y_train, y_test, y_pred, summary: str) -> dict:
    """Metrics/series/tables payload for forecasting predictions."""
    import pandas as pd
    from sktime.performance_metrics.forecasting import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )

    y = pd.concat([y_train, y_test]).sort_index()
    residual = (y_test - y_pred).astype(float)
    chart = {
        "kind": "forecast",
        "points": [
            {
                "x": _index_to_label(index),
                "actual": _clean_number(y.loc[index]),
                "prediction": _clean_number(y_pred.loc[index])
                if index in y_pred.index
                else None,
                "split": "test" if index in y_test.index else "train",
            }
            for index in y.index
        ],
        "meta": {
            "total": int(len(y)),
            "test_start": int(len(y_train)),
            "horizon": int(len(y_test)),
        },
    }
    return {
        "status": "ok",
        "metrics": {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MSE": float(mean_squared_error(y_test, y_pred)),
            "MAPE": float(mean_absolute_percentage_error(y_test, y_pred)),
        },
        "series": chart,
        "tables": {
            "forecast": [
                {
                    "time": _index_to_label(index),
                    "actual": _clean_number(y_test.loc[index]),
                    "prediction": _clean_number(y_pred.loc[index]),
                    "residual": _clean_number(residual.loc[index]),
                }
                for index in y_test.index
            ],
        },
        "summary": summary,
    }


def _classification_payload(y_test, y_pred, summary: str) -> dict:
    """Metrics/series/tables payload for classification predictions."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labels = sorted({str(x) for x in list(y_test) + list(y_pred)})
    cm = confusion_matrix(
        [str(x) for x in y_test], [str(x) for x in y_pred], labels=labels
    )
    counts = {
        label: {
            "actual": int(sum(str(x) == label for x in y_test)),
            "predicted": int(sum(str(x) == label for x in y_pred)),
        }
        for label in labels
    }
    return {
        "status": "ok",
        "metrics": {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Macro F1": float(f1_score(y_test, y_pred, average="macro")),
        },
        "series": {
            "kind": "classification",
            "points": [
                {"x": i, "actual": str(actual), "prediction": str(pred)}
                for i, (actual, pred) in enumerate(zip(y_test, y_pred))
            ],
        },
        "tables": {
            "predictions": [
                {"row": i, "actual": str(actual), "prediction": str(pred)}
                for i, (actual, pred) in enumerate(zip(y_test[:30], y_pred[:30]))
            ],
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.astype(int).tolist(),
                "class_counts": counts,
            },
        },
        "summary": summary,
    }


def _regression_payload(y_test, y_pred, summary: str) -> dict:
    """Metrics/series/tables payload for regression predictions."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_test, dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_hat))
    rmse = float(mean_squared_error(y_true, y_hat) ** 0.5)
    r2 = float(r2_score(y_true, y_hat)) if len(y_true) > 1 else 0.0
    residual = y_true - y_hat

    return {
        "status": "ok",
        "metrics": {"MAE": mae, "RMSE": rmse, "R²": r2},
        "series": {
            "kind": "regression",
            "points": [
                {"x": i, "actual": _clean_number(a), "prediction": _clean_number(p)}
                for i, (a, p) in enumerate(zip(y_true, y_hat))
            ],
        },
        "tables": {
            "predictions": [
                {
                    "row": i,
                    "actual": _clean_number(a),
                    "prediction": _clean_number(p),
                    "residual": _clean_number(r),
                }
                for i, (a, p, r) in enumerate(
                    zip(y_true[:30], y_hat[:30], residual[:30])
                )
            ],
        },
        "summary": summary,
    }


def _clustering_payload(y_test, y_pred, summary: str) -> dict:
    """Metrics/series/tables payload for clustering predictions."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    y_true = np.asarray([str(x) for x in y_test])
    y_hat = np.asarray(y_pred, dtype=int)
    ari = float(adjusted_rand_score(y_true, y_hat))
    nmi = float(normalized_mutual_info_score(y_true, y_hat))
    clusters, sizes = np.unique(y_hat, return_counts=True)

    return {
        "status": "ok",
        "metrics": {
            "ARI": ari,
            "NMI": nmi,
            "Clusters": int(len(clusters)),
            "Largest Cluster": int(sizes.max()) if len(sizes) else 0,
        },
        "series": {
            "kind": "clustering",
            "points": [
                {"x": i, "cluster": int(c), "actual": str(a)}
                for i, (a, c) in enumerate(zip(y_true, y_hat))
            ],
        },
        "tables": {
            "predictions": [
                {"row": i, "actual": str(a), "cluster": int(c)}
                for i, (a, c) in enumerate(zip(y_true[:30], y_hat[:30]))
            ],
            "cluster_sizes": {
                "labels": [str(c) for c in clusters.tolist()],
                "sizes": sizes.astype(int).tolist(),
            },
        },
        "summary": summary,
    }


def _predict_script(model_id: str, task: str, dataset: dict, meta: dict) -> str:
    """Self-contained reproduction snippet for a trained-model prediction."""
    if meta.get("multiseries"):
        return (
            "from sktime.base import BaseEstimator\n"
            "# one fitted detector per series: series/<NNNN>/model.zip, mapped\n"
            "# to series names in manifest.json\n"
            "est = BaseEstimator.load_from_path("
            f'"playground/models/{model_id}/series/0000/model.zip")\n'
            "# X: single-channel DataFrame of the series to score\n"
            "out = est.predict(X)\n"
            'print("detections:", out)\n'
        )
    load_line = (
        "from sktime.base import BaseEstimator\n"
        "est = BaseEstimator.load_from_path("
        f'"playground/models/{model_id}/model.zip")\n'
    )
    if task == "causal":
        return (
            load_line
            + 'print("variables:", [str(v) for v in est.variable_names_])\n'
            + 'print("adjacency:", est.get_adjacency_matrix())\n'
        )
    if task == "forecasting":
        horizon = int(meta.get("horizon") or 12)
        return (
            load_line
            + f"y_pred = est.predict(fh=list(range(1, {horizon} + 1)))\n"
            + 'print("predictions:", y_pred.to_numpy())\n'
        )
    if task == "anomaly_detection":
        path = dataset.get("path") or "sktime/datasets/data/yahoo/yahoo.csv"
        return (
            "import pandas as pd\n"
            + load_line
            + f'frame = pd.read_csv("{path}")\n'
            + 'out = est.predict(frame["data"].astype(float).to_frame("data"))\n'
            + 'print("detections:", out)\n'
        )
    return (
        load_line
        + "# X_test: held-out panel from the dataset loader used at train time\n"
        + "y_pred = est.predict(X_test)\n"
        + 'print("predictions:", y_pred)\n'
    )


def _index_to_label(index) -> str:
    return str(index)


def _clean_number(value):
    import math

    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 6)


def _resolve_devad_algorithm(algorithm_id: str | None):
    if not algorithm_id:
        raise PlaygroundError(
            "`labts train` requires --algorithm with a DevAD adapter id "
            "(see `labts.py ls algorithms --task anomaly_detection`)."
        )
    algorithm = get_enabled_algorithm(algorithm_id)
    if algorithm is None or not algorithm.get("enabled"):
        raise PlaygroundError(f"Algorithm is not enabled: {algorithm_id}")
    module = algorithm.get("module") or ""
    if not module.startswith(_ADAPTER_MODULE):
        raise PlaygroundError(
            "`labts train` supports DevAD adapters only "
            f"(module {_ADAPTER_MODULE}*); got `{module}`. Non-trainable "
            "detectors do not need a train step — use `labts run` directly."
        )
    return algorithm, import_estimator_class(module)


def _resolve_anomaly_dataset(dataset_id: str | None):
    dataset = get_dataset(dataset_id or "yahoo")
    if dataset is None or not dataset.get("enabled"):
        raise PlaygroundError(f"Dataset is not enabled: {dataset_id}")
    if dataset.get("task") != "anomaly_detection":
        raise PlaygroundError(
            f"Dataset `{dataset['id']}` is a {dataset.get('task')} dataset; "
            "train/detect need an anomaly_detection dataset (see `labts.py ls datasets`)."
        )
    return dataset


def _split_train_params(params: dict) -> tuple[dict, dict]:
    """Split CLI params into adapter-constructor params and DevAD HP overrides."""
    adapter_params = {k: v for k, v in params.items() if k in _ADAPTER_PARAMS}
    hp_params = {
        k: v for k, v in params.items() if k not in _ADAPTER_PARAMS and k not in _EVAL_PARAMS
    }
    return adapter_params, hp_params


def train_devad(spec: dict) -> dict:
    """Train a DevAD model on a catalog dataset and persist it."""
    started = time.perf_counter()
    algorithm, klass = _resolve_devad_algorithm(spec.get("algorithm_id"))
    dataset = _resolve_anomaly_dataset(spec.get("dataset_id"))
    preprocessor = get_enabled_preprocessor(spec.get("preprocessor_id"))
    if preprocessor is None or not preprocessor.get("enabled"):
        raise PlaygroundError(f"Preprocessor is not enabled: {spec.get('preprocessor_id')}")

    params = spec.get("params") or {}
    adapter_params, hp_params = _split_train_params(params)
    explicit = adapter_params.get("params")
    if isinstance(explicit, str):
        explicit = json.loads(explicit)
    adapter_params["params"] = {**hp_params, **(explicit or {})}
    seed = int(adapter_params.pop("seed", 2026))
    device = str(adapter_params.pop("device", "cpu"))
    estimator = klass(**adapter_params)

    family = klass.family
    devad_params = estimator._devad_params()
    model_id = spec.get("model_id") or f"{family}-{dataset['id']}"

    log = [
        f"Selected dataset: {dataset['name']}",
        f"Selected preprocessor: {preprocessor['name']}",
        f"Selected algorithm: {algorithm['name']} (DevAD family: {family})",
        f"DevAD params: {devad_params or 'defaults'}",
        f"Model id: {model_id} (seed={seed}, device={device})",
    ]
    raw, _labels = load_anomaly_series(dataset, preprocessor, spec, log)
    values = raw.to_numpy(dtype=np.float32)

    val_fraction = float(spec.get("val_fraction") or 0.0)
    if not 0.0 <= val_fraction < 0.9:
        raise PlaygroundError(f"val_fraction must be in [0, 0.9), got {val_fraction}")
    x_val = None
    if val_fraction > 0.0:
        split = max(1, int(len(values) * (1.0 - val_fraction)))
        values, x_val = values[:split], values[split:]
        log.append(f"Validation split: train={len(values)} val={len(x_val)}")

    try:
        from sktime.libs.devad.services.model_service import train_model

        with tempfile.TemporaryDirectory(prefix="labts-train-") as tmp:
            tmp = Path(tmp)
            train_path = tmp / "train.npy"
            np.save(train_path, values)
            val_path = None
            if x_val is not None:
                val_path = tmp / "val.npy"
                np.save(val_path, x_val)
            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps({"family": family, "params": devad_params}, indent=2) + "\n",
                encoding="utf-8",
            )
            service_result = train_model(
                model_root=MODELS_ROOT,
                model_id=model_id,
                family=family,
                config_path=config_path,
                x_train=train_path,
                x_val=val_path,
                seed=seed,
                device=device,
                show_progress=False,
            )
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(f"Training failed: {type(exc).__name__}: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    log.append(f"Finished in {elapsed_ms} ms")
    return {
        "status": "ok",
        "model_id": model_id,
        "family": family,
        "params": devad_params,
        "dataset_id": dataset["id"],
        "train_length": int(len(values)),
        "val_length": int(len(x_val)) if x_val is not None else 0,
        "model_dir": service_result["model_dir"],
        "manifest_path": service_result["manifest_path"],
        "log_path": service_result["log_path"],
        "duration_ms": elapsed_ms,
        "log": log,
        "next_steps": [
            f"labts.py detect --model-id {model_id} --dataset {dataset['id']}",
            "labts.py ls models",
        ],
    }


def _detect_script(model_id: str, dataset: dict, threshold_quantile: float, device: str) -> str:
    return (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sktime.detection.adapters.devad import scores_to_point_ilocs\n"
        "from sktime.libs.devad.services.model_service import load_model\n"
        "\n"
        f'model = load_model("playground/models/{model_id}", device="{device}")\n'
        f'frame = pd.read_csv("{dataset["path"]}")\n'
        'values = frame["data"].astype(float).to_numpy()\n'
        "result = model.detect(values)\n"
        f"ilocs = scores_to_point_ilocs(result.scores, result.start_pos, {threshold_quantile!r})\n"
        'y_pred = np.zeros(len(values), dtype=int)\n'
        "y_pred[ilocs] = 1\n"
        'labels = frame["label"].astype(int).to_numpy()\n'
        "tp = int(((labels == 1) & (y_pred == 1)).sum())\n"
        "fp = int(((labels == 0) & (y_pred == 1)).sum())\n"
        "fn = int(((labels == 1) & (y_pred == 0)).sum())\n"
        'print("Precision", tp / (tp + fp) if tp + fp else 0.0)\n'
        'print("Recall", tp / (tp + fn) if tp + fn else 0.0)\n'
        'print("flagged ilocs:", ilocs.tolist())\n'
    )


def detect_devad(spec: dict) -> dict:
    """Run a persisted DevAD model on a dataset; run-compatible result envelope."""
    started = time.perf_counter()
    model_id = spec.get("model_id")
    if not model_id:
        raise PlaygroundError(
            "`labts detect` requires --model-id (see `labts.py ls models`)."
        )
    model_dir = MODELS_ROOT / model_id
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PlaygroundError(
            f"No trained model `{model_id}` under {MODELS_ROOT} "
            "(see `labts.py ls models`)."
        )
    dataset = _resolve_anomaly_dataset(spec.get("dataset_id"))
    preprocessor = get_enabled_preprocessor(spec.get("preprocessor_id"))
    if preprocessor is None or not preprocessor.get("enabled"):
        raise PlaygroundError(f"Preprocessor is not enabled: {spec.get('preprocessor_id')}")

    params = spec.get("params") or {}
    threshold_quantile = float(params.get("threshold_quantile", 0.99))
    device = str(params.get("device", "cpu"))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family = manifest.get("family", model_id)
    display_name = f"{family} (trained: {model_id})"
    log = [
        f"Selected dataset: {dataset['name']}",
        f"Selected preprocessor: {preprocessor['name']}",
        f"Selected model: {model_id} (DevAD family: {family}, "
        f"best_epoch={manifest.get('best_epoch')})",
        f"Threshold quantile: {threshold_quantile}",
    ]

    try:
        from sktime.detection.adapters.devad import scores_to_point_ilocs
        from sktime.libs.devad.services.model_service import load_model

        model = load_model(model_dir, device=device)
        raw, y_true = load_anomaly_series(dataset, preprocessor, spec, log)
        detect_result = model.detect(raw.to_numpy(dtype=np.float32))
        pred_indices = scores_to_point_ilocs(
            detect_result.scores, detect_result.start_pos, threshold_quantile
        )
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(f"Detection failed: {type(exc).__name__}: {exc}") from exc

    result = build_anomaly_result(raw, y_true, pred_indices, display_name)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    run_id = uuid.uuid4().hex[:12]
    spec_out = {
        "task": "anomaly_detection",
        "dataset_id": dataset["id"],
        "algorithm_id": f"devad-trained:{model_id}",
        "preprocessor_id": preprocessor.get("id", "none"),
        "params": params,
    }
    algorithm_entry = {
        "id": spec_out["algorithm_id"],
        "name": display_name,
        "task": "anomaly_detection",
        "subtype": "point_anomaly",
        "module": "sktime.libs.devad",
        "curated": False,
        "trained": True,
        "model_dir": str(model_dir),
        "manifest": manifest,
    }
    result.update(
        {
            "run_id": run_id,
            "spec": spec_out,
            "task": "anomaly_detection",
            "dataset": dataset,
            "algorithm": algorithm_entry,
            "preprocessor": preprocessor,
            "duration_ms": elapsed_ms,
            "log": log + [f"Finished in {elapsed_ms} ms"],
        }
    )
    result["code"] = _detect_script(model_id, dataset, threshold_quantile, device)
    result["report"] = generate_report(result)
    return result
