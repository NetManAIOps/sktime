"""Experiment runners for the TSBox Sandbox Playground."""

from __future__ import annotations

import io
import json
import math
import numpy as np
import textwrap
import time
import traceback
import uuid
from contextlib import redirect_stdout
from catalog import REPO_ROOT, get_dataset, get_enabled_algorithm, get_enabled_preprocessor, import_estimator_class, split_params
from metrics import MetricError, compute_metric, resolve_metrics


RUNS: dict[str, dict] = {}


class PlaygroundError(Exception):
    """Structured user-facing runtime error."""


def run_experiment(spec: dict) -> dict:
    """Validate and run an experiment spec."""
    started = time.perf_counter()
    spec = _normalize_spec(spec)
    algorithm = get_enabled_algorithm(spec["algorithm_id"])
    dataset = get_dataset(spec["dataset_id"])
    chain = _resolve_preprocessor_chain(spec)

    if algorithm is None or not algorithm.get("enabled"):
        raise PlaygroundError(f"Algorithm is not enabled: {spec['algorithm_id']}")
    if dataset is None or not dataset.get("enabled"):
        raise PlaygroundError(f"Dataset is not enabled: {spec['dataset_id']}")
    for step in chain:
        compat = step["entry"].get("compatible_tasks")
        if compat and spec["task"] not in compat and "all" not in compat:
            raise PlaygroundError(
                f"Preprocessor `{step['entry']['name']}` is not compatible with task "
                f"`{spec['task']}`. Compatible tasks: {', '.join(compat)}."
            )
    if algorithm["task"] != spec["task"] or dataset["task"] != spec["task"]:
        raise PlaygroundError("Selected task, algorithm, and dataset are incompatible.")

    preprocessor = chain[0]["entry"] if chain else get_enabled_preprocessor("none")
    log = [
        f"Selected task: {spec['task']}",
        f"Selected dataset: {dataset['name']}",
        f"Selected preprocessor: {_chain_label(chain)}",
        f"Selected algorithm: {algorithm['name']}",
    ]
    try:
        with io.StringIO() as buffer, redirect_stdout(buffer):
            curated = algorithm.get("curated", False)
            if spec["task"] == "forecasting":
                result = (
                    _run_forecasting(spec, dataset, chain, log)
                    if curated
                    else _run_forecasting_generic(spec, dataset, algorithm, chain, log)
                )
            elif spec["task"] == "classification":
                result = (
                    _run_classification(spec, dataset, chain, log)
                    if curated
                    else _run_classification_generic(spec, dataset, algorithm, chain, log)
                )
            elif spec["task"] == "regression":
                result = (
                    _run_regression(spec, dataset, chain, log)
                    if curated
                    else _run_regression_generic(spec, dataset, algorithm, chain, log)
                )
            elif spec["task"] == "clustering":
                result = (
                    _run_clustering(spec, dataset, chain, log)
                    if curated
                    else _run_clustering_generic(spec, dataset, algorithm, chain, log)
                )
            elif spec["task"] == "anomaly_detection":
                result = (
                    _run_anomaly(spec, dataset, chain, log)
                    if curated
                    else _run_anomaly_generic(spec, dataset, algorithm, chain, log)
                )
            elif spec["task"] == "causal":
                from domain_runners import run_causal

                result = run_causal(spec, dataset, algorithm, log)
            else:
                raise PlaygroundError(f"Unknown task: {spec['task']}")
            captured = buffer.getvalue().strip()
            if captured:
                log.append(captured)
    except PlaygroundError:
        raise
    except MetricError as exc:
        raise PlaygroundError(str(exc)) from exc
    except Exception as exc:
        raise PlaygroundError(_dependency_or_trace_error(exc)) from exc

    context = result.pop("_metric_context", {})
    _apply_requested_metrics(result, spec, context)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    run_id = uuid.uuid4().hex[:12]
    result.update(
        {
            "run_id": run_id,
            "spec": spec,
            "task": spec["task"],
            "dataset": dataset,
            "algorithm": algorithm,
            "preprocessor": preprocessor,
            "preprocessors": [step["entry"] for step in chain],
            "duration_ms": elapsed_ms,
            "log": log + [f"Finished in {elapsed_ms} ms"],
        }
    )
    result["code"] = generate_script(result)
    result["report"] = generate_report(result)
    RUNS[run_id] = result
    return result


def _chain_label(chain: list[dict]) -> str:
    if not chain:
        return "Identity / none"
    return " -> ".join(step["entry"]["name"] for step in chain)


def _apply_requested_metrics(result: dict, spec: dict, context: dict) -> None:
    """Compute extra registry metrics requested via spec["metrics"]."""
    metric_ids = spec.get("metrics") or []
    if not metric_ids:
        return
    try:
        entries = resolve_metrics(metric_ids, spec["task"])
        for entry in entries:
            result["metrics"][entry["name"]] = compute_metric(entry, spec["task"], context)
    except MetricError as exc:
        raise PlaygroundError(str(exc)) from exc


def _normalize_spec(spec: dict) -> dict:
    task = spec.get("task") or "forecasting"
    defaults = {
        "forecasting": ("airline", "naive-seasonal-last"),
        "classification": ("unit-test", "summary-random-forest"),
        "regression": ("covid-3month", "summary-random-forest-regressor"),
        "clustering": ("unit-test-cl", "ts-kmeans"),
        "anomaly_detection": ("yahoo", "threshold-detector"),
        "causal": ("causal-sachs", "causal-notears"),
    }
    dataset_id, algorithm_id = defaults.get(task, defaults["forecasting"])
    steps = _normalize_preprocessor_steps(spec)
    preprocessor_id = spec.get("preprocessor_id") or (steps[0]["id"] if steps else "none")
    normalized = {
        "task": task,
        "dataset_id": spec.get("dataset_id") or dataset_id,
        "algorithm_id": spec.get("algorithm_id") or algorithm_id,
        "preprocessor_id": preprocessor_id,
        "preprocessors": steps,
        "params": spec.get("params") or {},
        "preprocessor_params": spec.get("preprocessor_params") or {},
    }
    if spec.get("metrics"):
        normalized["metrics"] = list(spec["metrics"])
    return normalized


def _normalize_preprocessor_steps(spec: dict) -> list[dict]:
    """Normalize the preprocessing config to a list of {"id", "params"} steps.

    Accepts the new `preprocessors` list (entries as ids or objects) and the
    legacy single `preprocessor_id` + `preprocessor_params` pair.
    """
    raw = spec.get("preprocessors")
    if raw:
        steps = []
        for step in raw:
            if isinstance(step, str):
                steps.append({"id": step, "params": {}})
            else:
                steps.append({"id": step.get("id"), "params": step.get("params") or {}})
        return steps
    pid = spec.get("preprocessor_id")
    if pid and pid != "none":
        return [{"id": pid, "params": spec.get("preprocessor_params") or {}}]
    return []


def _resolve_preprocessor_chain(spec: dict) -> list[dict]:
    """Resolve normalized preprocessor steps to catalog entries.

    Returns a list of {"entry": <catalog entry>, "params": {...}} in order.
    """
    chain = []
    for step in spec.get("preprocessors") or []:
        entry = get_enabled_preprocessor(step.get("id"))
        if entry is None or not entry.get("enabled"):
            raise PlaygroundError(f"Preprocessor is not enabled: {step.get('id')}")
        if not entry.get("module"):
            continue  # the "none" identity entry
        chain.append({"entry": entry, "params": step.get("params") or {}})
    return chain


def _naive_benchmark(y_train, horizon: int, sp: int = 1):
    """Seasonal-naive benchmark predictions (tiles the train tail)."""
    tail = np.asarray(y_train, dtype=float)[-max(1, int(sp)):]
    reps = int(np.ceil(horizon / len(tail))) if len(tail) else 1
    return np.tile(tail, max(1, reps))[:horizon]


def _forecasting_context(y_train, y_test, y_pred, sp: int = 1) -> dict:
    return {
        "values": np.asarray(y_test, dtype=float),
        "predictions": np.asarray(y_pred, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "y_pred_benchmark": _naive_benchmark(y_train, len(y_test), sp),
    }


def _run_forecasting(spec: dict, dataset: dict, preprocessor, log: list[str]) -> dict:
    import numpy as np
    import pandas as pd
    from sktime.datasets import load_airline, load_lynx, load_shampoo_sales
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.performance_metrics.forecasting import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )

    params = {"horizon": 12, "seasonal_period": 12, "context_window": 36}
    params.update(spec.get("params") or {})
    horizon = max(1, int(params.get("horizon") or 12))
    seasonal_period = max(1, int(params.get("seasonal_period") or 12))
    context_window = max(1, int(params.get("context_window") or 36))

    if dataset["source"] == "huggingface":
        from hf_data import load_hf_series

        y = load_hf_series(dataset["hf_config"])
        log.append(f"Loaded Hugging Face config {dataset['hf_config']} ({len(y)} rows)")
    else:
        loaders = {
            "airline": load_airline,
            "shampoo-sales": load_shampoo_sales,
            "lynx": load_lynx,
        }
        y = loaders[dataset["id"]]()
        log.append(f"Loaded local forecasting dataset {dataset['name']} ({len(y)} rows)")

    y = pd.Series(y).dropna()
    if dataset["source"] == "huggingface" or (
        hasattr(y.index, "freq") and y.index.freq is None and not isinstance(y.index, pd.RangeIndex)
    ):
        y.index = pd.RangeIndex(start=0, stop=len(y), step=1)
        log.append("Normalized time index to RangeIndex for reproducible forecasting")
    if len(y) <= horizon + seasonal_period:
        horizon = max(1, min(6, len(y) // 4))
        log.append(f"Adjusted horizon to {horizon} for short series")

    y_train = y.iloc[:-horizon]
    y_test = y.iloc[-horizon:]
    y_train, y_test = _apply_series_preprocessor(y_train, y_test, preprocessor, spec, log)
    y = pd.concat([y_train, y_test]).sort_index()
    forecaster = NaiveForecaster(strategy="last", sp=seasonal_period)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=list(range(1, len(y_test) + 1)))
    y_pred.index = y_test.index

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    mape = float(mean_absolute_percentage_error(y_test, y_pred))
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
            "context_window": int(context_window),
            "horizon": int(horizon),
        },
    }
    return {
        "status": "ok",
        "metrics": {"MAE": mae, "MSE": mse, "MAPE": mape},
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
        "summary": f"Forecasted {len(y_test)} steps with seasonal naive baseline.",
        "_metric_context": _forecasting_context(y_train, y_test, y_pred, seasonal_period),
    }


def _run_classification(spec: dict, dataset: dict, preprocessor, log: list[str]) -> dict:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sktime.classification.feature_based import SummaryClassifier
    from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand
    from sktime.datasets._single_problem_loaders import load_unit_test

    params = {"n_estimators": 25, "random_state": 7}
    params.update(spec.get("params") or {})
    X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
    X_train, X_test = _apply_panel_preprocessor(X_train, X_test, preprocessor, spec, log)

    estimator = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        random_state=int(params["random_state"]),
    )
    classifier = SummaryClassifier(estimator=estimator, random_state=int(params["random_state"]))
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    labels = sorted({str(x) for x in list(y_test) + list(y_pred)})
    cm = confusion_matrix([str(x) for x in y_test], [str(x) for x in y_pred], labels=labels)
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    counts = {}
    for label in labels:
        counts[label] = {
            "actual": int(sum(str(x) == label for x in y_test)),
            "predicted": int(sum(str(x) == label for x in y_pred)),
        }

    return {
        "status": "ok",
        "metrics": {"Accuracy": accuracy, "Macro F1": macro_f1},
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
        "summary": f"Classified {len(y_test)} held-out time series.",
        "_metric_context": {"labels": list(y_test), "predictions": list(y_pred)},
    }


def _regression_result(y_test, y_pred, summary: str) -> dict:
    import numpy as np
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
                for i, (a, p, r) in enumerate(zip(y_true[:30], y_hat[:30], residual[:30]))
            ],
        },
        "summary": summary,
        "_metric_context": {"values": y_true, "predictions": y_hat},
    }


def _run_regression(spec: dict, dataset: dict, preprocessor, log: list[str]) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sktime.regression.compose import SklearnRegressorPipeline
    from sktime.transformations.series.summarize import SummaryTransformer

    params = {"n_estimators": 25, "random_state": 7}
    params.update(spec.get("params") or {})
    X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
    X_train, X_test = _apply_panel_preprocessor(X_train, X_test, preprocessor, spec, log)

    regressor = SklearnRegressorPipeline(
        regressor=RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            random_state=int(params["random_state"]),
        ),
        transformers=[SummaryTransformer()],
    )
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    return _regression_result(
        y_test,
        y_pred,
        f"Regressed {len(y_test)} held-out time series targets with summary features + random forest.",
    )


def _clustering_result(y_test, y_pred, summary: str) -> dict:
    import numpy as np
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
        "_metric_context": {"labels": y_true, "predictions": y_hat},
    }


def _run_clustering(spec: dict, dataset: dict, preprocessor, log: list[str]) -> dict:
    from sktime.clustering.k_means import TimeSeriesKMeans

    params = {"n_clusters": 2, "random_state": 7}
    params.update(spec.get("params") or {})
    X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
    X_train, X_test = _apply_panel_preprocessor(X_train, X_test, preprocessor, spec, log)

    clusterer = TimeSeriesKMeans(
        n_clusters=int(params["n_clusters"]),
        random_state=int(params["random_state"]),
    )
    clusterer.fit(X_train)
    y_pred = clusterer.predict(X_test)
    log.append(f"Fitted TimeSeriesKMeans on {len(y_train)} train series")

    return _clustering_result(
        y_test,
        y_pred,
        f"Clustered {len(y_test)} held-out time series into {params['n_clusters']} clusters.",
    )


def _run_anomaly(spec: dict, dataset: dict, preprocessor, log: list[str]) -> dict:
    import numpy as np
    import pandas as pd
    from sktime.detection.naive import ThresholdDetector

    params = {"threshold": 2.0, "window": 24}
    params.update(spec.get("params") or {})
    threshold = float(params["threshold"])
    window = max(2, int(params.get("window") or 24))

    frame = pd.read_csv(REPO_ROOT / dataset["path"])
    y_true = frame["label"].astype(int).to_numpy()
    raw = frame["data"].astype(float)
    raw = _apply_series_preprocessor(raw, None, preprocessor, spec, log)[0]

    # ``ThresholdDetector`` thresholds the values it is handed. The raw series is
    # strongly trending/seasonal (values in the hundreds to thousands), so a fixed
    # threshold on the raw scale is meaningless and barely reacts to the slider.
    # Detrend with a rolling median and standardize the residual first, so the
    # threshold is in units of "standard deviations from the local level" and
    # flags genuine spikes/dips in either direction.
    baseline = raw.rolling(window, center=True, min_periods=1).median()
    residual = raw - baseline
    zscore = (residual - residual.mean()) / (residual.std(ddof=0) or 1.0)

    detector = ThresholdDetector(upper=threshold, lower=-threshold, mode="points")
    sparse = detector.fit_predict(zscore.to_frame("data"))
    pred_indices = _extract_sparse_ilocs(sparse)
    y_pred = np.zeros(len(raw), dtype=int)
    y_pred[pred_indices[(pred_indices >= 0) & (pred_indices < len(raw))]] = 1

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    log.append(
        f"Loaded {dataset['name']} rows={len(raw)} threshold={threshold} window={window}"
    )

    # The z-score magnitude is this detector's natural continuous anomaly score.
    scores = np.abs(zscore.to_numpy(dtype=float))

    max_points = 30000
    stride = max(1, len(raw) // max_points) if len(raw) > max_points else 1
    points = []
    for i in range(0, len(raw), stride):
        points.append(
            {
                "x": i,
                "value": _clean_number(raw.iloc[i]),
                "actual_anomaly": int(y_true[i]),
                "predicted_anomaly": int(y_pred[i]),
            }
        )

    return {
        "status": "ok",
        "metrics": {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Detected": int(y_pred.sum()),
            "Ground Truth": int(y_true.sum()),
        },
        "scores": [_clean_number(s) for s in scores],
        "evaluation": {
            "labels": y_true.astype(int).tolist(),
            "predictions": y_pred.astype(int).tolist(),
        },
        "series": {"kind": "anomaly", "points": points, "meta": {"total": int(len(raw)), "stride": int(stride)}},
        "tables": {
            "detections": [
                {
                    "iloc": int(i),
                    "value": _clean_number(raw.iloc[i]),
                    "ground_truth": int(y_true[i]),
                }
                for i in np.where(y_pred == 1)[0][:40]
            ]
        },
        "summary": f"Detected {int(y_pred.sum())} anomalies against {int(y_true.sum())} labels.",
        "_metric_context": {
            "labels": y_true,
            "predictions": y_pred,
            "scores": scores,
        },
    }


def _extract_sparse_ilocs(sparse):
    import numpy as np
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
    """Load a panel (X_train, y_train, X_test, y_test) for classification,
    regression, and clustering datasets."""
    if dataset.get("source") == "ucr_uea":
        from sktime.datasets import load_UCR_UEA_dataset

        name = dataset["ucr_name"]
        X_train, y_train = load_UCR_UEA_dataset(name=name, split="TRAIN", return_X_y=True)
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


def _coerce_steps(preprocessor_or_chain, spec: dict) -> list[dict]:
    """Normalize the preprocessor argument of the apply helpers to a chain.

    Accepts the new chain form (list of {"entry", "params"}) and the legacy
    single catalog entry (params taken from spec["preprocessor_params"], the
    form trainer.py uses).
    """
    if not preprocessor_or_chain:
        return []
    if isinstance(preprocessor_or_chain, list):
        return [
            step
            for step in preprocessor_or_chain
            if step.get("entry", {}).get("module")
        ]
    entry = preprocessor_or_chain
    if not entry or entry.get("id") == "none" or not entry.get("module"):
        return []
    return [{"entry": entry, "params": (spec or {}).get("preprocessor_params") or {}}]


def _coerce_series_output(value, index):
    import numpy as np
    import pandas as pd

    if isinstance(value, pd.Series):
        return value
    if isinstance(value, pd.DataFrame):
        if value.shape[1] < 1:
            raise ValueError("Preprocessor returned an empty DataFrame")
        return value.iloc[:, 0]
    arr = np.asarray(value).ravel()
    if len(arr) != len(index):
        raise ValueError(f"Preprocessor changed series length from {len(index)} to {len(arr)}")
    return pd.Series(arr, index=index)


def _apply_series_preprocessor(y_train, y_test, preprocessor, spec: dict, log: list[str]):
    """Apply the preprocessing chain to a (train, test) series pair, in order.

    Every step must preserve the series length; a step that does not is
    rejected with a clear error naming the step.
    """
    for position, step in enumerate(_coerce_steps(preprocessor, spec), start=1):
        entry = step["entry"]
        name = entry.get("name", "preprocessor")
        est = _build_estimator(entry, step["params"])
        try:
            if hasattr(est, "fit_transform"):
                y_train_t = est.fit_transform(y_train)
            else:
                y_train_t = est.fit(y_train).transform(y_train)
            y_test_t = y_test
            if y_test is not None:
                y_test_t = est.transform(y_test)
        except Exception as exc:
            raise PlaygroundError(
                f"Preprocessor step {position} `{name}` cannot be applied to this "
                f"univariate series: {type(exc).__name__}: {exc}. Pick a "
                f"series-to-series transformer (e.g. Detrender, Deseasonalizer, "
                f"BoxCox, Log, Imputer)."
            ) from exc
        try:
            y_train_t = _coerce_series_output(y_train_t, y_train.index)
            if y_test is not None:
                y_test_t = _coerce_series_output(y_test_t, y_test.index)
        except (ValueError, TypeError) as exc:
            raise PlaygroundError(
                f"Preprocessor step {position} `{name}` changed the series "
                f"shape/length and cannot be used in this pipeline: {exc}. "
                "Choose a length-preserving transformer."
            ) from exc
        log.append(f"Applied preprocessor step {position}: {name}")
        y_train, y_test = y_train_t, y_test_t
    return y_train, y_test


def _apply_panel_preprocessor(X_train, X_test, preprocessor, spec: dict, log: list[str]):
    """Apply the preprocessing chain to a (train, test) panel pair, in order."""
    for position, step in enumerate(_coerce_steps(preprocessor, spec), start=1):
        entry = step["entry"]
        name = entry.get("name", "preprocessor")
        est = _build_estimator(entry, step["params"])
        try:
            if hasattr(est, "fit_transform"):
                X_train_t = est.fit_transform(X_train)
            else:
                X_train_t = est.fit(X_train).transform(X_train)
            X_test_t = est.transform(X_test)
        except Exception as exc:
            raise PlaygroundError(
                f"Preprocessor step {position} `{name}` cannot be applied to this "
                f"panel: {type(exc).__name__}: {exc}. Pick a panel-to-panel transformer."
            ) from exc
        try:
            if len(X_train_t) != len(X_train) or len(X_test_t) != len(X_test):
                raise PlaygroundError(
                    f"Preprocessor step {position} `{name}` changed the number of "
                    f"instances ({len(X_train)} -> {len(X_train_t)}); choose an "
                    "instance-preserving transformer."
                )
        except TypeError:
            pass
        log.append(f"Applied preprocessor step {position}: {name}")
        X_train, X_test = X_train_t, X_test_t
    return X_train, X_test


_TASK_SCITYPES = {
    "forecasting": {"forecaster", "transformer"},
    "classification": {"classifier", "transformer"},
    "regression": {"regressor", "transformer"},
    "clustering": {"clusterer", "transformer"},
    "anomaly_detection": {"detector", "transformer"},
    "causal": {"causal_discoverer"},
}


def _contains_nested_estimator(value) -> bool:
    if isinstance(value, dict):
        return "estimator" in value or any(
            _contains_nested_estimator(v) for v in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_nested_estimator(v) for v in value)
    return False


def _validate_sub_estimator_scitype(klass: type, parent_task: str | None) -> None:
    """A nested estimator must match the parent task (or be a transformer)."""
    allowed = _TASK_SCITYPES.get(parent_task or "")
    if not allowed:
        return
    try:
        scitype = klass.get_class_tag("object_type")
    except Exception:
        scitype = None
    if isinstance(scitype, (list, tuple)):
        scitype = scitype[0] if scitype else None
    if scitype and scitype not in allowed:
        raise PlaygroundError(
            f"Nested estimator `{klass.__name__}` (scitype `{scitype}`) cannot be "
            f"used inside a `{parent_task}` pipeline (allowed scitypes: "
            f"{', '.join(sorted(allowed))})."
        )


def _build_sub_estimator(sub_spec: dict, parent_algorithm: dict):
    """Build one nested estimator from {"algorithm_id"|"module", "params"}."""
    if not isinstance(sub_spec, dict):
        raise PlaygroundError(
            "Nested estimator entries must be objects like "
            '{"name": "x", "estimator": {"algorithm_id": "...", "params": {...}}}.'
        )
    params = sub_spec.get("params") or {}
    parent_task = parent_algorithm.get("task")
    if sub_spec.get("algorithm_id"):
        sub = get_enabled_algorithm(sub_spec["algorithm_id"])
        if sub is None:
            sub = _curated_entry_for_registered_id(sub_spec["algorithm_id"])
        if sub is None or not sub.get("enabled"):
            raise PlaygroundError(
                f"Nested estimator is not enabled: {sub_spec['algorithm_id']}"
            )
        sub_task = sub.get("task")
        if sub_task not in (parent_task, "all"):
            raise PlaygroundError(
                f"Nested estimator `{sub['name']}` is a `{sub_task}` algorithm and "
                f"cannot be used inside a `{parent_task}` pipeline."
            )
        return _build_estimator(sub, params)
    if sub_spec.get("module"):
        klass = import_estimator_class(sub_spec["module"])
        _validate_sub_estimator_scitype(klass, parent_task)
        return _build_estimator(
            {"module": sub_spec["module"], "name": klass.__name__, "params": {}},
            params,
        )
    raise PlaygroundError(
        "Nested estimator spec needs `algorithm_id` or `module`: "
        f"{json.dumps(sub_spec)[:120]}"
    )


def _curated_entry_for_registered_id(algorithm_id: str) -> dict | None:
    """Resolve `registered-<task>-<Class>` to a curated entry of that class.

    Curated classes are skipped during registry discovery (their curated entry
    takes their place), so the registered-style id does not exist for them —
    map it back to the curated entry so nested specs can reference either.
    """
    if not algorithm_id.startswith("registered-"):
        return None
    from catalog import ENABLED_ALGORITHMS

    rest = algorithm_id[len("registered-"):]
    task, _, class_name = rest.partition("-")
    if not class_name:
        return None
    for entry in ENABLED_ALGORITHMS:
        if entry.get("task") == task and entry.get("class_name") == class_name:
            return entry
    return None


def _resolve_nested_params(value, parent_algorithm: dict):
    """Recursively resolve nested estimator specs inside a param value.

    Contract: a dict with an ``estimator`` key holds a sub-estimator spec
    (``{"algorithm_id"|"module": ..., "params": {...}}``); an optional ``name``
    sibling produces the ``(name, estimator)`` tuple sktime pipelines expect.
    Lists/dicts are walked recursively, so e.g. ``steps``/``forecasters`` take
    lists of such entries.
    """
    if isinstance(value, dict):
        if "estimator" in value:
            est = _build_sub_estimator(value["estimator"], parent_algorithm)
            if value.get("name") is not None:
                return (str(value["name"]), est)
            return est
        return {k: _resolve_nested_params(v, parent_algorithm) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve_nested_params(v, parent_algorithm) for v in value]
    return value


def _build_estimator(algorithm: dict, est_params: dict):
    """Instantiate a discovered estimator, coercing param types to its defaults.

    Param values may contain nested estimator specs (see
    ``_resolve_nested_params``); they are built recursively and task-checked.
    """
    klass = import_estimator_class(algorithm["module"])
    defaults = algorithm.get("params") or {}
    if algorithm.get("user"):
        # plugin PARAMS are the declared defaults; apply them under the
        # run's params (excluding per-task eval params)
        from catalog import EVAL_PARAMS

        eval_keys = EVAL_PARAMS.get(algorithm.get("task"), set())
        est_params = {
            **{k: v for k, v in defaults.items() if k not in eval_keys},
            **est_params,
        }
    coerced = {}
    for key, value in est_params.items():
        default = defaults.get(key)
        if _contains_nested_estimator(value):
            coerced[key] = _resolve_nested_params(value, algorithm)
        elif isinstance(default, bool):
            coerced[key] = bool(value)
        elif isinstance(default, int) and not isinstance(default, bool):
            coerced[key] = int(value)
        elif isinstance(default, float):
            coerced[key] = float(value)
        else:
            coerced[key] = value
    missing = [p for p in (algorithm.get("required_params") or []) if p not in coerced]
    if missing:
        raise PlaygroundError(
            f"Algorithm `{algorithm['name']}` requires constructor params "
            f"{missing}. Pass them in spec.params; sub-estimators use the nested "
            'form {"<param>": [{"name": "x", "estimator": {"algorithm_id": '
            '"registered-<task>-<Class>", "params": {...}}}]} '
            "(see `labts.py ls algorithms --all` for entries marked "
            "required_params/accepts_estimators)."
        )
    return klass(**coerced)


def _run_forecasting_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor, log: list[str]) -> dict:
    import pandas as pd
    from sktime.performance_metrics.forecasting import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )

    eval_params, est_params = split_params("forecasting", spec.get("params") or {})
    horizon = max(1, int(eval_params.get("horizon") or 12))
    context_window = max(1, int(eval_params.get("context_window") or 36))
    forecaster = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    y = pd.Series(_load_forecasting_series(dataset, log)).dropna()
    if hasattr(y.index, "freq") and y.index.freq is None and not isinstance(y.index, pd.RangeIndex):
        y.index = pd.RangeIndex(start=0, stop=len(y), step=1)
        log.append("Normalized time index to RangeIndex for reproducible forecasting")
    if len(y) <= horizon + 1:
        horizon = max(1, len(y) // 4)
        log.append(f"Adjusted horizon to {horizon} for short series")

    y_train = y.iloc[:-horizon]
    y_test = y.iloc[-horizon:]
    y_train, y_test = _apply_series_preprocessor(y_train, y_test, preprocessor, spec, log)
    y = pd.concat([y_train, y_test]).sort_index()
    if algorithm.get("user") and not callable(getattr(forecaster, "get_params", None)):
        # simple plugin contract: fit(y) + predict(steps)
        forecaster.fit(y_train)
        y_pred = pd.Series(
            np.asarray(forecaster.predict(len(y_test)), dtype=float).ravel(),
            index=y_test.index,
        )
    else:
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=list(range(1, len(y_test) + 1)))
        y_pred.index = y_test.index

    residual = (y_test - y_pred).astype(float)
    chart = {
        "kind": "forecast",
        "points": [
            {
                "x": _index_to_label(index),
                "actual": _clean_number(y.loc[index]),
                "prediction": _clean_number(y_pred.loc[index]) if index in y_pred.index else None,
                "split": "test" if index in y_test.index else "train",
            }
            for index in y.index
        ],
        "meta": {
            "total": int(len(y)),
            "test_start": int(len(y_train)),
            "context_window": int(context_window),
            "horizon": int(horizon),
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
        "summary": f"Forecasted {len(y_test)} steps with {algorithm['name']}.",
        "_metric_context": _forecasting_context(y_train, y_test, y_pred),
    }


def _run_classification_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor, log: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    _eval_params, est_params = split_params("classification", spec.get("params") or {})
    classifier = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
    X_train, X_test = _apply_panel_preprocessor(X_train, X_test, preprocessor, spec, log)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    labels = sorted({str(x) for x in list(y_test) + list(y_pred)})
    cm = confusion_matrix([str(x) for x in y_test], [str(x) for x in y_pred], labels=labels)
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
        "summary": f"Classified {len(y_test)} held-out time series with {algorithm['name']}.",
        "_metric_context": {"labels": list(y_test), "predictions": list(y_pred)},
    }


def _run_regression_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor, log: list[str]) -> dict:
    _eval_params, est_params = split_params("regression", spec.get("params") or {})
    regressor = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
    X_train, X_test = _apply_panel_preprocessor(X_train, X_test, preprocessor, spec, log)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    return _regression_result(
        y_test,
        y_pred,
        f"Regressed {len(y_test)} held-out time series targets with {algorithm['name']}.",
    )


def _run_clustering_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor, log: list[str]) -> dict:
    _eval_params, est_params = split_params("clustering", spec.get("params") or {})
    clusterer = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    X_train, y_train, X_test, y_test = _load_panel_xy(dataset, log)
    X_train, X_test = _apply_panel_preprocessor(X_train, X_test, preprocessor, spec, log)
    if hasattr(clusterer, "predict"):
        clusterer.fit(X_train)
        y_pred = clusterer.predict(X_test)
    else:
        y_pred = clusterer.fit_predict(X_test)

    return _clustering_result(
        y_test,
        y_pred,
        f"Clustered {len(y_test)} held-out time series with {algorithm['name']}.",
    )


def load_anomaly_series(dataset: dict, preprocessor, spec: dict, log: list[str]):
    """Load (values, labels) of an anomaly_detection dataset, with preprocessing.

    `preprocessor` accepts a chain (list of {"entry", "params"}) or a single
    catalog entry (legacy trainer.py form).
    """
    import pandas as pd

    frame = pd.read_csv(REPO_ROOT / dataset["path"])
    y_true = frame["label"].astype(int).to_numpy()
    raw = frame["data"].astype(float)
    raw = _apply_series_preprocessor(raw, None, preprocessor, spec, log)[0]
    log.append(f"Loaded {dataset['name']} rows={len(raw)}")
    return raw, y_true


def _extract_anomaly_scores(detector, raw, y_pred) -> np.ndarray:
    """Best-effort continuous anomaly scores for a fitted detector.

    Order: (1) the detector's native ``transform_scores``; (2) DevAD adapters
    (windowed scores + ``start_pos`` offset, padded to full length); (3) PyOD
    adapters (``decision_scores_``); (4) the binary predictions as a fallback
    so score-based metrics always have something to rank.
    """
    X = raw.to_frame("data")
    try:
        scores = detector.transform_scores(X)
        arr = np.asarray(scores, dtype=float).ravel()
        if arr.size == len(raw):
            return arr
    except Exception:
        pass
    model = getattr(detector, "_model", None)
    detect = getattr(model, "detect", None)
    if callable(detect):
        try:
            result = detect(raw.to_numpy(dtype=np.float32))
            s = np.asarray(result.scores, dtype=float).ravel()
            start = int(getattr(result, "start_pos", 0))
            start = min(max(start, 0), len(raw))
            out = np.zeros(len(raw), dtype=float)
            out[:start] = s[0] if s.size else 0.0
            take = min(s.size, len(raw) - start)
            out[start : start + take] = s[:take]
            if start + take < len(raw):
                out[start + take :] = s[take - 1] if take else 0.0
            return out
        except Exception:
            pass
    decision_scores = getattr(getattr(detector, "estimator_", None), "decision_scores_", None)
    if decision_scores is not None:
        arr = np.asarray(decision_scores, dtype=float).ravel()
        if arr.size == len(raw):
            return arr
    return np.asarray(y_pred, dtype=float)


def build_anomaly_result(raw, y_true, pred_indices, detector_name: str, scores=None) -> dict:
    """Shared metrics/series/tables payload for point-anomaly detections."""
    import numpy as np

    y_pred = np.zeros(len(raw), dtype=int)
    pred_indices = pred_indices[(pred_indices >= 0) & (pred_indices < len(raw))]
    y_pred[pred_indices] = 1

    if scores is None:
        scores = y_pred.astype(float)
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size != len(raw):
        scores = y_pred.astype(float)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    max_points = 30000
    stride = max(1, len(raw) // max_points) if len(raw) > max_points else 1
    points = [
        {
            "x": i,
            "value": _clean_number(raw.iloc[i]),
            "actual_anomaly": int(y_true[i]),
            "predicted_anomaly": int(y_pred[i]),
        }
        for i in range(0, len(raw), stride)
    ]
    return {
        "status": "ok",
        "metrics": {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Detected": int(y_pred.sum()),
            "Ground Truth": int(y_true.sum()),
        },
        "scores": [_clean_number(s) for s in scores],
        "evaluation": {
            "labels": y_true.astype(int).tolist(),
            "predictions": y_pred.astype(int).tolist(),
        },
        "series": {"kind": "anomaly", "points": points, "meta": {"total": int(len(raw)), "stride": int(stride)}},
        "tables": {
            "detections": [
                {"iloc": int(i), "value": _clean_number(raw.iloc[i]), "ground_truth": int(y_true[i])}
                for i in np.where(y_pred == 1)[0][:40]
            ]
        },
        "summary": (
            f"Detected {int(y_pred.sum())} anomalies against "
            f"{int(y_true.sum())} labels with {detector_name}."
        ),
        "_metric_context": {
            "labels": y_true,
            "predictions": y_pred,
            "scores": scores,
        },
    }


def _run_anomaly_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor, log: list[str]) -> dict:
    import numpy as np

    _eval_params, est_params = split_params("anomaly_detection", spec.get("params") or {})
    detector = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    raw, y_true = load_anomaly_series(dataset, preprocessor, spec, log)

    sparse = detector.fit_predict(raw.to_frame("data"))
    arr = np.asarray(sparse).ravel() if sparse is not None else np.array([])
    if arr.size == len(raw) and arr.size > 0:
        pred_indices = np.where(arr != 0)[0]
    else:
        pred_indices = _extract_sparse_ilocs(sparse)
    y_pred = np.zeros(len(raw), dtype=int)
    valid = pred_indices[(pred_indices >= 0) & (pred_indices < len(raw))]
    y_pred[valid] = 1
    scores = _extract_anomaly_scores(detector, raw, y_pred)
    return build_anomaly_result(raw, y_true, pred_indices, algorithm["name"], scores=scores)


def get_run(run_id: str) -> dict | None:
    return RUNS.get(run_id)


def evaluate_saved_run(data: dict, metric_ids: list[str]) -> dict:
    """Re-score a saved run result with registry metrics (no re-fit).

    Uses the payloads persisted by `run --out`: continuous ``scores`` plus the
    ``evaluation`` labels/predictions for anomaly runs, the forecast table and
    series points for forecasting, and the (full) series points for
    regression/classification/clustering. Runs saved without those payloads
    (e.g. captured from `--compact` stdout) are rejected with a pointer to
    `labts evaluate --spec`, which re-runs the experiment instead.
    """
    task = data.get("task")
    try:
        entries = resolve_metrics(metric_ids, task)
        context = _saved_run_context(data, task)
        metrics = {}
        for entry in entries:
            metrics[entry["name"]] = compute_metric(entry, task, context)
    except MetricError as exc:
        raise PlaygroundError(str(exc)) from exc
    return {
        "status": "ok",
        "task": task,
        "run_id": data.get("run_id"),
        "dataset_id": (data.get("dataset") or {}).get("id"),
        "algorithm_id": (data.get("algorithm") or {}).get("id"),
        "metric_ids": [entry["id"] for entry in entries],
        "metrics": metrics,
        "source": "saved run payload (no re-fit)",
    }


def _saved_run_context(data: dict, task: str) -> dict:
    def _missing(what: str):
        raise PlaygroundError(
            f"Saved run has no {what}; it was probably captured from `--compact` "
            "stdout. Re-save with `run --out`, or use `labts evaluate --spec` "
            "to re-run the experiment with metrics."
        )

    if task == "anomaly_detection":
        evaluation = data.get("evaluation") or {}
        scores = data.get("scores")
        if not evaluation.get("labels") or scores is None:
            _missing("scores/evaluation payload")
        return {
            "labels": evaluation.get("labels"),
            "predictions": evaluation.get("predictions"),
            "scores": scores,
        }
    if task == "forecasting":
        table = (data.get("tables") or {}).get("forecast")
        if not table:
            _missing("forecast table")
        context = {
            "values": [row["actual"] for row in table],
            "predictions": [row["prediction"] for row in table],
        }
        points = (data.get("series") or {}).get("points") or []
        train = [p["actual"] for p in points if p.get("split") == "train"]
        if train:
            context["y_train"] = train
            context["y_pred_benchmark"] = _naive_benchmark(train, len(table))
        return context
    if task == "regression":
        points = (data.get("series") or {}).get("points") or []
        if not points:
            _missing("series points")
        return {
            "values": [p["actual"] for p in points],
            "predictions": [p["prediction"] for p in points],
        }
    if task in ("classification", "clustering"):
        points = (data.get("series") or {}).get("points") or []
        if not points:
            _missing("series points")
        pred_key = "prediction" if task == "classification" else "cluster"
        return {
            "labels": [p["actual"] for p in points],
            "predictions": [p[pred_key] for p in points],
        }
    raise PlaygroundError(
        f"`labts evaluate --from` is not supported for task `{task}`; "
        "use `labts evaluate --spec` to re-run with metrics."
    )


def generate_script(result: dict) -> str:
    """Generate a self-contained reproduction script for a completed run."""
    spec = result["spec"]
    algorithm = result["algorithm"]
    dataset_id = spec["dataset_id"]
    params = spec.get("params") or {}
    if spec["task"] == "causal":
        from domain_runners import causal_script

        return causal_script(result)
    if algorithm.get("curated"):
        if spec["task"] == "forecasting":
            return _forecasting_script(dataset_id, params)
        if spec["task"] == "classification":
            return _classification_script(dataset_id, params)
        if spec["task"] == "regression":
            return _regression_script(dataset_id, params)
        if spec["task"] == "clustering":
            return _clustering_script(dataset_id, params)
        return _anomaly_script(dataset_id, params)
    return _generic_script(result)


def _render_param_value(value, imports: set) -> str:
    """Render a (possibly nested-spec) param value as Python code."""
    if isinstance(value, dict) and "estimator" in value:
        code = _render_sub_estimator(value["estimator"], imports)
        if value.get("name") is not None:
            return f"({value['name']!r}, {code})"
        return code
    if isinstance(value, (list, tuple)):
        open_, close_ = ("[", "]") if isinstance(value, list) else ("(", ")")
        inner = ", ".join(_render_param_value(v, imports) for v in value)
        if isinstance(value, tuple) and len(value) == 1:
            inner += ","
        return f"{open_}{inner}{close_}"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{k!r}: {_render_param_value(v, imports)}" for k, v in value.items()
        )
        return "{" + inner + "}"
    return repr(value)


def _render_sub_estimator(sub_spec: dict, imports: set) -> str:
    module = sub_spec.get("module")
    if not module and sub_spec.get("algorithm_id"):
        algorithm = get_enabled_algorithm(sub_spec["algorithm_id"]) or _curated_entry_for_registered_id(
            sub_spec["algorithm_id"]
        )
        if algorithm is None:
            return f"# unresolved nested estimator {sub_spec['algorithm_id']!r}"
        module = algorithm["module"]
    if not module:
        return "# unresolved nested estimator"
    module_name, _, class_name = module.rpartition(".")
    imports.add(f"from {module_name} import {class_name}")
    args = ", ".join(
        f"{k}={_render_param_value(v, imports)}"
        for k, v in (sub_spec.get("params") or {}).items()
    )
    return f"{class_name}({args})"


def generate_report(result: dict) -> str:
    metric_lines = "\n".join(
        f"- {name}: {_format_metric(value)}" for name, value in result["metrics"].items()
    )
    return textwrap.dedent(
        f"""\
        # TSBox Sandbox Experiment Report

        - Task: {result['task']}
        - Dataset: {result['dataset']['name']}
        - Algorithm: {result['algorithm']['name']}
        - Duration: {result['duration_ms']} ms

        ## Metrics

        {metric_lines}

        ## Summary

        {result['summary']}
        """
    )


def _forecasting_script(dataset_id: str, params: dict) -> str:
    horizon = int(params.get("horizon", 12))
    sp = int(params.get("seasonal_period", 12))
    return textwrap.dedent(
        f"""\
        from sktime.datasets import load_airline, load_lynx, load_shampoo_sales
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.performance_metrics.forecasting import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

        loaders = {{
            "airline": load_airline,
            "shampoo-sales": load_shampoo_sales,
            "lynx": load_lynx,
        }}
        y = loaders["{dataset_id}"]().dropna()
        horizon = {horizon}
        y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
        forecaster = NaiveForecaster(strategy="last", sp={sp})
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=list(range(1, len(y_test) + 1)))
        y_pred.index = y_test.index
        print("MAE", mean_absolute_error(y_test, y_pred))
        print("MSE", mean_squared_error(y_test, y_pred))
        print("MAPE", mean_absolute_percentage_error(y_test, y_pred))
        print(y_pred)
        """
    )


def _classification_script(dataset_id: str, params: dict) -> str:
    n_estimators = int(params.get("n_estimators", 25))
    random_state = int(params.get("random_state", 7))
    return textwrap.dedent(
        f"""\
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sktime.classification.feature_based import SummaryClassifier
        from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand
        from sktime.datasets._single_problem_loaders import load_unit_test

        loaders = {{
            "unit-test": load_unit_test,
            "arrow-head": load_arrow_head,
            "italy-power-demand": load_italy_power_demand,
            "gunpoint": load_gunpoint,
        }}
        X_train, y_train = loaders["{dataset_id}"](split="train", return_X_y=True)
        X_test, y_test = loaders["{dataset_id}"](split="test", return_X_y=True)
        clf = SummaryClassifier(
            estimator=RandomForestClassifier(n_estimators={n_estimators}, random_state={random_state}),
            random_state={random_state},
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("accuracy", accuracy_score(y_test, y_pred))
        print("macro_f1", f1_score(y_test, y_pred, average="macro"))
        print(y_pred[:20])
        """
    )


def _regression_script(dataset_id: str, params: dict) -> str:
    n_estimators = int(params.get("n_estimators", 25))
    random_state = int(params.get("random_state", 7))
    dataset = get_dataset(dataset_id)
    loader_path = dataset["loader"] if dataset else "sktime.datasets.load_covid_3month"
    module_name, _, loader_name = loader_path.rpartition(".")
    return textwrap.dedent(
        f"""\
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sktime.regression.compose import SklearnRegressorPipeline
        from sktime.transformations.series.summarize import SummaryTransformer
        from {module_name} import {loader_name}

        X_train, y_train = {loader_name}(split="train", return_X_y=True)
        X_test, y_test = {loader_name}(split="test", return_X_y=True)
        reg = SklearnRegressorPipeline(
            regressor=RandomForestRegressor(n_estimators={n_estimators}, random_state={random_state}),
            transformers=[SummaryTransformer()],
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("mae", mean_absolute_error(y_test, y_pred))
        print("rmse", mean_squared_error(y_test, y_pred) ** 0.5)
        print("r2", r2_score(y_test, y_pred))
        print(y_pred[:20])
        """
    )


def _clustering_script(dataset_id: str, params: dict) -> str:
    n_clusters = int(params.get("n_clusters", 2))
    random_state = int(params.get("random_state", 7))
    dataset = get_dataset(dataset_id)
    loader_path = dataset["loader"] if dataset else "sktime.datasets._single_problem_loaders.load_unit_test"
    module_name, _, loader_name = loader_path.rpartition(".")
    return textwrap.dedent(
        f"""\
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from sktime.clustering.k_means import TimeSeriesKMeans
        from {module_name} import {loader_name}

        X_train, y_train = {loader_name}(split="train", return_X_y=True)
        X_test, y_test = {loader_name}(split="test", return_X_y=True)
        clusterer = TimeSeriesKMeans(n_clusters={n_clusters}, random_state={random_state})
        clusterer.fit(X_train)
        y_pred = clusterer.predict(X_test)
        print("ari", adjusted_rand_score(y_test, y_pred))
        print("nmi", normalized_mutual_info_score(y_test, y_pred))
        print(y_pred[:20])
        """
    )


def _anomaly_script(dataset_id: str, params: dict) -> str:
    threshold = float(params.get("threshold", 2.0))
    window = int(params.get("window", 24))
    dataset = get_dataset(dataset_id)
    path = dataset["path"] if dataset else "sktime/datasets/data/yahoo/yahoo.csv"
    return textwrap.dedent(
        f"""\
        import numpy as np
        import pandas as pd
        from sktime.detection.naive import ThresholdDetector

        frame = pd.read_csv("{path}")
        y_true = frame["label"].astype(int).to_numpy()
        raw = frame["data"].astype(float)
        baseline = raw.rolling({window}, center=True, min_periods=1).median()
        residual = raw - baseline
        zscore = (residual - residual.mean()) / (residual.std(ddof=0) or 1.0)
        detector = ThresholdDetector(upper={threshold}, lower=-{threshold}, mode="points")
        sparse = detector.fit_predict(zscore.to_frame("data"))
        pred_indices = sparse["ilocs"].to_numpy(dtype=int) if hasattr(sparse, "columns") and "ilocs" in sparse.columns else np.array(sparse, dtype=int)
        y_pred = np.zeros(len(raw), dtype=int)
        y_pred[pred_indices[(pred_indices >= 0) & (pred_indices < len(raw))]] = 1
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print("precision", precision)
        print("recall", recall)
        print("f1", f1)
        print("detected", int(y_pred.sum()))
        """
    )


def _generic_script(result: dict) -> str:
    spec = result["spec"]
    algorithm = result["algorithm"]
    task = spec["task"]
    dataset_id = spec["dataset_id"]
    eval_params, est_params = split_params(task, spec.get("params") or {})
    module_name, _, class_name = algorithm["module"].rpartition(".")
    if algorithm.get("user"):
        from catalog import EVAL_PARAMS

        eval_keys = EVAL_PARAMS.get(task, set())
        est_params = {
            **{k: v for k, v in (algorithm.get("params") or {}).items() if k not in eval_keys},
            **est_params,
        }
    imports: set = set()
    args = ", ".join(f"{k}={_render_param_value(v, imports)}" for k, v in est_params.items())
    if algorithm.get("user"):
        import_line = (
            "import sys\n"
            "sys.path.insert(0, \"playground\")  # user plugins live in playground/experiments\n"
            f"from {module_name} import {class_name}"
        )
    else:
        import_line = f"from {module_name} import {class_name}"
    if imports:
        import_line = import_line + "\n" + "\n".join(sorted(imports))

    if task == "forecasting":
        horizon = int(eval_params.get("horizon") or 12)
        loaders = {
            "airline": ("load_airline", "sktime.datasets"),
            "shampoo-sales": ("load_shampoo_sales", "sktime.datasets"),
            "lynx": ("load_lynx", "sktime.datasets"),
        }
        if dataset_id in loaders:
            fn, mod = loaders[dataset_id]
            load_line = f"from {mod} import {fn}\ny = {fn}().dropna()"
        else:
            load_line = f"# dataset {dataset_id!r} needs an online/custom loader\ny = None  # TODO: load a pandas Series"
        if algorithm.get("user"):
            _klass = import_estimator_class(algorithm["module"])
            _simple_contract = not callable(getattr(_klass, "get_params", None))
        else:
            _simple_contract = False
        if _simple_contract:
            predict_block = (
                "y_pred = est.predict(len(y_test))\n"
                "import pandas as pd\n"
                "y_pred = pd.Series(y_pred, index=y_test.index)"
            )
        else:
            predict_block = (
                "y_pred = est.predict(fh=list(range(1, len(y_test) + 1)))"
            )
        return (
            f"{import_line}\n"
            f"{load_line}\n"
            f"\n"
            f"horizon = {horizon}\n"
            f"y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]\n"
            f"est = {class_name}({args})\n"
            f"est.fit(y_train)\n"
            f"{predict_block}\n"
            f"print(y_pred)\n"
        )

    if task == "classification":
        loaders = {
            "unit-test": ("load_unit_test", "sktime.datasets._single_problem_loaders"),
            "arrow-head": ("load_arrow_head", "sktime.datasets"),
            "italy-power-demand": ("load_italy_power_demand", "sktime.datasets"),
            "gunpoint": ("load_gunpoint", "sktime.datasets"),
        }
        fn, mod = loaders.get(
            dataset_id, ("load_unit_test", "sktime.datasets._single_problem_loaders")
        )
        return (
            f"{import_line}\n"
            f"from {mod} import {fn}\n"
            f"from sklearn.metrics import accuracy_score, f1_score\n"
            f"\n"
            f"X_train, y_train = {fn}(split=\"train\", return_X_y=True)\n"
            f"X_test, y_test = {fn}(split=\"test\", return_X_y=True)\n"
            f"est = {class_name}({args})\n"
            f"est.fit(X_train, y_train)\n"
            f"y_pred = est.predict(X_test)\n"
            f"print(\"accuracy\", accuracy_score(y_test, y_pred))\n"
            f"print(\"macro_f1\", f1_score(y_test, y_pred, average=\"macro\"))\n"
        )

    if task == "regression":
        dataset = get_dataset(dataset_id)
        loader_path = dataset["loader"] if dataset else "sktime.datasets.load_covid_3month"
        mod, _, fn = loader_path.rpartition(".")
        return (
            f"{import_line}\n"
            f"from {mod} import {fn}\n"
            f"from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
            f"\n"
            f"X_train, y_train = {fn}(split=\"train\", return_X_y=True)\n"
            f"X_test, y_test = {fn}(split=\"test\", return_X_y=True)\n"
            f"est = {class_name}({args})\n"
            f"est.fit(X_train, y_train)\n"
            f"y_pred = est.predict(X_test)\n"
            f"print(\"mae\", mean_absolute_error(y_test, y_pred))\n"
            f"print(\"rmse\", mean_squared_error(y_test, y_pred) ** 0.5)\n"
            f"print(\"r2\", r2_score(y_test, y_pred))\n"
        )

    if task == "clustering":
        dataset = get_dataset(dataset_id)
        loader_path = (
            dataset["loader"]
            if dataset
            else "sktime.datasets._single_problem_loaders.load_unit_test"
        )
        mod, _, fn = loader_path.rpartition(".")
        return (
            f"{import_line}\n"
            f"from {mod} import {fn}\n"
            f"from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score\n"
            f"\n"
            f"X_train, y_train = {fn}(split=\"train\", return_X_y=True)\n"
            f"X_test, y_test = {fn}(split=\"test\", return_X_y=True)\n"
            f"est = {class_name}({args})\n"
            f"est.fit(X_train)\n"
            f"y_pred = est.predict(X_test)\n"
            f"print(\"ari\", adjusted_rand_score(y_test, y_pred))\n"
            f"print(\"nmi\", normalized_mutual_info_score(y_test, y_pred))\n"
        )

    dataset = get_dataset(dataset_id)
    path = dataset["path"] if dataset else "sktime/datasets/data/yahoo/yahoo.csv"
    return (
        f"import numpy as np\n"
        f"import pandas as pd\n"
        f"{import_line}\n"
        f"\n"
        f"frame = pd.read_csv(\"{path}\")\n"
        f"raw = frame[\"data\"].astype(float)\n"
        f"est = {class_name}({args})\n"
        f"out = est.fit_predict(raw.to_frame(\"data\"))\n"
        f"arr = np.asarray(out).ravel()\n"
        f"pred = np.where(arr != 0)[0] if arr.size == len(raw) else np.array(arr, dtype=int)\n"
        f"y_pred = np.zeros(len(raw), dtype=int)\n"
        f"y_pred[pred[(pred >= 0) & (pred < len(raw))]] = 1\n"
        f"print(\"detected\", int(y_pred.sum()))\n"
    )


def _index_to_label(index) -> str:
    return str(index)


def _clean_number(value):
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 6)


def _format_metric(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _dependency_or_trace_error(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError):
        import re

        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
        package = exc.name or (match.group(1) if match else str(exc))
        return (
            f"Missing dependency `{package}`. Install it in the playground environment, "
            f"e.g. `python3 -m pip install {package}` "
            f"(or `uv pip install --python .venv/bin/python {package}`)."
        )
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
