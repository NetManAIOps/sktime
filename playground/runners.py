"""Experiment runners for the TSBox Sandbox Playground."""

from __future__ import annotations

import io
import json
import math
import textwrap
import time
import traceback
import uuid
from contextlib import redirect_stdout
from catalog import REPO_ROOT, get_dataset, get_enabled_algorithm, get_enabled_preprocessor, import_estimator_class, split_params


RUNS: dict[str, dict] = {}


class PlaygroundError(Exception):
    """Structured user-facing runtime error."""


def run_experiment(spec: dict) -> dict:
    """Validate and run an experiment spec."""
    started = time.perf_counter()
    spec = _normalize_spec(spec)
    algorithm = get_enabled_algorithm(spec["algorithm_id"])
    dataset = get_dataset(spec["dataset_id"])
    preprocessor = get_enabled_preprocessor(spec.get("preprocessor_id"))

    if algorithm is None or not algorithm.get("enabled"):
        raise PlaygroundError(f"Algorithm is not enabled: {spec['algorithm_id']}")
    if dataset is None or not dataset.get("enabled"):
        raise PlaygroundError(f"Dataset is not enabled: {spec['dataset_id']}")
    if preprocessor is None or not preprocessor.get("enabled"):
        raise PlaygroundError(f"Preprocessor is not enabled: {spec.get('preprocessor_id')}")
    if algorithm["task"] != spec["task"] or dataset["task"] != spec["task"]:
        raise PlaygroundError("Selected task, algorithm, and dataset are incompatible.")

    log = [
        f"Selected task: {spec['task']}",
        f"Selected dataset: {dataset['name']}",
        f"Selected preprocessor: {preprocessor['name']}",
        f"Selected algorithm: {algorithm['name']}",
    ]
    try:
        with io.StringIO() as buffer, redirect_stdout(buffer):
            curated = algorithm.get("curated", False)
            if spec["task"] == "forecasting":
                result = (
                    _run_forecasting(spec, dataset, preprocessor, log)
                    if curated
                    else _run_forecasting_generic(spec, dataset, algorithm, preprocessor, log)
                )
            elif spec["task"] == "classification":
                result = (
                    _run_classification(spec, dataset, preprocessor, log)
                    if curated
                    else _run_classification_generic(spec, dataset, algorithm, preprocessor, log)
                )
            elif spec["task"] == "anomaly_detection":
                result = (
                    _run_anomaly(spec, dataset, preprocessor, log)
                    if curated
                    else _run_anomaly_generic(spec, dataset, algorithm, preprocessor, log)
                )
            else:
                raise PlaygroundError(f"Unknown task: {spec['task']}")
            captured = buffer.getvalue().strip()
            if captured:
                log.append(captured)
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(_dependency_or_trace_error(exc)) from exc

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
            "duration_ms": elapsed_ms,
            "log": log + [f"Finished in {elapsed_ms} ms"],
        }
    )
    result["code"] = generate_script(result)
    result["report"] = generate_report(result)
    RUNS[run_id] = result
    return result


def _normalize_spec(spec: dict) -> dict:
    task = spec.get("task") or "forecasting"
    defaults = {
        "forecasting": ("airline", "naive-seasonal-last"),
        "classification": ("unit-test", "summary-random-forest"),
        "anomaly_detection": ("yahoo", "threshold-detector"),
    }
    dataset_id, algorithm_id = defaults.get(task, defaults["forecasting"])
    normalized = {
        "task": task,
        "dataset_id": spec.get("dataset_id") or dataset_id,
        "algorithm_id": spec.get("algorithm_id") or algorithm_id,
        "preprocessor_id": spec.get("preprocessor_id") or "none",
        "params": spec.get("params") or {},
        "preprocessor_params": spec.get("preprocessor_params") or {},
    }
    return normalized


def _run_forecasting(spec: dict, dataset: dict, preprocessor: dict, log: list[str]) -> dict:
    import numpy as np
    import pandas as pd
    from sktime.datasets import load_airline, load_lynx, load_shampoo_sales
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.performance_metrics.forecasting import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )

    params = {"horizon": 12, "seasonal_period": 12}
    params.update(spec.get("params") or {})
    horizon = max(1, int(params.get("horizon") or 12))
    seasonal_period = max(1, int(params.get("seasonal_period") or 12))

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
    }


def _run_classification(spec: dict, dataset: dict, preprocessor: dict, log: list[str]) -> dict:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sktime.classification.feature_based import SummaryClassifier
    from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand
    from sktime.datasets._single_problem_loaders import load_unit_test

    params = {"n_estimators": 25, "random_state": 7}
    params.update(spec.get("params") or {})
    X_train, y_train, X_test, y_test = _load_classification_xy(dataset, log)
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
    }


def _run_anomaly(spec: dict, dataset: dict, preprocessor: dict, log: list[str]) -> dict:
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

    stride = max(1, len(raw) // 800)
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
        "series": {"kind": "anomaly", "points": points},
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


def _load_classification_xy(dataset: dict, log: list[str]):
    if dataset.get("source") == "ucr_uea":
        from sktime.datasets import load_UCR_UEA_dataset

        name = dataset["ucr_name"]
        X_train, y_train = load_UCR_UEA_dataset(name=name, split="TRAIN", return_X_y=True)
        X_test, y_test = load_UCR_UEA_dataset(name=name, split="TEST", return_X_y=True)
        log.append(f"Loaded UCR/UEA {name} train={len(y_train)} test={len(y_test)}")
        return X_train, y_train, X_test, y_test

    from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand
    from sktime.datasets._single_problem_loaders import load_unit_test

    loaders = {
        "unit-test": load_unit_test,
        "arrow-head": load_arrow_head,
        "italy-power-demand": load_italy_power_demand,
        "gunpoint": load_gunpoint,
    }
    loader = loaders[dataset["id"]]
    X_train, y_train = loader(split="train", return_X_y=True)
    X_test, y_test = loader(split="test", return_X_y=True)
    log.append(f"Loaded {dataset['name']} train={len(y_train)} test={len(y_test)}")
    return X_train, y_train, X_test, y_test


def _selected_preprocessor(preprocessor: dict, spec: dict):
    if not preprocessor or preprocessor.get("id") == "none" or not preprocessor.get("module"):
        return None
    return _build_estimator(preprocessor, spec.get("preprocessor_params") or {})


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


def _apply_series_preprocessor(y_train, y_test, preprocessor: dict, spec: dict, log: list[str]):
    est = _selected_preprocessor(preprocessor, spec)
    if est is None:
        return y_train, y_test
    if hasattr(est, "fit_transform"):
        y_train_t = est.fit_transform(y_train)
    else:
        y_train_t = est.fit(y_train).transform(y_train)
    y_test_t = y_test
    if y_test is not None:
        y_test_t = est.transform(y_test)
    y_train_t = _coerce_series_output(y_train_t, y_train.index)
    if y_test is not None:
        y_test_t = _coerce_series_output(y_test_t, y_test.index)
    log.append(f"Applied preprocessor: {preprocessor['name']}")
    return y_train_t, y_test_t


def _apply_panel_preprocessor(X_train, X_test, preprocessor: dict, spec: dict, log: list[str]):
    est = _selected_preprocessor(preprocessor, spec)
    if est is None:
        return X_train, X_test
    if hasattr(est, "fit_transform"):
        X_train_t = est.fit_transform(X_train)
    else:
        X_train_t = est.fit(X_train).transform(X_train)
    X_test_t = est.transform(X_test)
    log.append(f"Applied preprocessor: {preprocessor['name']}")
    return X_train_t, X_test_t


def _build_estimator(algorithm: dict, est_params: dict):
    """Instantiate a discovered estimator, coercing param types to its defaults."""
    klass = import_estimator_class(algorithm["module"])
    defaults = algorithm.get("params") or {}
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


def _run_forecasting_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor: dict, log: list[str]) -> dict:
    import pandas as pd
    from sktime.performance_metrics.forecasting import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )

    eval_params, est_params = split_params("forecasting", spec.get("params") or {})
    horizon = max(1, int(eval_params.get("horizon") or 12))
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
    }


def _run_classification_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor: dict, log: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    _eval_params, est_params = split_params("classification", spec.get("params") or {})
    classifier = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    X_train, y_train, X_test, y_test = _load_classification_xy(dataset, log)
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
    }


def _run_anomaly_generic(spec: dict, dataset: dict, algorithm: dict, preprocessor: dict, log: list[str]) -> dict:
    import numpy as np
    import pandas as pd

    _eval_params, est_params = split_params("anomaly_detection", spec.get("params") or {})
    detector = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    frame = pd.read_csv(REPO_ROOT / dataset["path"])
    y_true = frame["label"].astype(int).to_numpy()
    raw = frame["data"].astype(float)
    raw = _apply_series_preprocessor(raw, None, preprocessor, spec, log)[0]

    sparse = detector.fit_predict(raw.to_frame("data"))
    arr = np.asarray(sparse).ravel() if sparse is not None else np.array([])
    if arr.size == len(raw) and arr.size > 0:
        pred_indices = np.where(arr != 0)[0]
    else:
        pred_indices = _extract_sparse_ilocs(sparse)
    y_pred = np.zeros(len(raw), dtype=int)
    y_pred[pred_indices[(pred_indices >= 0) & (pred_indices < len(raw))]] = 1

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    log.append(f"Loaded {dataset['name']} rows={len(raw)}")

    stride = max(1, len(raw) // 800)
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
        "series": {"kind": "anomaly", "points": points},
        "tables": {
            "detections": [
                {"iloc": int(i), "value": _clean_number(raw.iloc[i]), "ground_truth": int(y_true[i])}
                for i in np.where(y_pred == 1)[0][:40]
            ]
        },
        "summary": f"Detected {int(y_pred.sum())} anomalies against {int(y_true.sum())} labels with {algorithm['name']}.",
    }


def get_run(run_id: str) -> dict | None:
    return RUNS.get(run_id)


def generate_script(result: dict) -> str:
    """Generate a self-contained reproduction script for a completed run."""
    spec = result["spec"]
    algorithm = result["algorithm"]
    dataset_id = spec["dataset_id"]
    params = spec.get("params") or {}
    if algorithm.get("curated"):
        if spec["task"] == "forecasting":
            return _forecasting_script(dataset_id, params)
        if spec["task"] == "classification":
            return _classification_script(dataset_id, params)
        return _anomaly_script(dataset_id, params)
    return _generic_script(result)


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
    args = ", ".join(f"{k}={repr(v)}" for k, v in est_params.items())

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
        return textwrap.dedent(
            f"""\
            from {module_name} import {class_name}
            {load_line}

            horizon = {horizon}
            y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
            est = {class_name}({args})
            est.fit(y_train)
            y_pred = est.predict(fh=list(range(1, len(y_test) + 1)))
            print(y_pred)
            """
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
        return textwrap.dedent(
            f"""\
            from {module_name} import {class_name}
            from {mod} import {fn}
            from sklearn.metrics import accuracy_score, f1_score

            X_train, y_train = {fn}(split="train", return_X_y=True)
            X_test, y_test = {fn}(split="test", return_X_y=True)
            est = {class_name}({args})
            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            print("accuracy", accuracy_score(y_test, y_pred))
            print("macro_f1", f1_score(y_test, y_pred, average="macro"))
            """
        )

    dataset = get_dataset(dataset_id)
    path = dataset["path"] if dataset else "sktime/datasets/data/yahoo/yahoo.csv"
    return textwrap.dedent(
        f"""\
        import numpy as np
        import pandas as pd
        from {module_name} import {class_name}

        frame = pd.read_csv("{path}")
        raw = frame["data"].astype(float)
        est = {class_name}({args})
        out = est.fit_predict(raw.to_frame("data"))
        arr = np.asarray(out).ravel()
        pred = np.where(arr != 0)[0] if arr.size == len(raw) else np.array(arr, dtype=int)
        y_pred = np.zeros(len(raw), dtype=int)
        y_pred[pred[(pred >= 0) & (pred < len(raw))]] = 1
        print("detected", int(y_pred.sum()))
        """
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
        return f"Missing dependency `{package}`. Run `python3 -m pip install -e .` and optional `python3 -m pip install huggingface-hub pyarrow` for online datasets."
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
