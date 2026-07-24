"""Catalog and compatibility matrix for the local Sandbox Playground."""

from __future__ import annotations

import datetime
import importlib
import importlib.util
import json
import math
import numbers
import urllib.request
import warnings
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "sktime").is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    return here.parent


REPO_ROOT = _repo_root()

# Task-level evaluation parameters are not estimator hyperparameters; they
# configure the experiment (holdout size, anomaly threshold/window). Anything
# not listed here is forwarded to the selected estimator's constructor.
EVAL_PARAMS = {
    "forecasting": {"horizon", "context_window"},
    "classification": set(),
    "anomaly_detection": {"threshold", "window"},
}

_ESTIMATOR_TYPES = {
    "forecaster": "forecasting",
    "classifier": "classification",
    "detector": "anomaly_detection",
}

_DISCOVERED_CACHE: list[dict] | None = None
_PREPROCESSOR_CACHE: list[dict] | None = None
_CLASS_CACHE: dict[str, type] = {}


def split_params(task: str, params: dict) -> tuple[dict, dict]:
    """Split a flat param dict into (eval_params, estimator_params)."""
    eval_keys = EVAL_PARAMS.get(task, set())
    eval_params = {k: v for k, v in params.items() if k in eval_keys}
    est_params = {k: v for k, v in params.items() if k not in eval_keys}
    return eval_params, est_params


def import_estimator_class(module_path: str) -> type:
    """Import and cache an estimator class from a fully-qualified module path."""
    cached = _CLASS_CACHE.get(module_path)
    if cached is not None:
        return cached
    module_name, _, class_name = module_path.rpartition(".")
    klass = getattr(importlib.import_module(module_name), class_name)
    _CLASS_CACHE[module_path] = klass
    return klass


def _try_construct(klass: type) -> tuple[dict[str, float] | None, str | None]:
    """Return (numeric_params, None) or (None, error_message).

    An estimator is usable only if it can be constructed with its defaults
    (soft dependencies satisfied, no mandatory constructor argument).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            instance = klass()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    try:
        params = instance.get_params(deep=False)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    numeric = {
        name: value
        for name, value in params.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return numeric, None


HF_DATASET_ID = "Skyoung13/THU-ANM-DATASET"
HF_CONFIGS_FALLBACK = ["ETTm1", "ETTh1", "Exchange", "ILI", "NASDAQ", "Weather"]
# Backward-compatible alias for older imports.
HF_CONFIGS = HF_CONFIGS_FALLBACK
_HF_METADATA_CACHE: dict | None = None


def hf_dataset_configs() -> list[str]:
    """Return THU-ANM configs discovered from Hugging Face, with fallback."""
    metadata = hf_metadata()
    configs = metadata.get("configs") or HF_CONFIGS_FALLBACK
    clean = sorted({str(config) for config in configs if str(config).strip()})
    return clean or list(HF_CONFIGS_FALLBACK)


def hf_datasets() -> list[dict]:
    """Build dataset entries for all discovered THU-ANM Hugging Face configs."""
    return [
        {
            "id": f"hf-{config.lower()}",
            "name": f"THU-ANM {config}",
            "task": "forecasting",
            "source": "huggingface",
            "hf_dataset": HF_DATASET_ID,
            "hf_config": config,
            "enabled": True,
            "online": True,
        }
        for config in hf_dataset_configs()
    ]


def all_datasets() -> list[dict]:
    """Return local datasets plus dynamically discovered online datasets."""
    return list(DATASETS) + hf_datasets() + ucr_classification_datasets()

TASKS = [
    {
        "id": "forecasting",
        "label": "Forecasting",
        "description": "Forecast held-out future values and evaluate errors.",
    },
    {
        "id": "classification",
        "label": "Classification",
        "description": "Train a time-series classifier and evaluate test labels.",
    },
    {
        "id": "anomaly_detection",
        "label": "Anomaly Detection",
        "description": "Detect point anomalies and compare with known labels.",
    },
]

ENABLED_ALGORITHMS = [
    {
        "id": "naive-seasonal-last",
        "name": "NaiveForecaster",
        "task": "forecasting",
        "module": "sktime.forecasting.naive.NaiveForecaster",
        "enabled": True,
        "template": "NaiveForecaster(strategy='last', sp=seasonal_period)",
        "params": {"horizon": 12, "seasonal_period": 12, "context_window": 36},
    },
    {
        "id": "summary-random-forest",
        "name": "SummaryClassifier",
        "task": "classification",
        "module": "sktime.classification.feature_based.SummaryClassifier",
        "enabled": True,
        "template": "SummaryClassifier(RandomForestClassifier(n_estimators=25))",
        "params": {"n_estimators": 25, "random_state": 7},
    },
    {
        "id": "threshold-detector",
        "name": "ThresholdDetector",
        "task": "anomaly_detection",
        "module": "sktime.detection.naive.ThresholdDetector",
        "enabled": True,
        "subtype": "point_anomaly",
        "template": "ThresholdDetector(upper=threshold, lower=-threshold, mode='points')",
        "params": {"threshold": 2.0, "window": 24},
    },
]

for _entry in ENABLED_ALGORITHMS:
    _entry.setdefault("curated", True)
    _entry.setdefault("class_name", _entry["module"].rsplit(".", 1)[1])

PREPROCESSORS = [
    {
        "id": "none",
        "name": "Identity / none",
        "task": "all",
        "module": None,
        "enabled": True,
        "curated": True,
        "params": {},
    }
]

DATASETS = [
    {
        "id": "airline",
        "name": "Airline",
        "task": "forecasting",
        "source": "local",
        "loader": "sktime.datasets.load_airline",
        "enabled": True,
        "default": True,
    },
    {
        "id": "shampoo-sales",
        "name": "ShampooSales",
        "task": "forecasting",
        "source": "local",
        "loader": "sktime.datasets.load_shampoo_sales",
        "enabled": True,
    },
    {
        "id": "lynx",
        "name": "Lynx",
        "task": "forecasting",
        "source": "local",
        "loader": "sktime.datasets.load_lynx",
        "enabled": True,
    },
    {
        "id": "unit-test",
        "name": "UnitTest",
        "task": "classification",
        "source": "local",
        "loader": "sktime.datasets._single_problem_loaders.load_unit_test",
        "enabled": True,
        "default": True,
    },
    {
        "id": "arrow-head",
        "name": "ArrowHead",
        "task": "classification",
        "source": "local",
        "loader": "sktime.datasets.load_arrow_head",
        "enabled": True,
    },
    {
        "id": "italy-power-demand",
        "name": "ItalyPowerDemand",
        "task": "classification",
        "source": "local",
        "loader": "sktime.datasets.load_italy_power_demand",
        "enabled": True,
    },
    {
        "id": "gunpoint",
        "name": "GunPoint",
        "task": "classification",
        "source": "local",
        "loader": "sktime.datasets.load_gunpoint",
        "enabled": True,
    },
    {
        "id": "yahoo",
        "name": "Yahoo Anomaly",
        "task": "anomaly_detection",
        "source": "local",
        "path": "sktime/datasets/data/yahoo/yahoo.csv",
        "enabled": True,
        "default": True,
    },
    {
        "id": "mitdb",
        "name": "MIT-BIH ECG Anomaly",
        "task": "anomaly_detection",
        "source": "local",
        "path": "sktime/datasets/data/mitdb/mitdb.csv",
        "enabled": True,
    },
]


def ucr_classification_datasets() -> list[dict]:
    """Discover UCR/UEA classification datasets from sktime's archive list."""
    try:
        from sktime.datasets import UCRUEADataset

        names = sorted(UCRUEADataset.list_all())
    except Exception:
        return []
    return [
        {
            "id": f"ucr-{name.lower()}",
            "name": f"UCR/UEA {name}",
            "task": "classification",
            "source": "ucr_uea",
            "loader": "sktime.datasets.load_UCR_UEA_dataset",
            "ucr_name": name,
            "enabled": True,
            "online": True,
        }
        for name in names
    ]


def dependency_status() -> dict:
    """Return availability of runtime dependencies."""
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-base": "skbase",
        "scikit-learn": "sklearn",
        "huggingface-hub": "huggingface_hub",
        "pyarrow": "pyarrow",
    }
    status = {}
    for package, module_name in modules.items():
        found = importlib.util.find_spec(module_name) is not None
        status[package] = {"available": found}
        if not found:
            status[package]["install_hint"] = _install_hint(package)
    return status


def _install_hint(package: str) -> str:
    if package in {"huggingface-hub", "pyarrow"}:
        return "python3 -m pip install huggingface-hub pyarrow"
    return "python3 -m pip install -e ."


_DEP_IMPORT_ALIASES = {
    "scikit-learn": "sklearn",
    "scikit-base": "skbase",
    "scikit-optimize": "skopt",
    "scikit-posthocs": "scikit_posthocs",
    "hydra-core": "hydra",
    "pytorch-forecasting": "pytorch_forecasting",
    "dtw-python": "dtw",
    "pyyaml": "yaml",
    "python-dateutil": "dateutil",
}


def _dep_distribution_name(spec: str) -> str:
    import re

    return re.split(r"[>=<!~;\[\s]", spec.strip(), 1)[0].strip()


def _dep_marker_applies(spec: str) -> bool:
    if ";" not in spec:
        return True
    try:
        from packaging.markers import Marker

        return Marker(spec.split(";", 1)[1].strip()).evaluate()
    except Exception:
        return True


def _missing_python_dependency(deps) -> str | None:
    """Return the first declared dependency that is not importable, or None."""
    if not deps:
        return None
    if isinstance(deps, str):
        deps = [deps]
    import importlib.util

    for spec in deps:
        if isinstance(spec, (list, tuple)):
            candidates = [s for s in spec if isinstance(s, str) and _dep_marker_applies(s)]
            if candidates and not any(
                importlib.util.find_spec(
                    _DEP_IMPORT_ALIASES.get(_dep_distribution_name(s), _dep_distribution_name(s).replace("-", "_"))
                )
                for s in candidates
            ):
                return _dep_distribution_name(candidates[0])
            continue
        if not isinstance(spec, str) or not _dep_marker_applies(spec):
            continue
        name = _dep_distribution_name(spec)
        module = _DEP_IMPORT_ALIASES.get(name, name.replace("-", "_"))
        if importlib.util.find_spec(module) is None:
            return name
    return None


def _detector_subtype(name: str, module: str) -> str:
    """Classify a detector into a user-facing subtype for the anomaly task."""
    text = f"{name} {module}".lower()
    if any(k in text for k in ("capa", "mvcapa")):
        return "collective_anomaly"
    if any(
        k in text
        for k in (
            "threshold", "outlier", "anomal", "stray", "lof", "iforest",
            "isolation", "pyod", "knn", "mcd", "cblof", "cof", "abod", "sod",
            "subsequence", "matrixprofile", "stumpy", "sigma",
        )
    ):
        return "point_anomaly"
    if any(
        k in text
        for k in (
            "changepoint", "change_point", "binseg", "pelt", "clasp",
            "mle_binseg", "optimal_partition", "moving_window", "seeded",
        )
    ):
        return "change_point"
    if any(k in text for k in ("segment", "ggs", "hmm", "fluss", "clust", "wclust")):
        return "segmentation"
    return "other"


def discover_registered_algorithms() -> list[dict]:
    """Discover sktime estimators and enable the default-constructible ones.

    An estimator is enabled when it can be constructed with its defaults
    (soft dependencies satisfied, no mandatory argument). Enabled estimators
    expose their numeric scalar hyperparameters for the UI to render; the rest
    are reported as disabled with a reason. Results are cached per process.
    """
    global _DISCOVERED_CACHE
    if _DISCOVERED_CACHE is not None:
        return _DISCOVERED_CACHE

    try:
        from sktime.registry import all_estimators
    except Exception as exc:
        _DISCOVERED_CACHE = [
            {
                "id": "registry-unavailable",
                "name": "sktime registry unavailable",
                "task": "all",
                "enabled": False,
                "disabled_reason": f"{type(exc).__name__}: {exc}",
            }
        ]
        return _DISCOVERED_CACHE

    curated_names = {item["name"] for item in ENABLED_ALGORITHMS}
    discovered: list[dict] = []

    for estimator_type, task in _ESTIMATOR_TYPES.items():
        try:
            rows = all_estimators(
                estimator_types=estimator_type,
                return_tags=["python_dependencies"],
            )
        except Exception as exc:
            discovered.append(
                {
                    "id": f"{estimator_type}-registry-error",
                    "name": f"{estimator_type} registry error",
                    "task": task,
                    "enabled": False,
                    "disabled_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for row in rows:
            name, klass = row[0], row[1]
            if name in curated_names:
                continue
            deps = row[2] if len(row) > 2 else None
            module_path = f"{klass.__module__}.{klass.__name__}"
            params, err = _try_construct(klass)
            base = {
                "id": f"registered-{task}-{name}",
                "name": name,
                "task": task,
                "module": module_path,
                "class_name": klass.__name__,
                "curated": False,
                "python_dependencies": deps,
            }
            if task == "anomaly_detection":
                base["subtype"] = _detector_subtype(name, module_path)
            if err is not None:
                base.update(enabled=False, disabled_reason=err, params={})
                discovered.append(base)
                continue
            missing = _missing_python_dependency(deps)
            if missing is not None:
                base.update(
                    enabled=False,
                    disabled_reason=f"Missing dependency `{missing}`",
                    params={},
                )
                discovered.append(base)
                continue
            params = params or {}
            if task == "forecasting":
                params = {"horizon": 12, "context_window": 36, **params}
            base.update(enabled=True, params=params)
            discovered.append(base)

    _DISCOVERED_CACHE = discovered
    return discovered


def _tiny_series_fixtures():
    import numpy as np
    import pandas as pd

    # Strictly positive so positivity-requiring transformers (BoxCox/Log/Sqrt)
    # are not falsely rejected during the compatibility probe.
    values = np.arange(1, 61, dtype=float)
    return [
        pd.Series(values),
        pd.Series(values, index=pd.date_range("2000-01-01", periods=60, freq="D")),
    ]


def _tiny_panel_fixture():
    import numpy as np
    import pandas as pd

    return pd.DataFrame({"dim_0": [pd.Series(np.arange(30.0) + i) for i in range(8)]})


def _is_same_length_series(value, n):
    import pandas as pd

    if isinstance(value, pd.Series):
        return len(value) == n
    if isinstance(value, pd.DataFrame):
        return value.shape[1] == 1 and len(value) == n
    return False


def _probe_series_preprocessor(klass) -> bool:
    for fixture in _tiny_series_fixtures():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transformed = klass().fit_transform(fixture)
        except Exception:
            continue
        if _is_same_length_series(transformed, len(fixture)):
            return True
    return False


def _probe_panel_preprocessor(klass) -> bool:
    fixture = _tiny_panel_fixture()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = klass().fit_transform(fixture)
    except Exception:
        return False
    try:
        return len(transformed) == len(fixture)
    except Exception:
        return False


def discover_registered_preprocessors() -> list[dict]:
    """Discover default-constructible sktime transformers as preprocessors.

    A transformer is only offered for the tasks whose data it can round-trip
    without changing shape: same-length single-series output for
    forecasting/anomaly_detection, and the same number of instances for
    classification. Transformers that need ``y``, change length/columns, or
    rely on pairwise/feature extraction are filtered out per task.
    """
    global _PREPROCESSOR_CACHE
    if _PREPROCESSOR_CACHE is not None:
        return _PREPROCESSOR_CACHE

    try:
        from sktime.registry import all_estimators
    except Exception as exc:
        _PREPROCESSOR_CACHE = [
            {
                "id": "preprocessor-registry-unavailable",
                "name": "sktime transformer registry unavailable",
                "task": "all",
                "enabled": False,
                "compatible_tasks": [],
                "disabled_reason": f"{type(exc).__name__}: {exc}",
            }
        ]
        return _PREPROCESSOR_CACHE

    discovered: list[dict] = []
    try:
        rows = all_estimators(estimator_types="transformer", return_tags=["python_dependencies"])
    except Exception as exc:
        _PREPROCESSOR_CACHE = [
            {
                "id": "preprocessor-registry-error",
                "name": "transformer registry error",
                "task": "all",
                "enabled": False,
                "compatible_tasks": [],
                "disabled_reason": f"{type(exc).__name__}: {exc}",
            }
        ]
        return _PREPROCESSOR_CACHE

    for row in rows:
        name, klass = row[0], row[1]
        deps = row[2] if len(row) > 2 else None
        module_path = f"{klass.__module__}.{klass.__name__}"
        params, _err = _try_construct(klass)
        if params is None and _err is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    klass()
            except Exception:
                continue
        compatible_tasks: list[str] = []
        if _probe_series_preprocessor(klass):
            compatible_tasks.extend(["forecasting", "anomaly_detection"])
        if _probe_panel_preprocessor(klass):
            compatible_tasks.append("classification")
        if not compatible_tasks:
            continue
        discovered.append(
            {
                "id": f"registered-preprocessor-{name}",
                "name": name,
                "task": "all",
                "module": module_path,
                "class_name": klass.__name__,
                "curated": False,
                "python_dependencies": deps,
                "enabled": True,
                "compatible_tasks": compatible_tasks,
                "params": params or {},
            }
        )

    _PREPROCESSOR_CACHE = discovered
    return discovered


def hf_metadata() -> dict:
    """Fetch lightweight Hugging Face dataset metadata without optional deps."""
    global _HF_METADATA_CACHE
    if _HF_METADATA_CACHE is not None:
        return _HF_METADATA_CACHE

    url = f"https://datasets-server.huggingface.co/splits?dataset={HF_DATASET_ID}"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        configs = sorted({row["config"] for row in payload.get("splits", []) if row.get("config")})
        _HF_METADATA_CACHE = {"available": True, "dataset": HF_DATASET_ID, "configs": configs}
    except Exception as exc:
        _HF_METADATA_CACHE = {
            "available": False,
            "dataset": HF_DATASET_ID,
            "error": f"{type(exc).__name__}: {exc}",
            "configs": list(HF_CONFIGS_FALLBACK),
        }
    return _HF_METADATA_CACHE


def build_catalog(include_registered: bool = True) -> dict:
    """Return the Playground catalog consumed by the front end."""
    algorithms = list(ENABLED_ALGORITHMS)
    if include_registered:
        algorithms.extend(discover_registered_algorithms())
    preprocessors = list(PREPROCESSORS)
    if include_registered:
        preprocessors.extend(discover_registered_preprocessors())
    datasets = all_datasets()

    compatibility = []
    for algorithm in algorithms:
        if not algorithm.get("enabled"):
            continue
        for dataset in datasets:
            if algorithm["task"] == dataset["task"]:
                compatibility.append(
                    {
                        "task": algorithm["task"],
                        "algorithm_id": algorithm["id"],
                        "dataset_id": dataset["id"],
                    }
                )

    return {
        "tasks": TASKS,
        "algorithms": algorithms,
        "preprocessors": preprocessors,
        "datasets": datasets,
        "metrics": [
            {"id": "mae", "name": "MAE", "task": "forecasting"},
            {"id": "mse", "name": "MSE", "task": "forecasting"},
            {"id": "mape", "name": "MAPE", "task": "forecasting"},
            {"id": "accuracy", "name": "Accuracy", "task": "classification"},
            {"id": "macro_f1", "name": "Macro F1", "task": "classification"},
            {"id": "precision", "name": "Precision", "task": "anomaly_detection"},
            {"id": "recall", "name": "Recall", "task": "anomaly_detection"},
            {"id": "f1", "name": "F1", "task": "anomaly_detection"},
        ],
        "compatibility": compatibility,
        "dependencies": dependency_status(),
        "hf": hf_metadata(),
    }


def get_enabled_algorithm(algorithm_id: str) -> dict | None:
    for item in ENABLED_ALGORITHMS:
        if item["id"] == algorithm_id:
            return item
    for item in discover_registered_algorithms():
        if item["id"] == algorithm_id and item.get("enabled"):
            return item
    return None


def get_enabled_preprocessor(preprocessor_id: str | None) -> dict | None:
    if not preprocessor_id or preprocessor_id == "none":
        return PREPROCESSORS[0]
    for item in PREPROCESSORS:
        if item["id"] == preprocessor_id:
            return item
    for item in discover_registered_preprocessors():
        if item["id"] == preprocessor_id and item.get("enabled"):
            return item
    return None


def get_dataset(dataset_id: str) -> dict | None:
    return next((item for item in all_datasets() if item["id"] == dataset_id), None)


def _sanitize_for_json(obj):
    """Replace non-finite floats with None so the snapshot stays valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# Static, read-only mirror of `/api/catalog` for tools that should not start
# the HTTP server (coding agents, CI, docs). Kept next to the sandbox skill so
# it is easy to discover.
SNAPSHOT_PATH = (
    REPO_ROOT / ".agent" / "skills" / "time-series-sandbox" / "catalog_snapshot.json"
)


def write_snapshot(path: Path | None = None) -> Path:
    """Build the catalog and write it as a JSON snapshot.

    Best-effort view: network-backed entries (Hugging Face configs) fall back to
    defaults when offline, and the sktime registry degrades gracefully when soft
    dependencies are missing. Returns the path written.
    """
    target = Path(path) if path is not None else SNAPSHOT_PATH
    catalog = build_catalog(include_registered=True)
    payload = {
        "_meta": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": "playground.catalog.build_catalog",
            "refresh": "python playground/catalog.py (also refreshed on playground/server.py startup)",
            "note": "Static snapshot; may lag behind a running server. Regenerate to refresh.",
        },
        **catalog,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    out = write_snapshot()
    print(f"Wrote catalog snapshot to {out}")

