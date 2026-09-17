"""Persistent model training and detection for DevAD models.

`labts run` is stateless: fit and predict happen in one call and the fitted
model is discarded. For trainable anomaly detectors (the DevAD model zoo in
``sktime.libs.devad``) the playground also exposes the train-once/detect-many
workflow through `labts train` / `labts detect`:

    labts train  --algorithm registered-anomaly_detection-DevADFITSDetector \
        --dataset yahoo --model-id fits-v1 --param epochs=3
    labts detect --model-id fits-v1 --dataset yahoo --param threshold_quantile=0.99

`train` persists the model under ``playground/models/<model_id>/``
(``model.pt`` + ``manifest.json`` + ``training.log``, via DevAD's
``services.model_service``); `detect` reloads it and produces the same result
envelope as `labts run`, so `labts report --from` works on it unchanged.
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from catalog import (
    REPO_ROOT,
    get_dataset,
    get_enabled_algorithm,
    get_enabled_preprocessor,
    import_estimator_class,
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
        rows.append(
            {
                "model_id": manifest_path.parent.name,
                "family": manifest.get("family"),
                "params": manifest.get("params"),
                "seed": manifest.get("seed"),
                "best_epoch": manifest.get("best_epoch"),
                "model_dir": str(manifest_path.parent),
            }
        )
    return rows


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
