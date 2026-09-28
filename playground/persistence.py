"""sktime save/load persistence backend for playground models.

Trained models live under ``playground/models/<model_id>/``. Two backends
share that layout:

* ``devad`` — written by ``sktime.libs.devad.services.model_service``
  (``model.pt`` + ``training.log`` + a DevAD ``manifest.json`` with
  ``family``/``params``/``seed``/``best_epoch``). Owned by
  ``trainer.train_devad``; this module never writes it.
* ``sktime`` — written here for every non-DevAD algorithm. The fitted
  estimator is serialized with sktime's own ``BaseEstimator.save`` (a zip
  holding ``_metadata``/``_obj`` pickles) as ``model.zip``; when the train
  spec selected a real preprocessor, the fitted preprocessor is stored the
  same way as ``preprocessor.zip`` so predict can re-apply the exact
  transformation the model was trained on. An existing model with the same
  id is overwritten, mirroring the DevAD backend.

``manifest.json`` schema (backend ``sktime``)::

    {
      "model_id":       str,   # directory name under playground/models/
      "backend":        "sktime",
      "task":           str,   # forecasting | classification | regression |
                               # clustering | anomaly_detection
      "algorithm_id":   str,   # catalog algorithm id that was trained
      "algorithm_name": str,   # display name of the algorithm
      "module":         str,   # estimator class import path
      "params":         dict,  # estimator constructor params used at train time
      "eval_params":    dict,  # task-level eval params (e.g. horizon, after
                               # any short-series adjustment) used to
                               # reproduce the train/holdout split at
                               # predict time
      "created_at":     str,   # ISO-8601 UTC timestamp
      "spec":           dict,  # normalized train spec: task, dataset_id,
                               # algorithm_id, preprocessor_id, params,
                               # preprocessor_params
      "artifacts":      {"model": "model.zip",
                         "preprocessor": "preprocessor.zip" | null}
    }

Backend selection rule (implemented in ``trainer.train``): algorithms whose
catalog ``module`` starts with ``sktime.detection.adapters.devad.`` are
trained through the DevAD backend; everything else — registered sktime
estimators, curated non-anomaly entries, and user plugins following the
sktime contract — is fitted as a plain sktime estimator and persisted
here. ``detect_backend`` identifies the backend of a persisted manifest;
DevAD manifests predate the ``backend`` field and are recognized by their
``family`` key.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

from runners import PlaygroundError

SKTIME_BACKEND = "sktime"
DEVAD_BACKEND = "devad"

MODEL_ARTIFACT = "model.zip"
PREPROCESSOR_ARTIFACT = "preprocessor.zip"
MANIFEST_NAME = "manifest.json"


def validate_model_id(model_id: str | None) -> str:
    """Require a model id to be a single directory name (no path separators)."""
    if not model_id or Path(model_id).name != model_id or model_id in {".", ".."}:
        raise PlaygroundError(
            f"model_id must be a single directory name, got {model_id!r}."
        )
    return model_id


def detect_backend(manifest: dict) -> str:
    """Backend of a persisted manifest: explicit field, else DevAD by `family`."""
    backend = manifest.get("backend")
    if backend:
        return backend
    return DEVAD_BACKEND if manifest.get("family") else SKTIME_BACKEND


def read_manifest(model_dir: str | Path) -> dict:
    """Read and parse ``manifest.json`` of a persisted model."""
    path = Path(model_dir) / MANIFEST_NAME
    if not path.is_file():
        raise PlaygroundError(f"No manifest at {path} (see `labts.py ls models`).")
    return json.loads(path.read_text(encoding="utf-8"))


def save_sktime_model(
    estimator,
    *,
    models_root: str | Path,
    model_id: str,
    task: str,
    algorithm: dict,
    est_params: dict,
    eval_params: dict,
    spec: dict,
    preprocessor=None,
) -> dict:
    """Persist a fitted sktime estimator plus its manifest; return the manifest.

    Writes ``model.zip`` via the estimator's own ``save`` (and
    ``preprocessor.zip`` when a fitted preprocessor is given) under
    ``<models_root>/<model_id>/``, followed by ``manifest.json`` in the
    schema documented in the module docstring. An existing model with the
    same id is overwritten.
    """
    validate_model_id(model_id)
    if not callable(getattr(estimator, "save", None)):
        raise PlaygroundError(
            f"Estimator of type {type(estimator).__name__} does not implement the "
            "sktime save/load contract (no `save` method) and cannot be persisted. "
            "User plugins need to inherit from a sktime base class to work with "
            "`labts train`."
        )
    model_dir = Path(models_root) / model_id
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    try:
        estimator.save(model_dir / "model").close()
        artifacts = {"model": MODEL_ARTIFACT, "preprocessor": None}
        if preprocessor is not None:
            preprocessor.save(model_dir / "preprocessor").close()
            artifacts["preprocessor"] = PREPROCESSOR_ARTIFACT
    except Exception as exc:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise PlaygroundError(
            f"Could not persist model `{model_id}`: {type(exc).__name__}: {exc}"
        ) from exc
    manifest = {
        "model_id": model_id,
        "backend": SKTIME_BACKEND,
        "task": task,
        "algorithm_id": algorithm.get("id"),
        "algorithm_name": algorithm.get("name"),
        "module": algorithm.get("module"),
        "params": est_params,
        "eval_params": eval_params,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spec": spec,
        "artifacts": artifacts,
    }
    (model_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _load_artifact(model_dir: str | Path, name: str):
    from sktime.base import BaseEstimator

    return BaseEstimator.load_from_path(Path(model_dir) / name)


def load_sktime_model(model_dir: str | Path):
    """Reload the fitted estimator persisted by `save_sktime_model`."""
    path = Path(model_dir) / MODEL_ARTIFACT
    if not path.is_file():
        raise PlaygroundError(f"Model artifact not found: {path}")
    try:
        return _load_artifact(model_dir, MODEL_ARTIFACT)
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(
            f"Could not load model artifact {path}: {type(exc).__name__}: {exc}"
        ) from exc


def load_sktime_preprocessor(model_dir: str | Path):
    """Reload the fitted preprocessor, or None when the model has none."""
    path = Path(model_dir) / PREPROCESSOR_ARTIFACT
    if not path.is_file():
        return None
    try:
        return _load_artifact(model_dir, PREPROCESSOR_ARTIFACT)
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(
            f"Could not load preprocessor artifact {path}: {type(exc).__name__}: {exc}"
        ) from exc
