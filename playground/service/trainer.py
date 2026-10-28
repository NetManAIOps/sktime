"""Training services for native sktime estimators."""

import json
from datetime import datetime, timezone
from pathlib import Path

from playground.service.catalog import TASK_ESTIMATOR_TYPES
from playground.service.data import load_training_data


def train(experiment_spec: dict, output_dir: str | Path):
    """Train and save a model/pipeline, then return the fitted object.

    Currently targets anomaly detection. Model hyperparameters (including
    early stopping) stay in init_params; fit_params are passed to fit.
    A top-level seed overrides seed/random_state on components that accept them.
    The complete configuration is experiment_spec. Validation data comes from
    data.val, and test data is never loaded here.
    Write model.zip and manifest.json into a new output_dir. The ZIP contains
    the whole estimator, including fitted preprocessing steps in a pipeline.
    Existing output directories are not overwritten. Inference and evaluation
    belong to the caller.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    X_train, X_val = load_training_data(experiment_spec["data"])
    estimator = build_estimator(
        experiment_spec["estimator"],
        task=experiment_spec["task"],
        seed=experiment_spec.get("seed"),
    )

    fit_params = dict(experiment_spec.get("fit_params", {}))
    if X_val is not None:
        fit_params["X_val"] = X_val

    estimator.fit(X_train, **fit_params)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_spec": experiment_spec,
        "artifacts": {"model": "model.zip"},
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    output_dir.mkdir(parents=True)
    estimator.save(output_dir / "model").close()
    (output_dir / "manifest.json").write_text(
        manifest_json + "\n", encoding="utf-8"
    )
    return estimator


def build_estimator(spec: list[dict], *, task: str, seed: int | None = None):
    """Construct ``spec["estimator"]`` without fitting any component.

    Each step has ``class`` and optional ``init_params``. Classes are resolved
    by exact short name among transformers and the task's estimator type.
    A single step must match the task. Multi-step construction is currently
    implemented only for anomaly detection, using DetectorPipeline.
    init_params are passed directly to each class constructor. When provided,
    seed overrides each component's seed/random_state without changing spec.
    Omitting seed keeps the component's supplied or default settings.
    """
    from sktime.detection.compose import DetectorPipeline
    from sktime.registry import all_estimators, is_scitype
    from sktime.transformations.base import BaseTransformer

    if not spec:
        raise ValueError("estimator must contain at least one step")
    model_type = TASK_ESTIMATOR_TYPES.get(task)
    if model_type is None:
        raise ValueError(f"Unsupported task: {task}")
    if len(spec) > 1 and task != "anomaly_detection":
        raise NotImplementedError(f"Pipeline construction is not implemented for {task}")

    registry = all_estimators(estimator_types=["transformer", model_type])
    steps = []
    for step in spec:
        class_name = step["class"]
        params = dict(step.get("init_params", {}))
        klass = _resolve_class(class_name, registry)
        if seed is not None:
            param_names = klass.get_param_names()
            for key in ("seed", "random_state"):
                if key in param_names:
                    params[key] = seed
        steps.append(klass(**params))

    if not is_scitype(steps[-1], model_type):
        raise TypeError(f"The last step must be a {model_type} for task {task}")
    if len(steps) == 1:
        return steps[0]

    if not all(isinstance(step, BaseTransformer) for step in steps[:-1]):
        raise TypeError("All steps before the detector must be sktime transformers")
    return DetectorPipeline(steps=steps)


def _resolve_class(class_name: str, registry) -> type:
    """Find one uniquely named class in the supplied registry."""
    matches = {klass for name, klass in registry if name == class_name}
    if len(matches) != 1:
        raise ValueError(
            f"Expected one class named {class_name}, found {len(matches)}"
        )
    return matches.pop()
