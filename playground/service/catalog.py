"""Read-only algorithm discovery and constructor metadata for experiment specs."""

import inspect
import json


TASK_ESTIMATOR_TYPES = {
    "anomaly_detection": "detector",
    "forecasting": "forecaster",
    "classification": "classifier",
    "regression": "regressor",
    "clustering": "clusterer",
}


def list_algorithms(task: str, name: str | None = None) -> list[dict] | dict:
    """List a task's classes, or describe one exact short class name.

    The task selects an object_type via TASK_ESTIMATOR_TYPES. Detectors are
    additionally filtered by their task tag. Legacy DevAD adapters and the
    generic PyODDetector wrapper are hidden; concrete PyOD algorithms remain.
    No class is instantiated, fitted or checked with synthetic data. Registry
    discovery may import modules; discovery does not guarantee that constructor
    dependencies are installed or that every class fits the current JSON spec.

    Without name return [{"class": ..., "description": ...}, ...]. With name
    return one detail dictionary, including required/default constructor params,
    available annotations, capabilities and the original class documentation.
    Non-JSON defaults are explicitly represented as {"python_repr": ...}; they
    are descriptions, not directly usable init_params values. Missing parameter
    annotations are left to the original documentation, not guessed.
    """
    from sktime.registry import all_estimators

    estimator_type = TASK_ESTIMATOR_TYPES.get(task)
    if estimator_type is None:
        raise ValueError(f"Unsupported task: {task}")
    filter_tags = {"task": task}
    registry = all_estimators(
        estimator_types=estimator_type, filter_tags=filter_tags
    )
    if estimator_type == "detector":
        from sktime.detection.adapters import PyODDetector

        registry = [
            (class_name, klass)
            for class_name, klass in registry
            if klass.__module__ != "sktime.detection.adapters.devad"
            and klass is not PyODDetector
        ]
    if name is None:
        return [
            {
                "class": class_name,
                "description": " ".join(
                    (inspect.getdoc(klass) or "").split("\n\n", 1)[0].splitlines()
                ),
            }
            for class_name, klass in registry
        ]

    matches = {klass for class_name, klass in registry if class_name == name}
    if len(matches) != 1:
        raise ValueError(f"Expected one class named {name}, found {len(matches)}")
    klass = matches.pop()
    parameters = [
        param
        for key, param in inspect.signature(klass.__init__).parameters.items()
        if key != "self"
        and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    tags = klass.get_class_tags()
    return {
        "class": name,
        "task": task,
        "documentation": inspect.getdoc(klass) or "",
        "required_params": [
            param.name for param in parameters if param.default is inspect.Parameter.empty
        ],
        "default_params": {
            param.name: _describe_default(param.default)
            for param in parameters
            if param.default is not inspect.Parameter.empty
        },
        "parameter_types": {
            param.name: inspect.formatannotation(param.annotation)
            for param in parameters
            if param.annotation is not inspect.Parameter.empty
        },
        "capabilities": {
            key: _describe_default(value)
            for key, value in tags.items()
            if key.startswith("capability:")
        },
        "python_dependencies": _describe_default(tags.get("python_dependencies")),
    }


def _describe_default(value):
    """Keep JSON defaults intact and mark other Python values as descriptions."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return {"python_repr": repr(value)}
