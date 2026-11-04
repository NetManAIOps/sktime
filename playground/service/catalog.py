"""Read-only component discovery and constructor metadata for experiment specs."""

import inspect
import json


TASK_ESTIMATOR_TYPES = {
    "anomaly_detection": "detector",
    "forecasting": "forecaster",
    "classification": "classifier",
    "regression": "regressor",
    "clustering": "clusterer",
}

TASK_PREPROCESSORS = {
    "anomaly_detection": {
        "StandardScaler",
        "Imputer",
        "Differencer",
        "Detrender",
        "Deseasonalizer",
        "ConditionalDeseasonalizer",
        "BoxCoxTransformer",
        "LogTransformer",
        "ScaledAsinhTransformer",
        "ExponentTransformer",
        "MovingAverageTransformer",
    },
}


def list_algorithms(task: str, name: str | None = None) -> list[dict] | dict:
    """List a task's classes, or describe one exact short class name.

    The task selects an object_type via TASK_ESTIMATOR_TYPES. Classes are
    additionally filtered by their task tag. Legacy DevAD adapters and the
    generic PyODDetector wrapper are hidden; concrete PyOD algorithms remain.
    No class is instantiated, fitted or checked with synthetic data. Registry
    discovery may import modules; discovery does not guarantee that constructor
    dependencies are installed or that every class fits the current JSON spec.

    Without name return [{"class": ..., "description": ...}, ...]. With name
    return {"class": ..., "task": ..., "description": ..., "params": ...}.
    Descriptions use the first docstring paragraph. Params map names to defaults, or
    "required" when no default is defined.
    Non-JSON defaults are explicitly represented as {"python_repr": ...}; they
    are descriptions, not directly usable init_params values.
    """
    from sktime.registry import all_estimators

    estimator_type = TASK_ESTIMATOR_TYPES.get(task)
    if estimator_type is None:
        raise ValueError(f"Unsupported task: {task}")
    filter_tags = {"task": task}

    # 类型过滤 + task 过滤
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
    return _list_estimators(registry, name, task=task)


def list_preprocessors(task: str, name: str | None = None) -> list[dict] | dict:
    """List a task's selected transformers, or describe one exact class name.

    Discovery does not instantiate or fit classes, check dependencies, or
    guarantee compatibility for every parameter choice. TASK_PREPROCESSORS
    selects names from the transformer registry; shared transformers do not need
    a task tag. Only anomaly_detection currently has a configured list.
    Details contain class, task, description and params, as for algorithms.
    No parameters are overridden or preprocessing steps inserted automatically.
    """
    from sktime.registry import all_estimators

    names = TASK_PREPROCESSORS.get(task)
    if names is None:
        raise ValueError(f"Unsupported preprocessing task: {task}")
    registry = all_estimators(estimator_types="transformer")
    registry = [
        (class_name, klass)
        for class_name, klass in registry
        if class_name in names
    ]
    return _list_estimators(registry, name, task=task)


def _list_estimators(registry, name: str | None, *, task: str):
    """Describe discovered classes without constructing them."""
    # 列出 registry 的全部
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

    # 列出指定的那个 class 及详细信息
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
    return {
        "class": name,
        "task": task,
        "description": " ".join(
            (inspect.getdoc(klass) or "").split("\n\n", 1)[0].splitlines()
        ),
        "params": {
            param.name: (
                "required"
                if param.default is inspect.Parameter.empty
                else _describe_default(param.default)
            )
            for param in parameters
        },
    }


def _describe_default(value):
    """Keep JSON defaults intact and mark other Python values as descriptions."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return {"python_repr": repr(value)}
