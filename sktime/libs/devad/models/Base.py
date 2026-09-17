from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch

from copy import deepcopy
from ..utils.evaluate import base_metricor
from ..utils.train_utils import setup_seed

@dataclass
class DetectResult:
    scores: np.ndarray
    start_pos: int
    output: np.ndarray | None = None
    direction: str = "higher"

    def __post_init__(self):
        scores = np.asarray(self.scores, dtype=float)

        if scores.ndim != 1:
            raise ValueError(
                f"scores must have shape (n,), got {scores.shape}"
            )

        if self.start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        if not np.isfinite(scores).all():
            raise ValueError("scores contain NaN or Inf")

        object.__setattr__(self, "scores", scores)

        if self.output is not None:
            output = np.asarray(self.output, dtype=float)
            if output.ndim != 1:
                raise ValueError(
                    f"output must have shape (n,), got {output.shape}"
                )
            if len(output) != len(scores):
                raise ValueError("output and scores must have the same length")
            object.__setattr__(self, "output", output)


@dataclass
class BaseConfig:
    seed: int = 0
    device: str = "cpu"
    params: Dict[str, Any] = field(default_factory=dict)


REQUIRED = object()


class BaseModel(ABC):
    STATEFUL_ATTRS = ("scale_cfg", "best_epoch")

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.is_fitted = False
        self.seed = config.seed
        self.device = config.device
        self.params = self._resolve_params(config.params)
        self.scale_cfg: Dict[str, Any] = {}
        self.best_epoch: int | None = None
        setup_seed(self.seed)

    def _resolve_params(self, params: dict) -> dict: # 判断传入超参数的完整性

        hp = {
            **type(self).HP,
        }

        # 未知参数
        unknown = set(params) - set(hp)
        if unknown:
            raise ValueError(
                f"Unknown parameters for {type(self).__name__}: "
                f"{sorted(unknown)}"
            )

        missing = {
            name for name, default in hp.items() if default is REQUIRED and name not in params
        }

        # 必填参数缺失
        if missing:
            raise ValueError(
                f"Missing required parameters for {type(self).__name__}: "
                f"{sorted(missing)}"
            )

        # 非必填参数用默认值补全
        resolved = {
            name: deepcopy(default) for name, default in hp.items() if default is not REQUIRED
        }

        resolved.update(deepcopy(params))
        return resolved

    def fit(self, x_train: np.ndarray, y: np.ndarray = None, **kwargs) -> "BaseModel":
        setup_seed(self.seed)
        self.is_fitted = False
        self.best_epoch = None

        x_train = self.check_array(x_train, dtype=np.float32, name="x_train")
        if y is not None:
            y = self.check_array(y, dtype=np.int64, name="label")
            if len(x_train) != len(y):
                raise ValueError("x_train and label must have the same length")

        self._fit(x_train, y, **kwargs)

        if self.best_epoch is None:
            self.best_epoch = int(self.params["epochs"]) if "epochs" in self.params else None
        self.is_fitted = True

        checkpoint_path = kwargs.get("checkpoint_path")
        if checkpoint_path:
            self.save(checkpoint_path)
        return self

    def detect(self, x_test: np.ndarray) -> DetectResult:
        self._check_fitted()
        setup_seed(self.seed)
        x_test = self.check_array(x_test, dtype=np.float32, name="x")
        result = self._detect(x_test)
        if result.output is not None:
            result.output = self.inverse_transform(result.output, self.scale_cfg)
        return result

    def evaluate(self, x_test: np.ndarray, y: np.ndarray, delta_k: int = 3):
        x_test = self.check_array(x_test, dtype=np.float32, name="x")
        y = self.check_array(y, dtype=np.int64, name="label")

        if len(x_test) != len(y):
            raise ValueError("x_test and label must have the same length")

        result = self.detect(x_test)
        labels = y[result.start_pos:]
        return base_metricor().metric_all(label=labels, score=result.scores, delta_k=delta_k)

    def save(self, path: str) -> None:
        save_dir = os.path.dirname(os.fspath(path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        payload = {
            "model_class": type(self).__name__,
            "seed": self.seed,
            "device": str(self.device),
            "params": self.params,
        }
        extra_state = {}
        for name in self.STATEFUL_ATTRS:
            if hasattr(self, name):
                extra_state[name] = getattr(self, name)
        if extra_state:
            payload["extra_state"] = extra_state
        if hasattr(self.backend, "state_dict") and callable(self.backend.state_dict):
            payload["save_type"] = "state_dict"
            payload["state_dict"] = self.backend.state_dict()
        else:
            payload["save_type"] = "backend"
            payload["backend"] = self.backend
        torch.save(payload, path)

    def load(self, path: str, strict_config: bool = True) -> None:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid checkpoint payload type: {type(payload).__name__}")

        checkpoint_class = payload.get("model_class")
        current_class = type(self).__name__
        if checkpoint_class != current_class:
            raise ValueError(
                "Checkpoint model_class does not match current model: "
                f"checkpoint={checkpoint_class!r}, current={current_class!r}"
            )

        checkpoint_seed = payload.get("seed")
        if checkpoint_seed is not None:
            self.seed = int(checkpoint_seed)

        if strict_config and "params" not in payload:
            raise ValueError("Checkpoint missing params.")
        if strict_config and payload["params"] != self.params:
            raise ValueError(
                "Checkpoint params do not match current model params. "
                "Instantiate the model with the same candidate config before loading."
            )

        if "seed" in payload:
            self.seed = int(payload["seed"])

        if payload["save_type"] == "state_dict":
            self.backend.load_state_dict(payload["state_dict"])
        else:
            self.backend = payload["backend"]
        extra_state = payload.get("extra_state", {})
        for name, value in extra_state.items():
            setattr(self, name, value)
        self.is_fitted = True

    @abstractmethod
    def _fit(self, train: np.ndarray, y: np.ndarray, **kwargs) -> None: ...

    @abstractmethod
    def _detect(self, x: np.ndarray): ...

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

    @staticmethod
    def check_array(data, dtype, name: str) -> np.ndarray:
        array = np.asarray(data, dtype=dtype)

        if array.ndim != 1:
            raise ValueError(
                f"{name} must have shape (n,), got {array.shape}"
            )

        return array

    @staticmethod
    def inverse_transform(values: np.ndarray, scale_cfg: dict) -> np.ndarray:
        if not scale_cfg:
            raise ValueError("scale_cfg is required to inverse-transform model output")

        mode = scale_cfg["scale_mode"]

        if mode == "zscore":
            return values * scale_cfg["std"] + scale_cfg["mean"]

        if mode == "minmax":
            return (
                values * (scale_cfg["max"] - scale_cfg["min"])
                + scale_cfg["min"]
            )

        raise ValueError(f"Unsupported scale mode: {mode}")
