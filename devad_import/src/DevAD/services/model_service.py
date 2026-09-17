from __future__ import annotations

import json
import numpy as np
import shutil
from pathlib import Path
from typing import Any

from ..models.registry import ModelRegistry
from ..models.Base import BaseModel
from ..utils.evaluate import base_metricor
from ..utils.load import load_np
from ..utils.training_reporter import TrainingReporter


def train_model(
    model_root: str | Path,
    model_id: str,
    family: str,
    config_path: str | Path,
    x_train: str | Path,
    seed: int = 2026,
    device: str = "mps",
    x_val: str | Path | None = None,
    show_progress: bool = True,
) -> dict:
    if not model_id or Path(model_id).name != model_id or model_id in {".", ".."}:
        raise ValueError("model_id must be a single directory name")

    # 路径管理，创建产物 dir，默认新建一个空的名为 model_id 的文件夹
    model_dir = Path(model_root).expanduser().resolve() / model_id  # 训练过程统一保存的文件夹
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=False)

    weight_path = model_dir / "model.pt"  # 权重
    manifest_path = model_dir / "manifest.json"  # 具体配置
    log_path = model_dir / "training.log"  # log

    # 超参数配置文件读取 params
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Model config must be a JSON object")

    if family != payload.get("family"):
        raise ValueError("Selected model family is inconsistent with that in the model config file")

    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError("Model config must contain a params object")

    # 训练过程
    with TrainingReporter(
        info={
            "Family": family,
            "Config": config_path,
            "Train data": x_train,
            "Val data": x_val if x_val is not None else "—",
            "Output": model_dir,
            "Device": device,
            "Seed": seed,
        },
        log_path=log_path,
        show_progress=show_progress,
    ) as reporter:
        model = ModelRegistry.create_model(
            family=family,
            params=params,
            seed=seed,
            device=device,
        )

        fit_kwargs: dict[str, Any] = {"checkpoint_path": weight_path}
        fit_kwargs["reporter"] = reporter

        train_values = load_np(x_train)
        if x_val is not None:
            fit_kwargs["x_val"] = load_np(x_val)

        model.fit(train_values, **fit_kwargs)

        manifest = {
            "model_id": model_id,
            "family": family,
            "params": model.params,
            "seed": model.seed,
            "device": str(model.device),
            "best_epoch": model.best_epoch,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        reporter.finish(best_epoch=model.best_epoch)

    return {
        "model_id": model_id,
        "family": family,
        "model_dir": str(model_dir),
        "weight_path": str(weight_path),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
    }


def load_model(
    model_dir: str | Path,
    device: str,
) -> BaseModel:
    model_dir = Path(model_dir).expanduser().resolve()
    manifest_path = model_dir / "manifest.json"
    weight_path = model_dir / "model.pt"

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    model = ModelRegistry.create_model(
        family=manifest["family"],
        params=manifest["params"],
        seed=manifest["seed"],
        device=device,
    )

    model.load(weight_path)
    return model


def model_detect(
    model_dir: str | Path,
    device: str,
    x_test: str | Path,
    result_dir: str | Path,
) -> dict:
    model_dir = Path(model_dir).expanduser().resolve()
    test_path = Path(x_test).expanduser().resolve()
    result_dir = Path(result_dir).expanduser().resolve()

    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)

    test_values = load_np(test_path)
    trained_model = load_model(model_dir=model_dir, device=device)
    detect_result = trained_model.detect(test_values)

    input_length = len(test_values)
    score_length = len(detect_result.scores)
    start_pos = int(detect_result.start_pos)
    values = test_values[start_pos:]

    if len(values) != score_length:
        raise ValueError(
            "Scores do not satisfy the right-aligned suffix contract: "
            f"input_length={input_length}, start_pos={start_pos}, "
            f"score_length={score_length}"
        )

    score_path = result_dir / "scores.npy"
    value_path = result_dir / "values.npy"
    output_path = (
        result_dir / "output.npy"
        if detect_result.output is not None
        else None
    )
    meta_path = result_dir / "meta.json"

    np.save(score_path, detect_result.scores)
    np.save(value_path, values)
    if output_path is not None:
        np.save(output_path, detect_result.output)

    meta = {
        "input_length": input_length,
        "value_length": len(values),
        "score_length": score_length,
        "start_pos": start_pos,
        "direction": detect_result.direction,
    }
    if detect_result.output is not None:
        meta["output_length"] = len(detect_result.output)

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "model_dir": str(model_dir),
        "input_path": str(test_path),
        "result_dir": str(result_dir),
    }


def model_evaluate(
    model_dir: str | Path,
    device: str,
    x_test: str | Path,
    label: str | Path,
    result_dir: str | Path,
    metrics: list[str],
    metric_params: dict[str, dict] | None = None,
) -> dict:
    metric_names = list(dict.fromkeys(name.strip() for name in metrics))
    metric_params = metric_params or {}
    metricor = base_metricor()

    if not metric_names:
        raise ValueError("At least one metric must be specified")

    unknown = set(metric_names) - set(metricor.METRIC_MAP)
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}")

    unused_params = set(metric_params) - set(metric_names)
    if unused_params:
        raise ValueError(
            f"Parameters provided for unrequested metrics: {sorted(unused_params)}"
        )

    detect_result = model_detect(
        model_dir=model_dir,
        device=device,
        x_test=x_test,
        result_dir=result_dir,
    )

    result_dir = Path(result_dir).expanduser().resolve()
    with (result_dir / "meta.json").open("r", encoding="utf-8") as file:
        meta = json.load(file)

    required = {
        name
        for metric_name in metric_names
        for name in metricor.METRIC_MAP[metric_name]["requires"]
    }
    available: dict[str, np.ndarray] = {}

    if "score" in required:
        available["score"] = load_np(
            result_dir / "scores.npy",
            dtype=np.float64,
            name="scores",
        )

    if "label" in required:
        labels = load_np(label, dtype=np.int64, name="labels")
        input_length = int(meta["input_length"])
        start_pos = int(meta["start_pos"])
        if len(labels) != input_length:
            raise ValueError(
                "labels and detected input must have the same length: "
                f"labels={len(labels)}, input={input_length}"
            )

        available["label"] = labels[start_pos:]
        if len(available["label"]) != len(available["score"]):
            raise ValueError(
                "labels and scores are not aligned: "
                f"aligned_labels={len(available['label'])}, "
                f"scores={len(available['score'])}"
            )

    if "values" in required:
        available["values"] = load_np(
            result_dir / "values.npy",
            dtype=np.float64,
            name="values",
        )

    if "output" in required:
        available["output"] = load_np(
            result_dir / "output.npy",
            dtype=np.float64,
            name="output",
        )

    if "values" in required and "output" in required:
        if len(available["values"]) != len(available["output"]):
            raise ValueError(
                "values and output must have the same length: "
                f"values={len(available['values'])}, "
                f"output={len(available['output'])}"
            )

    computed = {
        metric_name: metricor(
            metric_name,
            **{
                name: available[name]
                for name in metricor.METRIC_MAP[metric_name]["requires"]
            },
            **metric_params.get(metric_name, {}),
        )
        for metric_name in metric_names
    }

    metrics_path = result_dir / "metrics.json"
    existing: dict[str, float] = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if not isinstance(existing, dict):
            raise ValueError("metrics.json must contain a JSON object")

    saved_metrics = {**existing, **computed}
    metrics_path.write_text(
        json.dumps(saved_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        **detect_result,
        "metrics_path": str(metrics_path),
        "metrics": saved_metrics,
    }


def get_model_info(family: str):
    return ModelRegistry.get_model_info(family=family)


def list_families():
    return ModelRegistry.list_families()
