from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..probes import PseudoAnomalyGenerator, mean_interval_percentile_lift
from ..utils.evaluate import base_metricor
from ..utils.load import load_np
from .model_service import load_model


def prepare_pseudo_anomaly(
    input_data: str | Path,
    output_dir: str | Path,
    anomaly_type: str,
    anomaly_rate: float,
    length_range: tuple[int, int],
    strength_range: tuple[float, float],
    seed: int = 2026,
) -> dict:
    """Save one pseudo-anomaly sequence, its labels and generation settings."""
    input_path = Path(input_data).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    generator = PseudoAnomalyGenerator(
        anomaly_type=anomaly_type,
        length_range=length_range,
        strength_range=strength_range,
        anomaly_rate=anomaly_rate,
        seed=seed,
    )
    result = generator.generate(load_np(input_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "injected.npy", result.injected)
    np.save(output_dir / "label.npy", result.label)

    config = {
        "probe_type": "pseudo-anomaly",
        "input_path": str(input_path),
        "anomaly_type": anomaly_type,
        "anomaly_rate": anomaly_rate,
        "length_range": length_range,
        "strength_range": strength_range,
        "seed": seed,
        "intervals": result.intervals,
    }
    (output_dir / "probe.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
    }


def run_pseudo_anomaly(
    model_root: str | Path,
    probe_dir: str | Path,
    device: str = "mps",
) -> dict:

    # 读取完整 probe 数据
    probe_dir = Path(probe_dir).expanduser().resolve()
    with (probe_dir / "probe.json").open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("probe_type") != "pseudo-anomaly":
        raise ValueError(
            f"Expected probe_type 'pseudo-anomaly', got {config.get('probe_type')!r}"
        )

    clean = load_np(config["input_path"])
    injected = load_np(probe_dir / "injected.npy")
    labels = load_np(probe_dir / "label.npy", dtype=np.int64, name="labels")
    if not len(clean) == len(injected) == len(labels):
        raise ValueError("Original, injected and label sequences must have the same length")

    intervals = config["intervals"]
    for start, end in intervals:
        if not 0 <= start < end <= len(clean):
            raise ValueError(f"Invalid input interval: ({start}, {end})")

    # 批量探测候选模型
    model_root = Path(model_root).expanduser().resolve()
    model_dirs = sorted(
        path for path in model_root.iterdir()
        if path.is_dir()
    )
    if not model_dirs:
        raise ValueError(f"No model directories found in {model_root}")

    detected = {}
    errors = {}
    common_start_pos = 0

    for model_dir in model_dirs:
        model = None
        try:
            model = load_model(model_dir=model_dir, device=device)
            clean_result = model.detect(clean)
            injected_result = model.detect(injected)
            for result in (clean_result, injected_result):
                if not len(result.scores) or result.start_pos + len(result.scores) != len(clean):
                    raise ValueError("Scores must be a non-empty right-aligned suffix")

            # 只缓存分数及其起点，不保留模型和重构结果。
            detected[model_dir.name] = (
                clean_result.scores, clean_result.start_pos,
                injected_result.scores, injected_result.start_pos,
            )
            common_start_pos = max(
                common_start_pos, clean_result.start_pos, injected_result.start_pos
            )
            del clean_result, injected_result
        except Exception as exc:
            errors[model_dir.name] = str(exc)
        finally:
            model = None

    scores = {}
    if detected:
        # 区间与标签统一对齐，跨 warm-up 边界的区间只保留有效部分。
        aligned_intervals = [
            (max(start, common_start_pos) - common_start_pos, end - common_start_pos)
            for start, end in intervals
            if end > common_start_pos
        ]
        if not aligned_intervals:
            raise ValueError("No anomaly intervals remain after score alignment")

        aligned_labels = labels[common_start_pos:]
        metricor = base_metricor()
        for model_id, results in detected.items():
            clean_scores, clean_start, injected_scores, injected_start = results
            clean_scores = clean_scores[common_start_pos - clean_start:]
            injected_scores = injected_scores[common_start_pos - injected_start:]
            try:
                scores[model_id] = {
                    "mean_interval_percentile_lift": mean_interval_percentile_lift(
                        clean_scores, injected_scores, aligned_intervals
                    ),
                    "synthetic_vus_pr": metricor(
                        "VUS-PR",
                        label=aligned_labels,
                        score=injected_scores,
                        sliding_window=50,
                    ),
                }
            except Exception as exc:
                errors[model_id] = str(exc)

    scores_path = probe_dir / "probe_scores.json"
    scores_path.write_text(
        json.dumps(scores, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "model_root": str(model_root),
        "probe_dir": str(probe_dir),
        "scores_path": str(scores_path),
        "success": len(scores),
        "errors": errors,
    }