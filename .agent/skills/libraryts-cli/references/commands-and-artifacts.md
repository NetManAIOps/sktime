# 命令与产物

## 查询可用组件

```bash
libraryts ls algorithms --task anomaly_detection
libraryts ls algorithms --task anomaly_detection --name Donut
libraryts ls preprocessors --task anomaly_detection
libraryts ls preprocessors --task anomaly_detection --name StandardScaler
```

列表是 JSON 数组，每项含 `class`、`description`。指定 `--name` 后返回一个 JSON 对象，另含 `task` 和 `params`。参数值为 `"required"` 时，编写 spec 时需要提供该参数。当前组件查询覆盖 algorithms 和 preprocessors。

## 分阶段执行

在 [实验配置](experiment-spec.md) 示例的工作目录执行。以下输出目录应尚未存在。根据用户任务执行需要的阶段即可。

```bash
libraryts train --spec experiment.json --output-dir artifacts/donut
libraryts predict --spec experiment.json --model-dir artifacts/donut --output-dir artifacts/donut-scores
libraryts evaluate --spec experiment.json --prediction-dir artifacts/donut-scores --output-dir artifacts/donut-metrics
```

| 命令 | 成功 JSON 字段 | 目录中的文件 |
|---|---|---|
| `train` | `model_dir`、`model_path`、`manifest_path` | `model.zip`、`manifest.json` |
| `predict` | `prediction_dir`、`scores_path`、`n_scores`、`manifest_path` | `scores.csv`、`manifest.json` |
| `evaluate` | `evaluation_dir`、`metrics_path`、`metrics`、`manifest_path` | `metrics.json`、`manifest.json` |

自动化调用时，把 `train` 返回的 `model_dir` 传给 `predict --model-dir`，把 `predict` 返回的 `prediction_dir` 传给 `evaluate --prediction-dir`。CLI 返回的路径是绝对路径。用可信来源的 `model.zip`，其中包含拟合后的模型与管线。逐点异常分数在 `scores.csv` 中，列为 `timestamp,score`，分数越高表示越异常。

## 一次执行完整实验

数据已有独立的训练 CSV 和带标签测试 CSV 时，可以一次执行。

```bash
libraryts run --spec experiment.json --output-dir artifacts/donut-run
```

`run` 在输出目录内创建 `model/`、`prediction/`、`evaluation/` 和顶层 `manifest.json`。成功 JSON 含 `run_dir`、`model_dir`、`prediction_dir`、`evaluation_dir`、`metrics`、`manifest_path`。若中途失败，已完成的阶段可能仍在输出目录里。检查已有产物后，可以在新目录重跑，或从已完成的阶段接着调用独立命令。

## 读取结果和排查失败

| 退出码 | stdout | stderr |
|---|---|---|
| `0` | 一个结果 JSON | 可能有训练进度或其他日志 |
| `1` | `{"error": "具体原因"}` | 可能有之前的日志。加 `--debug` 时额外输出完整 Python 堆栈 |
| `2` | 通常为空 | Typer 的用法提示，例如漏填必填参数 |

例如，排查模型加载失败时可运行：

```bash
libraryts --debug predict --spec experiment.json --model-dir artifacts/donut --output-dir artifacts/donut-debug
```

`--debug` 放在子命令前。它只控制执行错误的堆栈，深度训练的进度由 spec 中的 `fit_params.show_progress` 控制。调用时分别捕获 stdout 和 stderr，这样 stdout 的 JSON 可以直接解析。
