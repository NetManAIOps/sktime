# 命令与产物

## 查询可用组件

```bash
libraryts ls algorithms --task anomaly_detection
libraryts ls algorithms --task anomaly_detection --name Donut
libraryts ls preprocessors --task anomaly_detection
libraryts ls preprocessors --task anomaly_detection --name MinMaxScaler
```

列表是 JSON 数组，每项含 `class`、`description`。指定 `--name` 后返回一个 JSON 对象，另含 `task` 和 `params`。`params` 显示当前构造默认值，标为 `"required"` 的字段必须在 spec 中提供。检测器详情还返回 `core_params` 和 `recommended_preprocessors`。前者列出可优先尝试的参数名，后者按执行顺序提供可复制到 spec 的步骤。预处理器详情使用通用字段。模型侧使用经验见 [检测器参考](detectors/index.md)，上游实验差异见 [官方参考配置](official-configs.md)。

## 分阶段执行

在 [实验配置](experiment-spec.md) 示例的工作目录执行。`train` 可复用已有目录，遇到已有的 `model.zip` 会在训练前报错。`predict` 同样可复用已有目录，遇到已有的 `scores.csv` 会在加载模型前报错。`evaluate` 可复用已有目录，成功后更新 `metrics.json` 和评估用的 `manifest.json`。根据用户任务执行需要的阶段即可。

```bash
libraryts train --spec experiment.json --output-dir artifacts/donut
libraryts predict --spec experiment.json --model-path artifacts/donut/model.zip --output-dir artifacts/donut-scores
libraryts evaluate --spec experiment.json --scores-path artifacts/donut-scores/scores.csv --output-dir artifacts/donut-metrics
```

| 命令 | 成功 JSON 字段 | 目录中的文件 |
|---|---|---|
| `train` | `model_dir`、`model_path`、`manifest_path` | `model.zip`、`manifest.json` |
| `predict` | `prediction_dir`、`scores_path`、`n_scores`、`manifest_path` | `scores.csv`、`manifest.json` |
| `evaluate` | `evaluation_dir`、`metrics_path`、`metrics`、`manifest_path` | `metrics.json`、`manifest.json` |

自动化调用时，把 `train` 返回的 `model_path` 传给 `predict --model-path`，把 `predict` 返回的 `scores_path` 传给 `evaluate --scores-path`。CLI 返回的路径是绝对路径。用可信来源的 `model.zip`，其中包含拟合后的模型与管线。逐点异常分数在 `scores.csv` 中，列为 `timestamp,score`，分数越高表示越异常。

## 一次执行完整实验

数据已有独立的训练 CSV 和带标签测试 CSV 时，可以一次执行。

```bash
libraryts run --spec experiment.json --output-dir artifacts/donut-run
```

`run` 将本轮的 `spec.json`、`model.zip`、`scores.csv` 和 `metrics.json` 写在同一目录，成功后写入一份 `manifest.json`，记录 UTC 完成时间及各文件的相对路径。若目录中已有内容，只接受与本轮配置一致的 `spec.json`。CLI 返回 `run_dir`、`spec_path`、`model_path`、`scores_path`、`metrics_path`、`manifest_path` 和 `metrics`。失败时已完成的阶段文件可用于排查。

## 读取结果和排查失败

| 退出码 | stdout | stderr |
|---|---|---|
| `0` | 一个结果 JSON | 可能有训练进度或其他日志 |
| `1` | `{"error": "具体原因"}` | 可能有之前的日志。加 `--debug` 时额外输出完整 Python 堆栈 |
| `2` | 通常为空 | Typer 的用法提示，例如漏填必填参数 |

例如，排查模型加载失败时可运行：

```bash
libraryts --debug predict --spec experiment.json --model-path artifacts/donut/model.zip --output-dir artifacts/donut-debug
```

`--debug` 放在子命令前。它只控制执行错误的堆栈，深度训练的进度由 spec 中的 `fit_params.show_progress` 控制。调用时分别捕获 stdout 和 stderr，这样 stdout 的 JSON 可以直接解析。
