---
name: libraryts-cli
description: Use the native LibraryTS CLI for time-series anomaly detection. Discover detectors and preprocessors, build experiment specs, train and save pipelines, score new CSV data with saved models, and evaluate labeled experiments. This skill applies to the installed libraryts entry point backed by playground.cli.
---

# LibraryTS CLI

使用新版 `libraryts` 命令完成异常检测任务。先确认 `libraryts --help` 显示 `ls`、`train`、`predict`、`evaluate` 和 `run`。本 skill 对应 `playground.cli` 入口。

## 按目标选命令

| 任务 | 命令 | 结果 |
|---|---|---|
| 查询检测器或预处理器 | `ls algorithms` / `ls preprocessors`，带 `--task anomaly_detection` | 可选类名和简介 |
| 已知要用哪个类 | 在对应的 `ls` 命令追加 `--name CLASS` | 构造参数、必填项和默认值 |
| 训练并交付模型或管线 | `train` | `model.zip` 和 `manifest.json` |
| 用已保存模型对新数据打分 | `predict` | `scores.csv` |
| 用已有分数和测试标签算指标 | `evaluate` | `metrics.json` |
| 一次执行训练、打分和带标签评估 | `run` | 三阶段的产物和本轮 manifest |

已知目标类时，直接查询该类的详情。编写或调整 `--spec` 时读 [实验配置](references/experiment-spec.md)。执行命令、传递产物或排查失败时读 [命令与产物](references/commands-and-artifacts.md)。

## 执行约定

- 每次执行使用新的输出目录。以退出码判断成功或失败，从 `stdout` 读取结果 JSON，保留返回的目录路径用于下一步。需要定位执行错误时，在子命令前加 `--debug` 并查看 `stderr`。
- 复用已训练模型时调用 `predict` 并传入可信的模型目录。已有分数时可直接调用 `evaluate`。
- 在无标签模型选择阶段，只使用研究协议允许的数据和反馈。真实测试标签上的 `evaluate` 和 `run` 留给独立评估阶段。
