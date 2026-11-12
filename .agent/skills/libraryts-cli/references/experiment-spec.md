# 实验配置

`--spec` 接收 UTF-8 JSON 文件路径。命令参数和配置中的相对数据路径都按运行命令时的工作目录解析。移动 spec 文件时，仍以命令的工作目录为基准。

## 各阶段读取的字段

| 命令 | 使用的字段 |
|---|---|
| `train` | `task`、`data.train`、`estimator`，以及可选的 `data.val`、`seed`、`fit_params` |
| `predict` | `task`、`data.test`。模型与已拟合的预处理由 `--model-dir` 提供 |
| `evaluate` | `task`、`data.test`、`metrics`。分数由 `--prediction-dir` 提供 |
| `run` | 上述字段的并集。必须显式提供不同的 train/test 文件，并选择至少一个指标 |

- `task` 填 `"anomaly_detection"`。
- CSV 第一行为列名。单变量输入提供一列数值，例如 `value`。可选的 `timestamp` 列会作为索引，省略时使用行号。打分按输入顺序，评估要求测试标签与分数的索引和顺序一致。
- 带标签评估时，测试 CSV 包含 `label` 列。训练和打分都从模型输入中去除该列，评估才读取它。使用 VUS-PR 时测试标签应包含正常点 `0` 和异常点 `1`。
- 单步 `estimator` 也写成数组。管线按数组顺序执行，最后一步是检测器，前面的步骤是预处理器。每步用精确的短类名 `class`，构造参数放在可选的 `params`。可用 `ls ... --name` 查询签名和默认值。省略可选参数时使用构造默认值；只用默认值的检测器步骤可省略整个 `params`。
- `ls algorithms --name` 的 `core_params` 提示优先调参项，`recommended_preprocessors` 提供可选用的有序管线步骤。把适用步骤写进 `estimator`，再追加检测器和本次要覆盖的模型参数。复现实验时对照 [官方参考配置](official-configs.md)。
- `MinMaxScaler` 用训练段的最小值和最大值拟合。默认映射到 `[0, 1]`，可在 spec 用 `feature_range: [-1, 1]` 指定区间。默认保留验证或测试超出训练范围的值，需要裁剪时设置 `clip: true`。缺失位置由单独的填补步骤处理。
- `Differencer` 接受 `lag` 和 `order`，默认均为 1。每个数据段独立差分，保持索引和长度，开头最多 `lag × order` 个点补零。如果原始数据含缺失值，先处理缺失值。
- 顶层 `seed` 可选。提供后会覆盖各步骤支持的 `seed` 或 `random_state`。深度检测器可用 `fit_params.show_progress` 控制训练进度。早停等构造参数仍在检测器的 `params` 内。
- `metrics` 是从指标名到参数对象的映射，例如 `{"VUS-PR": {}}`。空对象表示用默认参数。
- 独立阶段可以使用可解析的 `data.dataset_id`。注册 ID 指向整条序列，当前没有独立的训练集和测试集划分。ID 与本地 split 路径二选一。需要 held-out 评估时准备明确分开的本地 CSV。

## StandardScaler → Donut 示例

将下列内容保存为当前工作目录下的 `experiment.json`。准备对应的 CSV 文件。验证集可省略，带标签测试集用于 `evaluate` 或 `run`。示例使用 CPU 和一轮训练，适合验证调用链。正式实验按任务设置模型参数和评估协议。

```json
{
  "task": "anomaly_detection",
  "seed": 2026,
  "data": {
    "train": "data/train.csv",
    "val": "data/val.csv",
    "test": "data/test.csv"
  },
  "estimator": [
    {"class": "StandardScaler"},
    {
      "class": "Donut",
      "params": {
        "win_len": 32,
        "h_dim": 32,
        "z_dim": 4,
        "batch_size": 64,
        "epochs": 1,
        "mc_samples": 8,
        "device": "cpu"
      }
    }
  ],
  "fit_params": {"show_progress": true},
  "metrics": {"VUS-PR": {}}
}
```
