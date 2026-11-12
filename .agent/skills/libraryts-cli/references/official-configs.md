# 官方参考配置

用 `libraryts ls algorithms --task anomaly_detection --name CLASS` 查询当前构造参数和默认值。按任务把适用的预处理步骤依序写入 `estimator`，并把需要覆盖的模型参数写入检测器步骤的 `params`。实验中的缩放统计量只从训练段拟合。

## 预处理步骤与参考参数

下表记录项目参考管线。覆盖参数一栏列出复现此前项目配置时需要显式设置的字段。其中 min-max 缩放使用训练段拟合。上游实验和 LibraryTS 的适配差异见下一节。

| 检测器 | 预处理步骤 | 参考配置中的覆盖参数 |
|---|---|---|
| AnomalyTransformer | `StandardScaler` | — |
| COUTA | `MinMaxScaler` → `Imputer(method="linear")` | `early_stop_patience=0` |
| Donut | `StandardScaler` | — |
| FCVAE | `StandardScaler` → `Imputer(method="constant", value=0, add_missing_mask=True)` | `win_len=64`、`stride=8`、`batch_size=512`、`missing_data_rate=0.01`、`point_ano_rate=0.05`、`seg_ano_rate=0.1` |
| FITS | `StandardScaler` | `win_len=100`、`downsample_rate=4`、`cut_freq=12`，三项一起设置 |
| KANAD | `Differencer(lag=1, order=1)` → `StandardScaler` | — |
| ModernTCN | `StandardScaler` | `win_len=100`、`patch_size=8`、`patch_stride=4`、`num_blocks=3`、`d_model=128`、`large_size=51`、`epochs=2`、`lr=0.0003`、`early_stop_patience=10` |
| OmniAnomaly | `Imputer(method="constant", value=0)` → `MinMaxScaler` | `win_len=100`、`mc_samples=1`、`hidden_dim=500`、`dense_dim=500`、`n_flows=20`、`epochs=10`、`early_stop_patience=0` |
| TimesNet | `StandardScaler` | — |
| TranAD | `MinMaxScaler` | `lr=0.006` |
| USAD | `MinMaxScaler` | `win_len=32`、`epochs=100` |

## 来源与适配说明

| 检测器 | 来源和需注意的差异 |
|---|---|
| AnomalyTransformer | 使用作者的 [CLI 默认值](https://github.com/thuml/Anomaly-Transformer/blob/main/main.py)、[模型参数](https://github.com/thuml/Anomaly-Transformer/blob/main/model/AnomalyTransformer.py) 和 [solver](https://github.com/thuml/Anomaly-Transformer/blob/main/solver.py)。引用 [数据加载器](https://github.com/thuml/Anomaly-Transformer/blob/main/data_factory/data_loader.py) 的标准化步骤。 |
| COUTA | 项目默认用窗口 64 和三块 16 通道。原作者的 [ASD 配置](https://github.com/xuhongzuo/couta/blob/main/configs/COUTA.yaml) 搭配 [构造默认值](https://github.com/xuhongzuo/couta/blob/main/src/algorithms/couta_algo.py) 使用窗口 100 和一个 32 通道块。[ASD 加载流程](https://github.com/xuhongzuo/couta/blob/main/main_utils.py) 先做 [min-max 缩放](https://github.com/xuhongzuo/couta/blob/main/src/utils_general.py)，再做线性插值。 |
| DAGMM | [原论文 §4.3](https://bzong.github.io/doc/iclr18-dagmm.pdf) 提供学习率和损失权重。切窗及网络尺寸由 LibraryTS 实现决定。 |
| Donut | 使用作者的 [模型示例](https://github.com/NetManAIOps/donut)、[训练默认值](https://github.com/NetManAIOps/donut/blob/master/donut/training.py) 和 [推理默认值](https://github.com/NetManAIOps/donut/blob/master/donut/prediction.py)。CLI 标准化步骤可用于完整的无标签序列，原版对缺失值和已标注异常的处理见作者示例。 |
| FCVAE | 使用作者的 [默认模型和训练参数](https://github.com/CSTCloudOps/FCVAE/blob/main/model.py) 和 [早停设置](https://github.com/CSTCloudOps/FCVAE/blob/main/train.py)。原版 [数据加载器](https://github.com/CSTCloudOps/FCVAE/blob/main/dataset.py) 补齐时间戳、缩放和裁剪，再将缺失位置置零。LibraryTS 在训练段拟合缩放，并由 Imputer 传递缺失掩码。 |
| FITS | 使用作者的 [异常检测 CLI](https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/AD/main.py)、[数据加载器](https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/AD/data_loader.py) 和 [训练循环](https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/AD/solver_recon.py)。旧参考组合用窗口 100、降采样倍数 4、cutoff 12。当前 LibraryTS 默认组合为 64 / 2 / 16。 |
| IForest | [原论文 §4.1](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf#page=6) 使用 100 棵树和 256 个子采样样本。LibraryTS 通过 [sklearn 的 `max_samples="auto"`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) 实现。滞后窗口和逐滞后维缩放属于 LibraryTS 的适配。 |
| KANAD | 使用作者的 [默认配置](https://github.com/CSTCloudOps/KAN-AD/blob/main/kanad/config.toml) 和 [训练循环](https://github.com/CSTCloudOps/KAN-AD/blob/main/kanad/kanad.py)。[EasyTSAD 加载器](https://github.com/dawnvince/EasyTSAD/blob/main/EasyTSAD/DataFactory/LoadData.py) 先用 `lag=1, order=1` 差分，再做 z-score。检测器自身的 `order` 控制余弦基项数。 |
| ModernTCN | 使用作者的 [SWaT 脚本](https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-detection/scripts/SWaT.sh) 和 [数据加载器](https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-detection/data_provider/data_loader.py) 中的标准化。 |
| OmniAnomaly | 使用作者的 [ExpConfig](https://github.com/NetManAIOps/OmniAnomaly/blob/master/main.py)。[数据处理](https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/utils.py) 先将缺失值置零，再做 min-max 缩放。 |
| TimesNet | 使用作者的 [SMD 异常检测脚本](https://github.com/thuml/Time-Series-Library/blob/main/scripts/anomaly_detection/SMD/TimesNet.sh)、[训练默认值](https://github.com/thuml/Time-Series-Library/blob/main/run.py) 和 [标准化](https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_loader.py)。 |
| TranAD | 使用作者的 [UCR 学习率](https://github.com/imperial-qore/TranAD/blob/main/src/constants.py)、[模型定义](https://github.com/imperial-qore/TranAD/blob/main/src/models.py) 和 [训练设置](https://github.com/imperial-qore/TranAD/blob/main/main.py)。原版 [UCR 数据处理](https://github.com/imperial-qore/TranAD/blob/main/preprocess.py) 对整条序列做 min-max 缩放。独立测试实验应使用训练段拟合的缩放。 |
| USAD | 项目参考配置用窗口 32 和批大小 128。作者的 [SWaT 示例](https://github.com/manigalati/usad/blob/master/USAD.ipynb) 使用窗口 12、批大小 7919，做 min-max 缩放，并按数据尺寸计算潜变量宽度。 |
| CBLOF、COPOD、ECOD、HBOS、KNN、LOF、MCD、OCSVM | 使用 [PyOD 上游构造函数](https://github.com/yzhao062/pyod/tree/master/pyod/models) 中由 LibraryTS 包装类暴露的参数。每个时间点是一条样本。 |

## 常用超参数

- `win_len` 以采样点计。窗口检测器需要足够长的训练段；使用验证或测试段打分时也要考虑窗口长度。
- `early_stop_patience` 按验证轮数计算，`0` 表示关闭。正值是否生效取决于检测器及是否传入 `data.val`。构造默认值以 `ls --name` 为准。当前默认 `3` 的模型是 AnomalyTransformer、FITS、KANAD、ModernTCN 和 TimesNet，默认 `5` 的模型是 FCVAE，默认 `7` 的模型是 BeatGAN、COUTA、Donut、LSTMAD 和 OmniAnomaly。DAGMM、TranAD 和 USAD 默认关闭。
- Donut 的 `h_dim` 可用列表表示每层的宽度。`mc_samples` 控制每个窗口的推理采样次数，同时影响计算开销。
- FCVAE 的 `kernel_size` 和 `stride` 控制局部频率条件的窗口和步长。`missing_data_rate`、`point_ano_rate`、`seg_ano_rate` 控制训练增强。
- IForest 的 `win_len` 决定每个样本包含几个滞后值。`max_samples` 决定每棵树训练用的样本数。
