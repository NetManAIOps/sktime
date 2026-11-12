# OmniAnomaly

用循环 VAE 和 normalizing flows 重构窗口，以末点负对数似然打分。

## 调参起点

先选历史窗口 `win_len`，再按容量和耗时比较 `latent_dim`、`n_flows`。比较训练方案时固定推理采样数。

## 适用条件

推荐先用常数 0 填补缺失，再用训练段拟合的 min-max 缩放。

## 实验记录

暂无经验证的记录。
