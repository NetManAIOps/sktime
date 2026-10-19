import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd


class ForecastDataset(Dataset):
    def __init__(
        self,
        raw_seqs,
        win_len,
        pred_len,
        stride=1,
        scale_cfg=None,
        clip=True,
        flag="train",
        labels=None,
        scale_mode="zscore",
    ):
        super().__init__()
        assert flag in ["train", "val", "test"]
        self._win_len = win_len
        self._pred_len = pred_len
        self.stride = stride
        self.flag = flag
        self.cfg = scale_cfg if scale_cfg is not None else {}

        raw = np.asarray(raw_seqs, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        mode = str(scale_mode).strip().lower()
        if mode not in {"zscore", "minmax"}:
            raise ValueError(f"Unsupported scale_mode: {scale_mode}")

        if flag == "train":
            if mode == "zscore":
                mean = float(np.nanmean(raw)) if raw.size else 0.0
                std = float(np.nanstd(raw)) if raw.size else 1.0
                if not np.isfinite(mean):
                    mean = 0.0
                if not np.isfinite(std) or std <= 0:
                    std = 1.0
                self.cfg = {"scale_mode": "zscore", "mean": mean, "std": std}
                raw = (raw - mean) / std
            else:
                data_min = float(np.nanmin(raw)) if raw.size else 0.0
                data_max = float(np.nanmax(raw)) if raw.size else 1.0
                if not np.isfinite(data_min) or not np.isfinite(data_max):
                    data_min, data_max = 0.0, 1.0
                self.cfg = {"scale_mode": "minmax", "min": data_min, "max": data_max}
                denom = data_max - data_min
                if denom <= 0:
                    raw = np.zeros_like(raw, dtype=float)
                else:
                    raw = (raw - data_min) / denom
                if clip:
                    raw = np.clip(raw, 0.0, 1.0)
        else:
            cfg_mode = str(self.cfg.get("scale_mode", mode)).strip().lower()
            if cfg_mode == "zscore":
                if "mean" not in self.cfg or "std" not in self.cfg:
                    raise ValueError("scale_cfg with 'mean' and 'std' is required for val/test in zscore mode")
                mean = float(self.cfg["mean"])
                std = float(self.cfg["std"])
                if std <= 0:
                    std = 1.0
                raw = (raw - mean) / std
            elif cfg_mode == "minmax":
                if "min" not in self.cfg or "max" not in self.cfg:
                    raise ValueError("scale_cfg with 'min' and 'max' is required for val/test in minmax mode")
                data_min = float(self.cfg["min"])
                data_max = float(self.cfg["max"])
                denom = data_max - data_min
                if denom <= 0:
                    raw = np.zeros_like(raw, dtype=float)
                else:
                    raw = (raw - data_min) / denom
                if clip:
                    raw = np.clip(raw, 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported scale_mode in scale_cfg: {cfg_mode}")

        self.raw_seqs = raw
        self.sample_num = max((self.raw_seqs.shape[0] - win_len - pred_len) // stride + 1, 0)

        self._labels = None
        if labels is not None:
            labels = np.asarray(labels, dtype=bool).reshape(-1)
            if labels.shape[0] != self.raw_seqs.shape[0]:
                raise ValueError("labels length must match raw_seqs length")
            self._labels = labels

        if self._labels is None:
            self.samples, self.targets = self._generate_samples()
            self.labels = None
        else:
            self.samples, self.targets, self.labels = self._generate_samples_with_labels()

    def _generate_samples(self):
        data = torch.tensor(self.raw_seqs, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        X = torch.stack([data[i : i + self._win_len] for i in indices])
        Y = torch.stack([data[i + self._win_len : i + self._win_len + self._pred_len] for i in indices])
        return X, Y

    def _generate_samples_with_labels(self):
        data = torch.tensor(self.raw_seqs, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        X = torch.stack([data[i : i + self._win_len] for i in indices])
        Y = torch.stack([data[i + self._win_len : i + self._win_len + self._pred_len] for i in indices])
        y = torch.tensor(
            [self._labels[i + self._win_len + self._pred_len - 1] for i in indices],
            dtype=torch.bool,
        )
        return X, Y, y

    def get_scale_cfg(self):
        return self.cfg

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        if self.flag in ["val", "test"] and self.labels is not None:
            return self.samples[index], self.targets[index], self.labels[index]
        return self.samples[index], self.targets[index]


def split_train_val(df, val_frac, time_col='timestamp', value_col='value'):
    if not (0 < val_frac < 1):
        raise ValueError("val_frac must be in (0, 1)")

    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"df must contain columns '{time_col}' and '{value_col}'")

    work = df.copy()
    n = len(work)
    val_len = max(1, int(round(n * val_frac)))
    train_df = work.iloc[:-val_len].copy()
    val_df = work.iloc[-val_len:].copy()
    return train_df, val_df


class ReconDataset(Dataset):
    def __init__(
        self,
        raw_seqs,
        labels=None,
        win_len=20,
        flag='train',
        scale_cfg=None,
        mask=None,
        clip=True,
        clip_value=0,
        scale_mode="zscore",
    ):
        super().__init__()
        assert flag in ['train', 'val', 'test']
        self._win_len = win_len
        self.flag = flag
        self.cfg = scale_cfg if scale_cfg is not None else {}

        raw = np.asarray(raw_seqs, dtype=float).reshape(-1)
        if raw.size == 0:
            raw = np.zeros(1, dtype=float)

        mode = str(scale_mode).strip().lower()
        if mode not in {"zscore", "minmax"}:
            raise ValueError(f"Unsupported scale_mode: {scale_mode}")
        
        mask_array = None
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool).reshape(-1)
            if mask_array.shape[0] != raw.shape[0]:
                raise ValueError("mask length must match raw_seqs length")
            valid = raw[~mask_array]
        else:
            valid = raw

        if flag == "train":
            if mode == "zscore":
                mean = float(np.nanmean(valid)) if valid.size else 0.0
                std = float(np.nanstd(valid)) if valid.size else 1.0
                if not np.isfinite(mean):
                    mean = 0.0
                if not np.isfinite(std) or std <= 0:
                    std = 1.0
                self.cfg = {"scale_mode": "zscore", "mean": mean, "std": std}
                raw = (raw - mean) / std
            else:
                data_min = float(np.nanmin(valid)) if valid.size else 0.0
                data_max = float(np.nanmax(valid)) if valid.size else 1.0
                if not np.isfinite(data_min) or not np.isfinite(data_max):
                    data_min, data_max = 0.0, 1.0
                self.cfg = {"scale_mode": "minmax", "min": data_min, "max": data_max}
                denom = data_max - data_min
                if denom <= 0:
                    raw = np.zeros_like(raw, dtype=float)
                else:
                    raw = (raw - data_min) / denom
                if clip:
                    raw = np.clip(raw, -clip_value, 1.0 + clip_value)
        else:
            cfg_mode = str(self.cfg.get("scale_mode", mode)).strip().lower()
            if cfg_mode == "zscore":
                if "mean" not in self.cfg or "std" not in self.cfg:
                    raise ValueError("scale_cfg with 'mean' and 'std' is required for val/test in zscore mode")
                mean = float(self.cfg["mean"])
                std = float(self.cfg["std"])
                if std <= 0:
                    std = 1.0
                raw = (raw - mean) / std
            elif cfg_mode == "minmax":
                if "min" not in self.cfg or "max" not in self.cfg:
                    raise ValueError("scale_cfg with 'min' and 'max' is required for val/test in minmax mode")
                data_min = float(self.cfg["min"])
                data_max = float(self.cfg["max"])
                denom = data_max - data_min
                if denom <= 0:
                    raw = np.zeros_like(raw, dtype=float)
                else:
                    raw = (raw - data_min) / denom
                if clip:
                    raw = np.clip(raw, -clip_value, 1.0 + clip_value)
            else:
                raise ValueError(f"Unsupported scale_mode in scale_cfg: {cfg_mode}")
        
        if mask_array is not None:
            raw[mask_array] = 0

        self._raw_seqs = torch.tensor(raw, dtype=torch.float32)
        self._labels = None
        if labels is not None:
            labels = np.asarray(labels, dtype=bool).reshape(-1)
            if labels.shape[0] != raw.shape[0]:
                raise ValueError("labels length must match raw_seqs length")
            self._labels = torch.tensor(labels, dtype=torch.bool)
        self._mask = torch.tensor(mask_array, dtype=torch.bool) if mask_array is not None else None

    def get_scale_cfg(self):
        return self.cfg

    def __len__(self):
        return max(len(self._raw_seqs) - self._win_len + 1, 0)

    def __getitem__(self, index):
        x = self._raw_seqs[index : index + self._win_len]
        if self._mask is not None:
            mask = self._mask[index : index + self._win_len]
            if self.flag in ['val', 'test'] and self._labels is not None:
                y = self._labels[index + self._win_len - 1]
                return x, y, mask
            return x, mask

        if self.flag in ['val', 'test'] and self._labels is not None:
            y = self._labels[index + self._win_len - 1]
            return x, y

        return x
