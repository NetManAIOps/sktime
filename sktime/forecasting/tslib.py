# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Forecasters wrapping THUML Time-Series-Library (TSLib) deep models.

This module adapts models vendored under ``sktime.libs.tslib`` (from
https://github.com/thuml/Time-Series-Library, MIT license) as sktime
forecasters. Each estimator builds the corresponding TSLib model from a
configs namespace and trains it with a sliding-window pipeline equivalent to
TSLib's ``exp/exp_long_term_forecasting.py``.
"""

__all__ = [
    "AutoformerForecaster",
    "DLinearForecaster",
    "FEDformerForecaster",
    "FreTSForecaster",
    "ITransformerForecaster",
    "TimesNetForecaster",
]

__author__ = ["xiezhe"]

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
Dataset = _safe_import("torch.utils.data.Dataset")
DataLoader = _safe_import("torch.utils.data.DataLoader")

# Mark width produced by tslib.utils.timefeatures.time_features per freq string.
_FREQ_MARK_DIM = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}


class BaseTSLibForecaster(BaseDeepNetworkPyTorch):
    """Base class for TSLib model adapters.

    Parameters
    ----------
    seq_len : int, default=36
        Length of the input (lookback) window.
    label_len : int, default=18
        Length of the decoder's overlap (label) section; only used by
        encoder-decoder models (Autoformer, FEDformer).
    pred_len : int, default=12
        Forecast horizon the network is built for. ``fit`` enlarges it to the
        maximum ``fh`` passed there, if larger. ``predict`` raises for fh
        values beyond it.
    d_model : int, default=64
        Model hidden dimension.
    n_heads : int, default=8
        Number of attention heads (attention-based models).
    e_layers : int, default=2
        Number of encoder layers.
    d_layers : int, default=1
        Number of decoder layers (encoder-decoder models).
    d_ff : int, default=128
        Feed-forward dimension.
    factor : int, default=3
        Attention factor (Autoformer/FEDformer).
    dropout : float, default=0.1
        Dropout rate.
    embed : str, default="timeF"
        Temporal embedding type ("timeF", "fixed", "learned").
    freq : str, default="h"
        Frequency string for time features.
    activation : str, default="gelu"
        Activation function ("gelu" or "relu").
    moving_avg : int, default=25
        Moving-average kernel for series decomposition (DLinear, Autoformer,
        FEDformer).
    top_k : int, default=3
        Number of dominant periods (TimesNet).
    num_kernels : int, default=6
        Inception block kernels (TimesNet).
    channel_independence : int, default=1
        1 = model channels independently (FreTS, iTransformer style).
    num_epochs : int, default=3
        Training epochs.
    batch_size : int, default=32
        Training batch size.
    lr : float, default=1e-3
        Learning rate.
    criterion : str, optional
        Loss name, one of BaseDeepNetworkPyTorch.criterions; default MSELoss.
    optimizer : str, optional
        Optimizer name, one of BaseDeepNetworkPyTorch.optimizers; default Adam.
    """

    _tags = {
        "authors": ["xiezhe", "thuml"],
        "maintainers": ["xiezhe"],
        # fh may also be supplied at predict time; fit falls back to pred_len
        "requires-fh-in-fit": False,
        # "python_dependencies": ["torch"] inherited from BaseDeepNetworkPyTorch
    }

    _model_name = None  # upstream module name in sktime.libs.tslib.models

    def __init__(
        self,
        seq_len=36,
        label_len=18,
        pred_len=12,
        d_model=64,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        factor=3,
        dropout=0.1,
        embed="timeF",
        freq="h",
        activation="gelu",
        moving_avg=25,
        top_k=3,
        num_kernels=6,
        channel_independence=1,
        num_epochs=3,
        batch_size=32,
        lr=1e-3,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.moving_avg = moving_avg
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.channel_independence = channel_independence

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
        )
        self.criterion = criterion

    # ------------------------------------------------------------------
    # network construction
    # ------------------------------------------------------------------
    def _build_network(self, fh):
        """Instantiate the vendored TSLib model from a configs namespace."""
        pred_len = max(int(self.pred_len), int(fh))
        module = importlib.import_module(
            f"sktime.libs.tslib.models.{self._model_name}"
        )
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            features="M",
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=pred_len,
            enc_in=self._n_channels,
            dec_in=self._n_channels,
            c_out=self._n_channels,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            factor=self.factor,
            dropout=self.dropout,
            embed=self.embed,
            freq=self.freq,
            activation=self.activation,
            moving_avg=self.moving_avg,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            channel_independence=self.channel_independence,
            output_attention=False,
            num_class=2,
        )
        network = module.Model(configs)
        network.seq_len = self.seq_len
        network.pred_len = pred_len
        return network

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def _time_marks(self, index):
        """Time features for a pandas index, or None when not datetime-like."""
        if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
            return None
        try:
            from sktime.libs.tslib.utils.timefeatures import time_features

            dates = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
            return time_features(pd.DatetimeIndex(dates), freq=self.freq).T.astype(
                np.float32
            )
        except Exception:
            width = _FREQ_MARK_DIM.get(self.freq, 4)
            return np.zeros((len(index), width), dtype=np.float32)

    def _scale_fit(self, values):
        self._mean = values.mean(axis=0, keepdims=True)
        self._std = values.std(axis=0, keepdims=True)
        self._std[self._std == 0] = 1.0
        return (values - self._mean) / self._std

    def build_pytorch_train_dataloader(self, y):
        """Sliding-window dataset over (seq_len + label_len + pred_len)."""
        values = self._scale_fit(y.to_numpy(dtype=np.float32))
        marks = self._time_marks(y.index)
        dataset = _TSLibWindowDataset(
            values=values,
            marks=marks,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.network.pred_len,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"Series of length {len(y)} is too short for seq_len="
                f"{self.seq_len} + pred_len={self.network.pred_len}."
            )
        return DataLoader(dataset, self.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def _fit(self, y, fh, X=None):
        """Fit the TSLib model; fh may be None (pred_len fallback)."""
        fh_max = 0
        if fh is not None:
            fh_max = int(max(fh.to_relative(self.cutoff)._values))

        self._n_channels = y.shape[1]
        self._y_len = len(y)
        self.network = self._build_network(fh_max)
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        dataloader = self.build_pytorch_train_dataloader(y)
        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)
        return self

    def _run_epoch(self, epoch, dataloader):
        for x_enc, x_mark, x_dec, x_mark_dec, y_true in dataloader:
            y_pred = self.network(x_enc, x_mark, x_dec, x_mark_dec)
            loss = self._criterion(y_pred, y_true)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _predict(self, fh=None, X=None):
        """Forecast from the last seq_len observations."""
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        fh_values = fh._values.values
        if max(fh_values) > self.network.pred_len or min(fh_values) < 0:
            raise ValueError(
                f"fh of {list(fh_values)} passed to {self.__class__.__name__} is not "
                f"within `pred_len` ({self.network.pred_len}). Please use a fh that "
                "aligns with the `pred_len` of the forecaster."
            )

        y = self._y.to_numpy(dtype=np.float32)
        y = (y - self._mean) / self._std
        marks = self._time_marks(self._y.index)
        dataset = _TSLibWindowDataset(
            values=y,
            marks=marks,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.network.pred_len,
            predict_mode=True,
        )
        x_enc, x_mark, x_dec, x_mark_dec, _ = dataset[0]

        self.network.eval()
        with torch.no_grad():
            out = self.network(
                x_enc.unsqueeze(0),
                x_mark.unsqueeze(0) if x_mark is not None else None,
                x_dec.unsqueeze(0),
                x_mark_dec.unsqueeze(0) if x_mark_dec is not None else None,
            )
        y_pred = out[0, :, :].numpy() * self._std + self._mean
        y_pred = y_pred[fh_values - 1]
        return pd.DataFrame(
            y_pred, columns=self._y.columns, index=fh.to_absolute_index(self.cutoff)
        )


class _TSLibWindowDataset(Dataset):
    """Sliding windows in TSLib's (seq_len, label_len, pred_len) layout."""

    def __init__(self, values, marks, seq_len, label_len, pred_len, predict_mode=False):
        self.values = values
        self.marks = marks
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.predict_mode = predict_mode

    def __len__(self):
        if self.predict_mode:
            return 1
        return max(len(self.values) - self.seq_len - self.pred_len + 1, 0)

    def _mark(self, sl):
        if self.marks is None:
            return None
        return torch.from_numpy(self.marks[sl]).float()

    def __getitem__(self, i):
        if self.predict_mode:
            i = max(len(self.values) - self.seq_len, 0)
        s_end = i + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        x_enc = torch.from_numpy(self.values[i:s_end]).float()
        x_mark = self._mark(slice(i, s_end))

        x_dec = torch.zeros(self.label_len + self.pred_len, self.values.shape[1])
        x_dec[: self.label_len] = torch.from_numpy(self.values[r_begin:s_end]).float()

        if self.predict_mode:
            # future marks are unknown; reuse the last known mark row
            if self.marks is not None:
                last = self.marks[-1:]
                reps = self.label_len + self.pred_len
                hist = self.marks[r_begin:s_end]
                x_mark_dec = torch.from_numpy(
                    np.concatenate([hist, np.repeat(last, r_end - s_end, axis=0)])
                ).float()
            else:
                x_mark_dec = None
            y_true = torch.zeros(self.pred_len, self.values.shape[1])
        else:
            x_mark_dec = self._mark(slice(r_begin, r_end))
            y_true = torch.from_numpy(
                self.values[s_end : s_end + self.pred_len]
            ).float()

        return x_enc, x_mark, x_dec, x_mark_dec, y_true


class DLinearForecaster(BaseTSLibForecaster):
    """DLinear (NLinear-style decomposition linear) forecaster from TSLib.

    Vendored from THUML Time-Series-Library (models/DLinear.py) [1]_.

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023. Are transformers effective for
       time series forecasting? AAAI 2023.
    """

    _model_name = "DLinear"


class TimesNetForecaster(BaseTSLibForecaster):
    """TimesNet forecaster from TSLib (temporal 2D-variation modeling) [1]_.

    References
    ----------
    .. [1] Wu H, Hu T, Liu Y, et al. 2023. TimesNet: Temporal 2D-Variation
       Modeling for General Time Series Analysis. ICLR 2023.
    """

    _model_name = "TimesNet"


class ITransformerForecaster(BaseTSLibForecaster):
    """iTransformer forecaster from TSLib (inverted transformer) [1]_.

    References
    ----------
    .. [1] Liu Y, Hu T, Zhang H, et al. 2024. iTransformer: Inverted
       Transformers Are Effective for Time Series Forecasting. ICLR 2024.
    """

    _model_name = "iTransformer"


class AutoformerForecaster(BaseTSLibForecaster):
    """Autoformer forecaster from TSLib (auto-correlation mechanism) [1]_.

    References
    ----------
    .. [1] Wu H, Xu J, Wang J, Long M. 2021. Autoformer: Decomposition
       Transformers with Auto-Correlation for Long-Term Series Forecasting.
       NeurIPS 2021.
    """

    _model_name = "Autoformer"


class FEDformerForecaster(BaseTSLibForecaster):
    """FEDformer forecaster from TSLib (frequency-enhanced transformer) [1]_.

    References
    ----------
    .. [1] Zhou T, Ma Z, Wen Q, et al. 2022. FEDformer: Frequency Enhanced
       Decomposed Transformer for Long-term Series Forecasting. ICML 2022.
    """

    _model_name = "FEDformer"


class FreTSForecaster(BaseTSLibForecaster):
    """FreTS forecaster from TSLib (frequency-domain MLP) [1]_.

    References
    ----------
    .. [1] Yi K, Zhang Q, Fan W, et al. 2024. FreTS: Frequency-domain MLPs are
       More Effective Learners in Time Series Forecasting. NeurIPS 2023.
    """

    _model_name = "FreTS"
