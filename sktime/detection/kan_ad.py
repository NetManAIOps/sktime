"""KAN-AD: one-step prediction of standardized first differences.

Ported from issaccv/KAN-AD, commit 7c3a0d96b77e0d609660a240c2b11db084ba523f.
MIT license: see KANAD.LICENSE. Network, MSE/Adam/StepLR and absolute-error
scoring follow the author. Differencing and scaling belong to upstream
transformers: difference each split independently, with a zero first value,
then apply a scaler fitted on training differences only. History length W
yields N-W scores, starting at input index W. No original-value output is returned.
"""

from __future__ import annotations


import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ForecastDataset
from sktime.detection.utils.early_stopping import EarlyStopping


class KANADBackbone(nn.Module):
    def __init__(self, window: int, order: int):
        super().__init__()
        self.window = window
        self.order = order
        self.channels = 2 * order + 1
        self.register_buffer("orders", self._create_custom_periodic_cosine().unsqueeze(0))
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, window)

    def _create_custom_periodic_cosine(self):
        basis = torch.empty(self.order, self.window, dtype=torch.float32)
        for i, frequency in enumerate(range(1, self.order + 1)):
            position = torch.arange(self.window, dtype=torch.float32)
            basis[i] = torch.cos(2 * torch.pi * position * frequency / self.window)
        return basis

    def forward(self, x):
        # x: [B, W]; output: [B, 1].
        raw = x.unsqueeze(1)
        features = torch.cat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [torch.cos(order * raw) for order in range(1, self.order + 1)]
            + [raw],
            dim=1,
        )
        hidden = self.act(self.bn1(self.init_conv(features)))
        hidden = self.act(self.bn2(self.inner_conv(hidden) + features))
        hidden = self.act(self.bn3(self.out_conv(hidden) + raw))
        return self.final_conv(hidden).squeeze(1)


class KANAD(BaseDeepDetector):
    """One-step predictor for already differenced and scaled univariate values.

    Preprocess with Differencer(lags=1, na_handling="fill_zero", memory="none")
    and a training-fitted scaling transformer before calling this detector.
    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, 
             "capability:missing_values": False}

    def __init__(
        self,
        win_len,
        order=2,
        batch_size=1024,
        epochs=100,
        lr=0.01,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.win_len = win_len
        self.order = order
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.backend_ = KANADBackbone(
            window=self.win_len, order=self.order,
        )
        self.criterion_ = nn.MSELoss()

        return self.backend_

    def _get_dataloader(self, X, shuffle=False):
        dataset = ForecastDataset(X, win_len=self.win_len, pred_len=1)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=False,
        )

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        self._set_seed()
        train_loader = self._get_dataloader(X, shuffle=True)
        valid_loader = None
        if X_val is not None:
            valid_loader = self._get_dataloader(X_val)

        optimizer = torch.optim.Adam(self.backend_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
        early_stopping = None
        if self.early_stop_patience > 0 and X_val is not None:
            early_stopping = EarlyStopping(
                inclusive=True,
                patience=self.early_stop_patience,
                delta=self.early_stop_delta,
            )

        epochs = self.epochs
        for epoch in range(epochs):
            self.backend_.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x, target in train_loader:
                x = x.to(self.device_).squeeze(-1)
                target = target.to(self.device_).squeeze(-1)
                optimizer.zero_grad()
                loss = self.criterion_(self.backend_(x), target)
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())
            val_loss = None
            if valid_loader is not None:
                val_loss = self._valid_loss(valid_loader)
            scheduler.step()
            if early_stopping is not None:
                improved = early_stopping.update(val_loss)
                if improved:
                    self._save_best_weights(epoch + 1)
            stopped_early = early_stopping is not None and early_stopping.early_stop
            reporter.end_epoch(
                val_loss=val_loss,
                best_epoch=self.best_epoch_,
                early_stopping=early_stopping,
            )
            if stopped_early:
                break


        if self.best_epoch_ is None:
            self.best_epoch_ = int(self.epochs)

    def _valid_loss(self, loader):
        self.backend_.eval()
        losses = []
        with torch.no_grad():
            for x, target in loader:
                x = x.to(self.device_).squeeze(-1)
                target = target.to(self.device_).squeeze(-1)
                losses.append(self.criterion_(self.backend_(x), target).item())
        return float(np.mean(losses))

    def _transform_scores(self, x_test, **kwargs):
        loader = self._get_dataloader(x_test)
        self.backend_.eval()
        scores = []
        with torch.no_grad():
            for x, target in loader:
                x = x.to(self.device_).squeeze(-1)
                target = target.to(self.device_).squeeze(-1)
                scores.append((self.backend_(x) - target).abs().squeeze(-1).cpu().numpy())
        return np.pad(
            np.asarray(np.concatenate(scores), dtype=float),
            (self.win_len, 0), constant_values=0,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
