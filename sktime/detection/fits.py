"""FITS frequency interpolation, ported from the author's AD implementation.

Source: https://github.com/VEWOXIC/FITS/tree/d040bb015b6299da26d879b90dd19c80fb72c160/AD
License: Apache-2.0 (see FITS.LICENSE).

The single-channel network, full-window MSE and Adam schedule follow upstream.
DevAD uses stride-1 ReconDataset windows, train-only scaling and external
validation. Each window's mean squared error is assigned to its endpoint;
output contains endpoint reconstructions, not the whole reconstructed window.
"""

from __future__ import annotations


import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping


class FITSBackbone(nn.Module):
    """Reconstruct [B, win_len] from uniformly subsampled [B, input_len]."""

    def __init__(self, input_len: int, win_len: int, cut_freq: int):
        super().__init__()
        self.win_len = win_len
        self.cut_freq = cut_freq
        self.length_ratio = win_len / input_len
        self.freq_upsampler = nn.Linear(
            cut_freq, int(cut_freq * self.length_ratio),
        ).to(torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = x.mean(dim=1, keepdim=True)
        x = x - mean
        variance = x.var(dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(variance)

        low_spectrum = torch.fft.rfft(x, dim=1)[:, :self.cut_freq, :]
        expanded = self.freq_upsampler(low_spectrum.transpose(1, 2)).transpose(1, 2)
        spectrum = expanded.new_zeros((x.shape[0], self.win_len // 2 + 1, 1))
        spectrum[:, :expanded.shape[1], :] = expanded
        # Explicit n also preserves odd output lengths (upstream omits it).
        reconstruction = torch.fft.irfft(spectrum, n=self.win_len, dim=1)
        reconstruction = reconstruction * self.length_ratio
        return (reconstruction * torch.sqrt(variance) + mean).squeeze(-1)


class FITS(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        batch_size=1024,
        win_len=100,
        downsample_rate=4,
        cut_freq=12,
        epochs=10,
        lr=0.0001,
        early_stop_patience=3,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.win_len = win_len
        self.downsample_rate = downsample_rate
        self.cut_freq = cut_freq
        self.epochs = epochs
        self.lr = lr
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        win_len = int(self.win_len)
        downsample_rate = int(self.downsample_rate)
        cut_freq = int(self.cut_freq)
        input_len = win_len // downsample_rate

        self.criterion_ = nn.MSELoss()
        self.backend_ = FITSBackbone(
            input_len=input_len,
            win_len=win_len,
            cut_freq=cut_freq,
        )

        return self.backend_

    def _get_dataloader(self, x, flag="train"):
        dataset = ReconDataset(x, win_len=int(self.win_len))
        return DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=flag == "train",
            drop_last=False,
        )

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        self._set_seed()
        train_loader = self._get_dataloader(X, flag="train")
        optimizer = torch.optim.Adam(self.backend_.parameters(), lr=self.lr)
        downsample_rate = int(self.downsample_rate)
        early_stopping = None
        if self.early_stop_patience > 0 and X_val is not None:
            early_stopping = EarlyStopping(
                inclusive=True,
                mode="min",
                patience=int(self.early_stop_patience),
                delta=float(self.early_stop_delta),
            )

        epochs = int(self.epochs)
        for epoch in range(epochs):
            self.backend_.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for batch in train_loader:
                batch = batch.to(self.device_).squeeze(-1)
                optimizer.zero_grad()
                reconstruction = self.backend_(batch[:, ::downsample_rate])
                loss = self.criterion_(reconstruction, batch)
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())

            val_loss = None
            if X_val is not None:
                val_loss = self._valid_loss(X_val)
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

            # Author schedule: first epoch keeps lr, final update is epoch 26.
            if epoch <= 25:
                for group in optimizer.param_groups:
                    group["lr"] = self.lr * (0.95 ** epoch)


        if self.best_epoch_ is None:
            self.best_epoch_ = int(self.epochs)

    def _valid_loss(self, x_val: np.ndarray) -> float:
        val_loader = self._get_dataloader(x_val, flag="val")
        downsample_rate = int(self.downsample_rate)
        losses = []
        self.backend_.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device_).squeeze(-1)
                reconstruction = self.backend_(batch[:, ::downsample_rate])
                losses.append(self.criterion_(reconstruction, batch).item())
        return float(np.mean(losses))

    def _transform_scores(self, x_test: np.ndarray, **kwargs):
        loader = self._get_dataloader(x_test, flag="test")
        downsample_rate = int(self.downsample_rate)
        scores, outputs = [], []
        self.backend_.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device_).squeeze(-1)
                reconstruction = self.backend_(batch[:, ::downsample_rate])
                score = (reconstruction - batch).square().mean(dim=1)
                scores.append(score.cpu().numpy())
                outputs.append(reconstruction[:, -1].cpu().numpy())
        return np.pad(
            np.asarray(np.concatenate(scores), dtype=float),
            (int(self.win_len) - 1, 0), constant_values=0,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
