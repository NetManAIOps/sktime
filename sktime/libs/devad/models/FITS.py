"""FITS frequency interpolation, ported from the author's AD implementation.

Source: https://github.com/VEWOXIC/FITS/tree/d040bb015b6299da26d879b90dd19c80fb72c160/AD
License: Apache-2.0 (see FITS.LICENSE).

The single-channel network, full-window MSE and Adam schedule follow upstream.
DevAD uses stride-1 ReconDataset windows, train-only scaling and external
validation. Each window's mean squared error is assigned to its endpoint;
output contains endpoint reconstructions, not the whole reconstructed window.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .Base import BaseModel, DetectResult
from ..utils.dataset import ReconDataset
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


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


class FITS(BaseModel):
    HP = {
        "batch_size": 1024,
        "win_len": 100,
        "downsample_rate": 4,
        "cut_freq": 12,
        "scale_mode": "zscore",
        "epochs": 10,
        "lr": 1e-4,
        "early_stop_patience": 3,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        win_len = int(self.params["win_len"])
        downsample_rate = int(self.params["downsample_rate"])
        cut_freq = int(self.params["cut_freq"])
        input_len = win_len // downsample_rate

        self.criterion = nn.MSELoss()
        self.backend = FITSBackbone(
            input_len=input_len,
            win_len=win_len,
            cut_freq=cut_freq,
        ).to(self.device)

    def _get_dataloader(self, x: np.ndarray, flag: str = "train"):
        win_len = int(self.params["win_len"])
        if len(x) < win_len:
            raise ValueError(f"FITS requires at least win_len={win_len} {flag} points")
        dataset = ReconDataset(
            x,
            win_len=win_len,
            flag=flag,
            scale_cfg=self.scale_cfg,
            scale_mode=self.params["scale_mode"],
            clip=False,
        )
        if flag == "train":
            self.scale_cfg = dataset.get_scale_cfg()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=flag == "train",
        )

    def _fit(
        self,
        x_train: np.ndarray,
        y: np.ndarray | None = None,
        x_val: np.ndarray | None = None,
        checkpoint_path=None,
        reporter: TrainingReporter | None = None,
    ) -> None:
        reporter = reporter or NullTrainingReporter()
        train_loader = self._get_dataloader(x_train, flag="train")
        if x_val is not None:
            x_val = self.check_array(x_val, dtype=np.float32, name="x_val")
        optimizer = torch.optim.Adam(self.backend.parameters(), lr=self.params["lr"])
        downsample_rate = int(self.params["downsample_rate"])
        early_stopping = None
        if self.params["early_stop_patience"] > 0 and checkpoint_path and x_val is not None:
            early_stopping = EarlyStopping(
                mode="min",
                patience=int(self.params["early_stop_patience"]),
                delta=float(self.params["early_stop_delta"]),
            )

        epochs = int(self.params["epochs"])
        for epoch in range(epochs):
            self.backend.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstruction = self.backend(batch[:, ::downsample_rate])
                loss = self.criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())

            val_loss = None
            if x_val is not None:
                val_loss = self._valid_loss(x_val)
            if early_stopping is not None:
                early_stopping(val_loss, self, checkpoint_path, epoch=epoch + 1)
            stopped_early = early_stopping is not None and early_stopping.early_stop
            reporter.end_epoch(
                val_loss=val_loss,
                best_epoch=self.best_epoch,
                early_stopping=early_stopping,
            )
            if stopped_early:
                break

            # Author schedule: first epoch keeps lr, final update is epoch 26.
            if epoch <= 25:
                for group in optimizer.param_groups:
                    group["lr"] = self.params["lr"] * (0.95 ** epoch)

        if early_stopping is not None and checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    def _valid_loss(self, x_val: np.ndarray) -> float:
        val_loader = self._get_dataloader(x_val, flag="val")
        downsample_rate = int(self.params["downsample_rate"])
        losses = []
        self.backend.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                reconstruction = self.backend(batch[:, ::downsample_rate])
                losses.append(self.criterion(reconstruction, batch).item())
        return float(np.mean(losses))

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        loader = self._get_dataloader(x_test, flag="test")
        downsample_rate = int(self.params["downsample_rate"])
        scores, outputs = [], []
        self.backend.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                reconstruction = self.backend(batch[:, ::downsample_rate])
                score = (reconstruction - batch).square().mean(dim=1)
                scores.append(score.cpu().numpy())
                outputs.append(reconstruction[:, -1].cpu().numpy())
        return DetectResult(
            scores=np.concatenate(scores),
            output=np.concatenate(outputs),
            start_pos=int(self.params["win_len"]) - 1,
        )
