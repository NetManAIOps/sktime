"""KAN-AD: one-step prediction of standardized first differences.

Ported from issaccv/KAN-AD, commit 7c3a0d96b77e0d609660a240c2b11db084ba523f.
MIT license: see KANAD.LICENSE. Network, MSE/Adam/StepLR and absolute-error
scoring follow the author. DevAD supplies external validation and checkpoint
handling. Each split is differenced independently, with a zero first value;
the scaler is fitted on training differences only. History length W yields
N-W scores, starting at original index W. No original-value output is returned.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .Base import BaseModel, DetectResult, REQUIRED
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


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


class KANAD(BaseModel):
    HP = {
        "win_len": REQUIRED,
        "order": 2,
        "batch_size": 1024,
        "epochs": 100,
        "lr": 0.01,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }
    STATEFUL_ATTRS = BaseModel.STATEFUL_ATTRS + ("scaler",)

    def __init__(self, config):
        super().__init__(config)
        self.backend = KANADBackbone(
            window=self.params["win_len"], order=self.params["order"],
        ).to(self.device)
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()

    def _get_dataloader(self, x: np.ndarray, flag: str = "train"):
        differences = np.diff(x, prepend=x[0]).reshape(-1, 1)
        if flag == "train":
            values = self.scaler.fit_transform(differences).ravel()
        else:
            values = self.scaler.transform(differences).ravel()
        win_len = self.params["win_len"]
        windows = np.lib.stride_tricks.sliding_window_view(values, win_len)[:-1]
        dataset = TensorDataset(
            torch.from_numpy(windows.copy()),
            torch.from_numpy(values[win_len:, None].copy()),
        )
        return DataLoader(
            dataset, batch_size=self.params["batch_size"], shuffle=flag == "train",
        )

    def _fit(
        self,
        x_train,
        y=None,
        x_val=None,
        checkpoint_path=None,
        reporter: TrainingReporter | None = None,
    ):
        reporter = reporter or NullTrainingReporter()
        train_loader = self._get_dataloader(x_train, "train")
        valid_loader = None
        if x_val is not None:
            x_val = self.check_array(x_val, dtype=np.float32, name="x_val")
            valid_loader = self._get_dataloader(x_val, "val")

        optimizer = torch.optim.Adam(self.backend.parameters(), lr=self.params["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
        early_stopping = None
        if self.params["early_stop_patience"] > 0 and checkpoint_path and x_val is not None:
            early_stopping = EarlyStopping(
                patience=self.params["early_stop_patience"],
                delta=self.params["early_stop_delta"],
            )

        epochs = self.params["epochs"]
        for epoch in range(epochs):
            self.backend.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x, target in train_loader:
                x, target = x.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.backend(x), target)
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())
            val_loss = None
            if valid_loader is not None:
                val_loss = self._valid_loss(valid_loader)
            scheduler.step()
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

        if early_stopping is not None and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    def _valid_loss(self, loader):
        self.backend.eval()
        losses = []
        with torch.no_grad():
            for x, target in loader:
                x, target = x.to(self.device), target.to(self.device)
                losses.append(self.criterion(self.backend(x), target).item())
        return float(np.mean(losses))

    def _detect(self, x_test):
        loader = self._get_dataloader(x_test, "test")
        self.backend.eval()
        scores = []
        with torch.no_grad():
            for x, target in loader:
                x, target = x.to(self.device), target.to(self.device)
                scores.append((self.backend(x) - target).abs().squeeze(-1).cpu().numpy())
        return DetectResult(scores=np.concatenate(scores), start_pos=self.params["win_len"])
