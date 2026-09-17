from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .Base import BaseModel, DetectResult
from ..utils.dataset import ReconDataset
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class USADBackbone(nn.Module):
    def __init__(self, feats: int, n_window: int, n_hidden: int, n_latent: int):
        super().__init__()
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_latent),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, self.n),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, self.n),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        z = self.encoder(x.view(batch_size, self.n))
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(batch_size, self.n), ae2.view(batch_size, self.n), ae2ae1.view(batch_size, self.n)


class USAD(BaseModel):
    HP = {
        "batch_size": 128,
        "win_len": 5,
        "hidden_dim": 16,
        "latent_dim": 5,
        "scale_mode": "minmax",
        "epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.criterion = nn.MSELoss(reduction="none")
        self.backend = USADBackbone(
            feats=1,
            n_window=int(self.params["win_len"]),
            n_hidden=int(self.params["hidden_dim"]),
            n_latent=int(self.params["latent_dim"]),
        ).to(self.device)

    def _get_dataloader(self, x: np.ndarray, flag: str = "train"):
        win_len = int(self.params["win_len"])
        scale_mode = self.params["scale_mode"]
        assert flag in ["train", "val"]

        values = x
        if flag == "train":
            dataset = ReconDataset(values, labels=None, win_len=win_len, flag="train", scale_cfg=self.scale_cfg, scale_mode=scale_mode)
            self.scale_cfg = dataset.get_scale_cfg()
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        if flag == "val":
            dataset = ReconDataset(values, labels=None, win_len=win_len, flag="val", scale_cfg=self.scale_cfg, scale_mode=scale_mode, clip=False)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _epoch_losses(self, x: torch.Tensor, n: int):
        ae1, ae2, ae2ae1 = self.backend(x)
        target = x.view(ae2ae1.shape[0], -1)
        l1 = (1.0 / n) * self.criterion(ae1, target) + (1.0 - 1.0 / n) * self.criterion(ae2ae1, target)
        l2 = (1.0 / n) * self.criterion(ae2, target) - (1.0 - 1.0 / n) * self.criterion(ae2ae1, target)
        return l1, l2, ae1, ae2ae1, target

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
        epochs = int(self.params["epochs"])
        lr = float(self.params["lr"])
        weight_decay = float(self.params["weight_decay"])

        optimizer1 = torch.optim.Adam(
            list(self.backend.encoder.parameters()) + list(self.backend.decoder1.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        optimizer2 = torch.optim.Adam(
            list(self.backend.encoder.parameters()) + list(self.backend.decoder2.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 5, 0.9)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 5, 0.9)

        early_stopping = None
        if x_val is not None:
            x_val = self.check_array(x_val, dtype=np.float32, name="x_val")
        if self.params["early_stop_patience"] > 0 and checkpoint_path and x_val is not None:
            early_stopping = EarlyStopping(
                mode="min",
                patience=int(self.params["early_stop_patience"]),
                delta=float(self.params["early_stop_delta"]),
            )

        for epoch in range(epochs):
            n = epoch + 1
            self.backend.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x in train_loader:
                x = x.to(self.device).unsqueeze(-1)
                optimizer1.zero_grad()
                l1, _, _, _, _ = self._epoch_losses(x, n=n)
                loss1 = torch.mean(l1)
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                _, l2, _, _, _ = self._epoch_losses(x, n=n)
                loss2 = torch.mean(l2)
                loss2.backward()
                optimizer2.step()

                reporter.step(loss1.item() + loss2.item())

            scheduler1.step()
            scheduler2.step()

            val_loss = None
            if early_stopping is not None:
                val_loss = self._valid_loss(x_val)
                early_stopping(val_loss, self, checkpoint_path, epoch=epoch + 1)
            stopped_early = early_stopping is not None and early_stopping.early_stop
            reporter.end_epoch(
                val_loss=val_loss,
                best_epoch=self.best_epoch,
                early_stopping=early_stopping,
            )
            if stopped_early:
                break

        if early_stopping is not None and checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    def _valid_loss(self, x_val: np.ndarray):
        val_loader = self._get_dataloader(x_val, flag="val")
        losses = []
        self.backend.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device).unsqueeze(-1)
                ae1, _, ae2ae1 = self.backend(x)
                target = x.view(ae2ae1.shape[0], -1)
                score = 0.1 * self.criterion(ae1, target) + 0.9 * self.criterion(ae2ae1, target)
                losses.append(torch.mean(score).item())
        return float(np.mean(losses)) if losses else float("inf")

    def _predict_windows(self, x_test: np.ndarray):
        loader = self._get_dataloader(x_test, flag="val")
        recon_last_all, score_all = [], []

        self.backend.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device).unsqueeze(-1)
                ae1, _, ae2ae1 = self.backend(x)
                target = x.view(ae2ae1.shape[0], -1)

                score = 0.1 * self.criterion(ae1, target) + 0.9 * self.criterion(ae2ae1, target)
                score = torch.mean(score, dim=-1)
                recon = 0.1 * ae1 + 0.9 * ae2ae1

                recon_last_all.append(recon[:, -1].cpu().numpy())
                score_all.append(score.cpu().numpy())

        recon_last = np.concatenate(recon_last_all, axis=0)
        scores = np.concatenate(score_all, axis=0)
        return recon_last, scores

    def _score_window(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        batch_size = x.shape[0]
        target = x.view(batch_size, -1)
        ae1, _, ae2ae1 = self.backend(x)
        score = 0.1 * self.criterion(ae1, target) + 0.9 * self.criterion(ae2ae1, target)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        output, scores = self._predict_windows(x_test)
        return DetectResult(
            scores=scores,
            output=output,
            start_pos=int(self.params["win_len"]) - 1,
        )
