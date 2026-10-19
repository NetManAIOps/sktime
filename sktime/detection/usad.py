from __future__ import annotations


import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping


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


class USAD(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        batch_size=128,
        win_len=5,
        hidden_dim=16,
        latent_dim=5,
        epochs=10,
        lr=0.0001,
        weight_decay=1e-05,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.win_len = win_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.criterion_ = nn.MSELoss(reduction="none")
        self.backend_ = USADBackbone(
            feats=1,
            n_window=int(self.win_len),
            n_hidden=int(self.hidden_dim),
            n_latent=int(self.latent_dim),
        )

        return self.backend_

    def _get_dataloader(self, x, flag="train"):
        dataset = ReconDataset(x, win_len=int(self.win_len))
        return DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=flag == "train",
            drop_last=False,
        )

    def _epoch_losses(self, x: torch.Tensor, n: int):
        ae1, ae2, ae2ae1 = self.backend_(x)
        target = x.view(ae2ae1.shape[0], -1)
        l1 = (1.0 / n) * self.criterion_(ae1, target) + (1.0 - 1.0 / n) * self.criterion_(ae2ae1, target)
        l2 = (1.0 / n) * self.criterion_(ae2, target) - (1.0 - 1.0 / n) * self.criterion_(ae2ae1, target)
        return l1, l2, ae1, ae2ae1, target

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        self._set_seed()
        train_loader = self._get_dataloader(X, flag="train")
        epochs = int(self.epochs)
        lr = float(self.lr)
        weight_decay = float(self.weight_decay)

        optimizer1 = torch.optim.Adam(
            list(self.backend_.encoder.parameters()) + list(self.backend_.decoder1.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        optimizer2 = torch.optim.Adam(
            list(self.backend_.encoder.parameters()) + list(self.backend_.decoder2.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 5, 0.9)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 5, 0.9)

        early_stopping = None
        if self.early_stop_patience > 0 and X_val is not None:
            early_stopping = EarlyStopping(
                inclusive=True,
                mode="min",
                patience=int(self.early_stop_patience),
                delta=float(self.early_stop_delta),
            )

        for epoch in range(epochs):
            n = epoch + 1
            self.backend_.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x in train_loader:
                x = x.to(self.device_)
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
                val_loss = self._valid_loss(X_val)
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

    def _valid_loss(self, x_val: np.ndarray):
        val_loader = self._get_dataloader(x_val, flag="val")
        losses = []
        self.backend_.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device_)
                ae1, _, ae2ae1 = self.backend_(x)
                target = x.view(ae2ae1.shape[0], -1)
                score = 0.1 * self.criterion_(ae1, target) + 0.9 * self.criterion_(ae2ae1, target)
                losses.append(torch.mean(score).item())
        return float(np.mean(losses)) if losses else float("inf")

    def _predict_windows(self, x_test: np.ndarray):
        loader = self._get_dataloader(x_test, flag="val")
        recon_last_all, score_all = [], []

        self.backend_.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device_)
                ae1, _, ae2ae1 = self.backend_(x)
                target = x.view(ae2ae1.shape[0], -1)

                score = 0.1 * self.criterion_(ae1, target) + 0.9 * self.criterion_(ae2ae1, target)
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
        ae1, _, ae2ae1 = self.backend_(x)
        score = 0.1 * self.criterion_(ae1, target) + 0.9 * self.criterion_(ae2ae1, target)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def _transform_scores(self, x_test: np.ndarray, **kwargs):
        output, scores = self._predict_windows(x_test)
        return np.pad(
            np.asarray(scores, dtype=float),
            (int(self.win_len) - 1, 0), constant_values=0,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
