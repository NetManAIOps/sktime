from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping


class BeatGANGenerator(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x.transpose(1, 2).contiguous())
        recon = self.decoder(z).transpose(1, 2).contiguous()
        return recon, z


class BeatGANDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor):
        feat = self.features(x.transpose(1, 2).contiguous()).flatten(1)
        logit = self.classifier(feat)
        return logit, feat


class BeatGANBackbone(nn.Module):
    """Adversarially regularized convolutional reconstruction model."""

    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.generator = BeatGANGenerator(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.discriminator = BeatGANDiscriminator(hidden_dim=hidden_dim)


class BeatGAN(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        batch_size=128,
        hidden_dim=32,
        latent_dim=16,
        win_len=64,
        lr=0.0001,
        epochs=30,
        recon_weight=10.0,
        feature_weight=1.0,
        adv_weight=1.0,
        score_recon_weight=0.9,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.win_len = win_len
        self.lr = lr
        self.epochs = epochs
        self.recon_weight = recon_weight
        self.feature_weight = feature_weight
        self.adv_weight = adv_weight
        self.score_recon_weight = score_recon_weight
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.backend_ = BeatGANBackbone(
            hidden_dim=int(self.hidden_dim),
            latent_dim=int(self.latent_dim),
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
        lr = float(self.lr)
        epochs = int(self.epochs)
        recon_weight = float(self.recon_weight)
        feature_weight = float(self.feature_weight)
        adv_weight = float(self.adv_weight)

        opt_g = Adam(self.backend_.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = Adam(self.backend_.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        early_stopping = None
        if self.early_stop_patience > 0 and X_val is not None:
            early_stopping = EarlyStopping(
                inclusive=True,
                mode="min",
                patience=int(self.early_stop_patience),
                delta=float(self.early_stop_delta),
            )

        for epoch in range(epochs):
            self.backend_.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x in train_loader:
                x = x.to(self.device_)
                real_target = torch.ones((x.shape[0], 1), dtype=x.dtype, device=self.device_)
                fake_target = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=self.device_)

                recon, _ = self.backend_.generator(x)
                real_logit, _ = self.backend_.discriminator(x)
                fake_logit, _ = self.backend_.discriminator(recon.detach())
                d_loss = bce(real_logit, real_target) + bce(fake_logit, fake_target)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                recon, _ = self.backend_.generator(x)
                fake_logit, fake_feat = self.backend_.discriminator(recon)
                _, real_feat = self.backend_.discriminator(x)
                recon_loss = F.mse_loss(recon, x)
                feature_loss = F.mse_loss(fake_feat, real_feat.detach())
                adv_loss = bce(fake_logit, real_target)
                g_loss = recon_weight * recon_loss + feature_weight * feature_loss + adv_weight * adv_loss
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                reporter.step(g_loss.item() + d_loss.item())

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
        loader = self._get_dataloader(x_val, flag="val")
        losses = []
        self.backend_.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device_)
                recon, _ = self.backend_.generator(x)
                losses.append(F.mse_loss(recon, x).cpu().item())
        return float(np.mean(losses)) if losses else float("inf")

    def _predict_energy(self, x_test: np.ndarray):
        loader = self._get_dataloader(x_test, flag="val")
        recon_all, score_all = [], []
        alpha = float(self.score_recon_weight)
        self.backend_.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device_)
                recon, _ = self.backend_.generator(x)
                _, real_feat = self.backend_.discriminator(x)
                _, fake_feat = self.backend_.discriminator(recon)
                point_errors = (x - recon).pow(2).squeeze(-1)
                feature_error = torch.mean((real_feat - fake_feat).pow(2), dim=1)
                score = alpha * point_errors[:, -1] + (1.0 - alpha) * feature_error
                recon_all.append(recon[:, -1, 0].cpu().numpy())
                score_all.append(score.cpu().numpy())
        recon = np.concatenate(recon_all)
        scores = np.concatenate(score_all)
        return recon, scores

    def _score_window(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        recon, _ = self.backend_.generator(x)
        score = (x - recon).pow(2).squeeze(-1)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def _transform_scores(self, x_test: np.ndarray, **kwargs):
        output, scores = self._predict_energy(x_test)
        return np.pad(
            np.asarray(scores, dtype=float),
            (int(self.win_len) - 1, 0), constant_values=0,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
