from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .Base import BaseModel, DetectResult
from ..utils.dataset import ReconDataset
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


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


class BeatGAN(BaseModel):
    HP = {
        "batch_size": 128,
        "hidden_dim": 32,
        "latent_dim": 16,
        "win_len": 64,
        "scale_mode": "zscore",
        "lr": 1e-4,
        "epochs": 30,
        "recon_weight": 10.0,
        "feature_weight": 1.0,
        "adv_weight": 1.0,
        "score_recon_weight": 0.9,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.backend = BeatGANBackbone(
            hidden_dim=int(self.params["hidden_dim"]),
            latent_dim=int(self.params["latent_dim"]),
        ).to(self.device)

    def _get_dataloader(self, x: np.ndarray, flag: str = "train"):
        win_len = int(self.params["win_len"])
        scale_mode = self.params["scale_mode"]
        assert flag in ["train", "val"]
        values = x

        if flag == "train":
            dataset = ReconDataset(values, win_len=win_len, flag="train", scale_cfg=self.scale_cfg, scale_mode=scale_mode)
            self.scale_cfg = dataset.get_scale_cfg()
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        if flag == "val":
            dataset = ReconDataset(values, win_len=win_len, flag="val", scale_cfg=self.scale_cfg, scale_mode=scale_mode)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _fit(
        self,
        x_train: np.ndarray,
        y: np.ndarray | None = None,
        x_val: np.ndarray | None = None,
        checkpoint_path=None,
        reporter: TrainingReporter | None = None,
    ):
        reporter = reporter or NullTrainingReporter()
        train_loader = self._get_dataloader(x_train, flag="train")
        lr = float(self.params["lr"])
        epochs = int(self.params["epochs"])
        recon_weight = float(self.params["recon_weight"])
        feature_weight = float(self.params["feature_weight"])
        adv_weight = float(self.params["adv_weight"])

        opt_g = Adam(self.backend.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = Adam(self.backend.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

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
            self.backend.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x in train_loader:
                x = x.to(self.device).unsqueeze(-1)
                real_target = torch.ones((x.shape[0], 1), dtype=x.dtype, device=self.device)
                fake_target = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=self.device)

                recon, _ = self.backend.generator(x)
                real_logit, _ = self.backend.discriminator(x)
                fake_logit, _ = self.backend.discriminator(recon.detach())
                d_loss = bce(real_logit, real_target) + bce(fake_logit, fake_target)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                recon, _ = self.backend.generator(x)
                fake_logit, fake_feat = self.backend.discriminator(recon)
                _, real_feat = self.backend.discriminator(x)
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
        loader = self._get_dataloader(x_val, flag="val")
        losses = []
        self.backend.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device).unsqueeze(-1)
                recon, _ = self.backend.generator(x)
                losses.append(F.mse_loss(recon, x).cpu().item())
        return float(np.mean(losses)) if losses else float("inf")

    def _predict(self, x_test: np.ndarray):
        loader = self._get_dataloader(x_test, flag="val")
        recon_all, score_all = [], []
        alpha = float(self.params["score_recon_weight"])
        self.backend.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device).unsqueeze(-1)
                recon, _ = self.backend.generator(x)
                _, real_feat = self.backend.discriminator(x)
                _, fake_feat = self.backend.discriminator(recon)
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
        recon, _ = self.backend.generator(x)
        score = (x - recon).pow(2).squeeze(-1)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        output, scores = self._predict(x_test)
        return DetectResult(
            scores=scores,
            output=output,
            start_pos=int(self.params["win_len"]) - 1,
        )
