from __future__ import annotations

import os
from typing import Tuple, Union, Sequence, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from .Base import BaseModel, DetectResult, REQUIRED
from ..utils.dataset import ReconDataset
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class MLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_layers: Union[int, Sequence[int]],
        output_features: int,
        activation: Callable = nn.Identity(),
        activation_after_last_layer: bool = False,
    ):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        self.activation = activation
        self.activation_after_last_layer = activation_after_last_layer
        dims = [input_features] + list(hidden_layers) + [output_features]
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(dims[:-1], dims[1:])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        if self.activation_after_last_layer:
            out = self.activation(out)
        return out


class VaeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Union[int, Sequence[int]], latent_dim: int):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, 2 * latent_dim, activation=nn.ReLU())
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(x)
        mean, std = out.tensor_split(2, dim=-1)
        std = self.softplus(std) + 1e-4
        return mean, std


def sample_normal(mu: torch.Tensor, std: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    if num_samples == 1:
        eps = torch.randn_like(mu)
        return mu + eps * std
    eps = torch.randn((num_samples,) + mu.shape, dtype=mu.dtype, device=mu.device)
    return mu.unsqueeze(0) + eps * std.unsqueeze(0)


def normal_standard_normal_kl(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2), dim=-1)


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        return_latent_sample: bool = False,
        num_samples: int = 1,
        force_sample: bool = False,
    ):
        z_mu, z_std = self.encoder(x)
        if self.training or num_samples > 1 or force_sample:
            z_sample = sample_normal(z_mu, z_std, num_samples=num_samples)
        else:
            z_sample = z_mu
        x_dec_mean, x_dec_std = self.decoder(z_sample)
        if not return_latent_sample:
            return z_mu, z_std, x_dec_mean, x_dec_std
        return z_mu, z_std, x_dec_mean, x_dec_std, z_sample


class MaskedVAELoss(nn.Module):
    def forward(self, predictions, targets) -> torch.Tensor:
        mean_z, std_z, mean_x, std_x, sample_z, mask = predictions
        actual_x, = targets

        if mask is None:
            nll_output = torch.sum(
                F.gaussian_nll_loss(mean_x, actual_x, std_x.pow(2), reduction="none"),
                dim=(1, 2),
            )
            kl_loss = normal_standard_normal_kl(mean_z, std_z)
            return (nll_output + kl_loss).mean()

        nll_output = torch.sum(
            mask * F.gaussian_nll_loss(mean_x, actual_x, std_x.pow(2), reduction="none"),
            dim=(1, 2),
        )
        beta = torch.mean(mask, dim=(1, 2))
        nll_prior = beta * 0.5 * torch.sum(sample_z * sample_z, dim=-1)
        nll_approx = torch.sum(
            F.gaussian_nll_loss(mean_z, sample_z, std_z.pow(2), reduction="none"),
            dim=-1,
        )
        return (nll_output + nll_prior - nll_approx).mean()


class DonutModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Union[int, Sequence[int]], latent_dim: int, mask_prob: float):
        super().__init__()
        self.mask_prob = float(mask_prob)
        encoder = VaeEncoder(input_dim, hidden_dim, latent_dim)
        decoder = VaeEncoder(latent_dim, hidden_dim, input_dim)
        self.vae = VAE(encoder=encoder, decoder=decoder)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        batch_size, win_len, feats = x.shape
        if self.training and self.mask_prob > 0:
            mask = torch.empty_like(x).bernoulli_(1 - self.mask_prob)
            x = x * mask
        else:
            mask = None

        x_flat = x.view(batch_size, -1)
        mean_z, std_z, mean_x, std_x, sample_z = self.vae(x_flat, return_latent_sample=True)
        mean_x = mean_x.view(batch_size, win_len, feats)
        std_x = std_x.view(batch_size, win_len, feats)
        return mean_z, std_z, mean_x, std_x, sample_z, mask


class Donut(BaseModel):
    HP = {
        "win_len": REQUIRED,
        "h_dim": REQUIRED,
        "z_dim": REQUIRED,
        "mask_prob": 0.01,
        "mc_samples": 256,
        "batch_size": 128,
        "epochs": 50,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "grad_clip": 10.0,
        "scale_mode": "zscore",
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.75,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.grad_clip = float(self.params["grad_clip"])
        self.mc_samples = int(self.params["mc_samples"])
        self.win_len = int(self.params["win_len"])
        self.loss_fn = MaskedVAELoss()
        self.backend = DonutModel(
            input_dim=self.win_len,
            hidden_dim=self.params["h_dim"],
            latent_dim=int(self.params["z_dim"]),
            mask_prob=float(self.params["mask_prob"]),
        ).to(self.device)

    def _get_dataloader(self, x: np.ndarray, flag: str = "train"):
        scale_mode = self.params["scale_mode"]
        assert flag in ["train", "val"]

        values = x
        if flag == "train":
            dataset = ReconDataset(values, labels=None, win_len=self.win_len, flag="train", scale_cfg=self.scale_cfg, scale_mode=scale_mode)
            self.scale_cfg = dataset.get_scale_cfg()
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        if flag == "val":
            dataset = ReconDataset(values, labels=None, win_len=self.win_len, flag="val", scale_cfg=self.scale_cfg, scale_mode=scale_mode)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _fit(
        self,
        x_train: np.ndarray,
        y: np.ndarray | None = None,
        x_val: np.ndarray | None = None,
        checkpoint_path=None,
        reporter: TrainingReporter | None = None,
    ) -> None:
        reporter = reporter or NullTrainingReporter()
        train_loader = self._get_dataloader(x=x_train, flag="train")
        epochs = int(self.params["epochs"])
        lr = float(self.params["lr"])
        weight_decay = float(self.params["weight_decay"])
        scheduler_step_size = int(self.params["scheduler_step_size"])
        scheduler_gamma = float(self.params["scheduler_gamma"])

        self.backend.train()
        optimizer = optim.AdamW(self.backend.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        if x_val is not None:
            x_val = self.check_array(x_val, dtype=np.float32, name='x_val')

        early_stopping = None
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
                output = self.backend(x)
                loss = self.loss_fn(output, (x,))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.backend.parameters(), self.grad_clip)
                optimizer.step()
                reporter.step(loss.item())

            scheduler.step()
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

    def _valid_loss(self, x_val):
        val_loader = self._get_dataloader(x=x_val, flag="val")
        losses = []
        self.backend.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device).unsqueeze(-1)
                output = self.backend(x)
                losses.append(self.loss_fn(output, (x,)).item())
        return float(np.mean(losses))

    def _mc_reconstruct(self, x: torch.Tensor):
        batch_size, win_len, feats = x.shape
        x_flat = x.view(batch_size, -1)
        z_mu, z_std, x_dec_mean, x_dec_std = self.backend.vae(
            x_flat,
            return_latent_sample=False,
            num_samples=self.mc_samples,
        )
        if x_dec_mean.dim() == 2:
            x_dec_mean = x_dec_mean.unsqueeze(0)
            x_dec_std = x_dec_std.unsqueeze(0)
        x_dec_mean = x_dec_mean.view(x_dec_mean.shape[0], batch_size, win_len, feats)
        x_dec_std = x_dec_std.view(x_dec_std.shape[0], batch_size, win_len, feats)
        return x_dec_mean, x_dec_std

    def _detect(self, x_test: np.ndarray):
        loader = self._get_dataloader(x=x_test, flag="val")
        raw_last_all, recon_last_all, score_all = [], [], []
        self.backend.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device).unsqueeze(-1)
                x_dec_mean, x_dec_std = self._mc_reconstruct(x)
                raw_last = x[:, -1, :].unsqueeze(0)
                mean_last = x_dec_mean[:, :, -1, :]
                std_last = x_dec_std[:, :, -1, :]
                nll_last = torch.sum(
                    F.gaussian_nll_loss(mean_last, raw_last, std_last.pow(2), reduction="none"),
                    dim=(0, 2),
                ) / x_dec_mean.shape[0]
                recon_last = mean_last.mean(dim=0).squeeze(-1)

                raw_last_all.append(x[:, -1, 0].cpu().numpy())
                recon_last_all.append(recon_last.cpu().numpy())
                score_all.append(nll_last.cpu().numpy())

        raw_last = np.concatenate(raw_last_all, axis=0)
        recon_last = np.concatenate(recon_last_all, axis=0)
        scores = np.concatenate(score_all, axis=0)

        return DetectResult(
            scores=scores,
            output=recon_last,
            start_pos=self.win_len - 1,
        )
