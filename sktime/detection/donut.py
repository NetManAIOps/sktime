"""Donut adapted to the sktime deep-detector interface."""

from typing import Tuple, Union, Sequence, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping

__all__ = ["Donut"]


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


class Donut(BaseDeepDetector):
    """Univariate Donut detector, using the sktime estimator interface.

    Parameters
    ----------
    win_len : int
        Sliding window length.
    h_dim : int or sequence of int
        Hidden layer dimensions of the encoder and decoder.
    z_dim : int
        Latent dimension.
    mask_prob : float, default=0.01
        Probability of masking a training input point.
    mc_samples : int, default=256
        Number of Monte Carlo samples used for scoring.
    batch_size : int, default=128
        Training and scoring batch size.
    epochs : int, default=50
        Maximum number of training epochs.
    lr : float, default=1e-4
        Learning rate.
    weight_decay : float, default=1e-3
        Optimizer weight decay.
    grad_clip : float, default=10.0
        Gradient norm clipping limit.
    scheduler_step_size : int, default=10
        Learning-rate scheduler step size in epochs.
    scheduler_gamma : float, default=0.75
        Learning-rate decay factor.
    early_stop_patience : int, default=7
        Early-stopping patience in epochs.
    early_stop_delta : float, default=0.0
        Minimum validation improvement for early stopping.
    seed : int, default=0
        Random seed for model initialization, training and scoring.
    device : str, default="cpu"
        PyTorch execution device.

    Notes
    -----
    Inputs must already be standardized by the caller or an external transformer.
    This detector performs no scaling and stores no scaling statistics.
    Network, loss, windowing and Monte Carlo scoring follow DevAD.
    transform_scores returns a one-dimensional array of input length. The first
    win_len - 1 positions have no window score and are filled with zeros.
    Validation enables in-memory best-weight selection when patience is positive;
    this corresponds to DevAD fit with validation and a checkpoint path.
    predict requires a threshold policy and is not implemented here.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:missing_values": False,
    }

    def __init__(
        self,
        win_len,
        h_dim,
        z_dim,
        mask_prob=0.01,
        mc_samples=256,
        batch_size=128,
        epochs=50,
        lr=1e-4,
        weight_decay=1e-3,
        grad_clip=10.0,
        scheduler_step_size=10,
        scheduler_gamma=0.75,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.win_len = win_len
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.mask_prob = mask_prob
        self.mc_samples = mc_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.loss_fn_ = MaskedVAELoss()
        return DonutModel(
            input_dim=int(self.win_len),
            hidden_dim=self.h_dim,
            latent_dim=int(self.z_dim),
            mask_prob=float(self.mask_prob),
        )

    def _get_dataloader(self, X, shuffle=False):
        dataset = ReconDataset(X, win_len=int(self.win_len))
        return DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=shuffle, drop_last=False
        )

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        # DevAD reseeds after constructing the network, before loader iteration.
        self._set_seed()
        train_loader = self._get_dataloader(X, shuffle=True)
        epochs = int(self.epochs)
        lr = float(self.lr)
        weight_decay = float(self.weight_decay)
        scheduler_step_size = int(self.scheduler_step_size)
        scheduler_gamma = float(self.scheduler_gamma)

        self.backend_.train()
        optimizer = optim.AdamW(self.backend_.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        early_stopping = None
        stopped_early = None
        if self.early_stop_patience > 0 and X_val is not None:
            early_stopping = EarlyStopping(
                mode="min",
                patience=int(self.early_stop_patience),
                delta=float(self.early_stop_delta),
                inclusive=True,
            )

        for epoch in range(epochs):
            self.backend_.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x in train_loader:
                x = x.to(self.device_)
                output = self.backend_(x)
                loss = self.loss_fn_(output, (x,))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.backend_.parameters(), float(self.grad_clip))
                optimizer.step()
                reporter.step(loss.item())

            scheduler.step()
            val_loss = None
            if early_stopping is not None:
                val_loss = self._valid_loss(X_val)
                improved = early_stopping.update(val_loss)
                if improved:
                    self._save_best_weights(epoch + 1)
                stopped_early = early_stopping.early_stop

            reporter.end_epoch(
                val_loss=val_loss,
                best_epoch=self.best_epoch_,
                early_stopping=early_stopping,
            )
            if stopped_early:
                break

        if self.best_epoch_ is None:
            self.best_epoch_ = epochs

    def _valid_loss(self, x_val):
        val_loader = self._get_dataloader(x_val)
        losses = []
        self.backend_.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device_)
                output = self.backend_(x)
                losses.append(self.loss_fn_(output, (x,)).item())
        return float(np.mean(losses))

    def _mc_reconstruct(self, x: torch.Tensor):
        batch_size, win_len, feats = x.shape
        x_flat = x.view(batch_size, -1)
        z_mu, z_std, x_dec_mean, x_dec_std = self.backend_.vae(
            x_flat,
            return_latent_sample=False,
            num_samples=int(self.mc_samples),
        )
        if x_dec_mean.dim() == 2:
            x_dec_mean = x_dec_mean.unsqueeze(0)
            x_dec_std = x_dec_std.unsqueeze(0)
        x_dec_mean = x_dec_mean.view(x_dec_mean.shape[0], batch_size, win_len, feats)
        x_dec_std = x_dec_std.view(x_dec_std.shape[0], batch_size, win_len, feats)
        return x_dec_mean, x_dec_std

    def _transform_scores(self, X, **kwargs):
        loader = self._get_dataloader(X)
        score_all = []
        self.backend_.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device_)
                x_dec_mean, x_dec_std = self._mc_reconstruct(x)
                raw_last = x[:, -1, :].unsqueeze(0)
                mean_last = x_dec_mean[:, :, -1, :]
                std_last = x_dec_std[:, :, -1, :]
                nll_last = torch.sum(
                    F.gaussian_nll_loss(mean_last, raw_last, std_last.pow(2), reduction="none"),
                    dim=(0, 2),
                ) / x_dec_mean.shape[0]
                score_all.append(nll_last.cpu().numpy())

        scores = np.asarray(np.concatenate(score_all, axis=0), dtype=float)
        return np.pad(scores, (int(self.win_len) - 1, 0), constant_values=0)

    def _predict(self, X):
        raise NotImplementedError(
            "Donut returns continuous scores via transform_scores; "
            "an anomaly threshold policy has not been defined."
        )
