from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping


def _mlp(dims: Sequence[int], activation: type[nn.Module] = nn.Tanh, last_activation: nn.Module | None = None):
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != dims[-1]:
            layers.append(activation())
    if last_activation is not None:
        layers.append(last_activation)
    return nn.Sequential(*layers)


class DAGMMBackbone(nn.Module):
    """DAGMM-style autoencoder + estimation network.

    Based on Zong et al., "Deep Autoencoding Gaussian Mixture Model for
    Unsupervised Anomaly Detection" (ICLR 2018). This implementation adapts
    the method to the repository's univariate sliding-window TSAD protocol.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        ae_hidden: Sequence[int],
        est_hidden: Sequence[int],
        n_gmm: int,
    ):
        super().__init__()
        self.encoder = _mlp([input_dim, *ae_hidden, latent_dim])
        self.decoder = _mlp([latent_dim, *reversed(ae_hidden), input_dim])
        self.estimation = _mlp([latent_dim + 2, *est_hidden, n_gmm], activation=nn.Tanh, last_activation=nn.Softmax(dim=1))
        zc_dim = latent_dim + 2
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, zc_dim))
        self.register_buffer("cov", torch.eye(zc_dim).unsqueeze(0).repeat(n_gmm, 1, 1))
        self.register_buffer("gmm_fitted", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def compression_features(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor):
        rec_euclidean = torch.norm(x - x_hat, p=2, dim=1) / (torch.norm(x, p=2, dim=1) + 1e-8)
        cos_sim = nn.functional.cosine_similarity(x, x_hat, dim=1)
        return torch.cat([z, rec_euclidean.unsqueeze(1), cos_sim.unsqueeze(1)], dim=1)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        zc = self.compression_features(x, x_hat, z)
        gamma = self.estimation(zc)
        return x_hat, z, zc, gamma

    @staticmethod
    def compute_gmm_params(zc: torch.Tensor, gamma: torch.Tensor):
        gamma_sum = torch.sum(gamma, dim=0) + 1e-8
        phi = gamma_sum / zc.shape[0]
        mu = torch.sum(gamma.unsqueeze(2) * zc.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(1)
        diff = zc.unsqueeze(1) - mu.unsqueeze(0)
        cov = torch.einsum("nk,nkd,nke->kde", gamma, diff, diff) / gamma_sum.view(-1, 1, 1)
        eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device).unsqueeze(0)
        cov = cov + 1e-6 * eye
        return phi, mu, cov

    @staticmethod
    def sample_energy(zc: torch.Tensor, phi: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor):
        original_dtype = zc.dtype
        calc_dtype = torch.float32 if zc.device.type == "mps" else torch.float64
        zc = zc.to(calc_dtype)
        phi = phi.to(calc_dtype)
        mu = mu.to(calc_dtype)
        cov = cov.to(calc_dtype)

        k, dim, _ = cov.shape
        eye = torch.eye(dim, dtype=zc.dtype, device=zc.device).unsqueeze(0)
        cov = cov + 1e-6 * eye
        inv_cov = torch.linalg.inv(cov)
        diff = zc.unsqueeze(1) - mu.unsqueeze(0)
        mahalanobis = torch.einsum("nkd,kde,nke->nk", diff, inv_cov, diff)
        sign, log_det = torch.linalg.slogdet(cov)
        log_det = torch.where(sign > 0, log_det, torch.full_like(log_det, np.log(1e-12)))
        log_det = torch.clamp(log_det, min=np.log(1e-12))
        log_phi = torch.log(torch.clamp(phi, min=1e-12)).unsqueeze(0)
        log_normalizer = 0.5 * (dim * np.log(2 * np.pi) + log_det).unsqueeze(0)
        log_component_prob = log_phi - 0.5 * mahalanobis - log_normalizer
        energy = -torch.logsumexp(log_component_prob, dim=1)
        energy = torch.nan_to_num(energy, nan=0.0, posinf=1e6, neginf=-1e6)
        cov_diag = torch.diagonal(cov, dim1=1, dim2=2)
        cov_penalty = torch.sum(1.0 / torch.clamp(cov_diag, min=1e-8))
        cov_penalty = torch.nan_to_num(cov_penalty, nan=0.0, posinf=1e6, neginf=-1e6)
        return energy.to(original_dtype), cov_penalty.to(original_dtype)

    def set_gmm_params(self, phi: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor):
        self.phi = phi.detach()
        self.mu = mu.detach()
        self.cov = cov.detach()
        self.gmm_fitted = torch.tensor(True, dtype=torch.bool, device=self.phi.device)


class DAGMM(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        batch_size=128,
        win_len=64,
        latent_dim=4,
        ae_hidden=(64, 32),
        est_hidden=(16,),
        n_gmm=3,
        energy_weight=0.1,
        cov_weight=0.005,
        lr=0.0001,
        epochs=30,
        weight_decay=0.0,
        grad_clip=5.0,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.win_len = win_len
        self.latent_dim = latent_dim
        self.ae_hidden = ae_hidden
        self.est_hidden = est_hidden
        self.n_gmm = n_gmm
        self.energy_weight = energy_weight
        self.cov_weight = cov_weight
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.backend_ = DAGMMBackbone(
            input_dim=int(self.win_len),
            latent_dim=int(self.latent_dim),
            ae_hidden=self.ae_hidden,
            est_hidden=self.est_hidden,
            n_gmm=int(self.n_gmm),
        )

        return self.backend_

    def _get_dataloader(self, x, flag="train"):
        dataset = ReconDataset(x, win_len=int(self.win_len))
        return DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=flag == "train",
            drop_last=False,
        )

    def _loss(self, x: torch.Tensor):
        x_hat, _, zc, gamma = self.backend_(x)
        recon_loss = torch.mean((x - x_hat) ** 2)
        phi, mu, cov = self.backend_.compute_gmm_params(zc, gamma)
        energy, cov_penalty = self.backend_.sample_energy(zc, phi, mu, cov)
        energy_weight = float(self.energy_weight)
        cov_weight = float(self.cov_weight)
        return recon_loss + energy_weight * torch.mean(energy) + cov_weight * cov_penalty

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        self._set_seed()
        train_loader = self._get_dataloader(X, flag="train")
        lr = float(self.lr)
        epochs = int(self.epochs)
        optimizer = Adam(self.backend_.parameters(), lr=lr, weight_decay=float(self.weight_decay))

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
                x = x.to(self.device_).squeeze(-1)
                loss = self._loss(x)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.backend_.parameters(), float(self.grad_clip))
                optimizer.step()
                reporter.step(loss.item())
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

        if hasattr(self, "_best_state"):
            self._restore_best_weights()
        reporter.begin_stage("Fitting GMM")
        self._fit_gmm_params(X)

        if self.best_epoch_ is None:
            self.best_epoch_ = int(self.epochs)

    def _valid_loss(self, x_val: np.ndarray):
        loader = self._get_dataloader(x_val, flag="val")
        losses = []
        self.backend_.eval()
        with torch.no_grad():
            for x in loader:
                losses.append(self._loss(x.to(self.device_).squeeze(-1)).item())
        return float(np.mean(losses)) if losses else float("inf")

    def _fit_gmm_params(self, x_train: np.ndarray):
        loader = self._get_dataloader(x_train, flag="val")
        zc_all, gamma_all = [], []
        self.backend_.eval()
        with torch.no_grad():
            for x in loader:
                _, _, zc, gamma = self.backend_(x.to(self.device_).squeeze(-1))
                zc_all.append(zc)
                gamma_all.append(gamma)
        zc = torch.cat(zc_all, dim=0)
        gamma = torch.cat(gamma_all, dim=0)
        self.backend_.set_gmm_params(*self.backend_.compute_gmm_params(zc, gamma))

    def _predict_energy(self, x_test: np.ndarray):
        loader = self._get_dataloader(x_test, flag="val")
        recon_all, score_all = [], []
        self.backend_.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device_).squeeze(-1)
                x_hat, _, zc, _ = self.backend_(x)
                if not bool(self.backend_.gmm_fitted.item()):
                    phi, mu, cov = self.backend_.compute_gmm_params(zc, self.backend_.estimation(zc))
                else:
                    phi, mu, cov = self.backend_.phi, self.backend_.mu, self.backend_.cov
                energy, _ = self.backend_.sample_energy(zc, phi, mu, cov)
                recon_all.append(x_hat[:, -1].cpu().numpy())
                score_all.append(energy.cpu().numpy())
        recon = np.concatenate(recon_all)
        scores = np.nan_to_num(np.concatenate(score_all), nan=0.0, posinf=1e6, neginf=-1e6)
        return recon, scores

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
