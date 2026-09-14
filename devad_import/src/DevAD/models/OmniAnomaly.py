from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .Base import BaseModel, DetectResult, REQUIRED
from ..utils.dataset import ReconDataset
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class PlanarFlow(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.w = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor):
        wu = torch.dot(self.w, self.u)
        u_hat = self.u + (-1 + F.softplus(wu) - wu) * self.w / self.w.square().sum()
        linear = torch.matmul(z, self.w) + self.b
        h = torch.tanh(linear)
        z_next = z + u_hat * h.unsqueeze(-1)
        psi = (1 - h.pow(2)).unsqueeze(-1) * self.w
        det_jac = 1 + torch.matmul(psi, u_hat)
        log_abs_det = torch.log(torch.abs(det_jac))
        return z_next, log_abs_det


class RecurrentNetwork(nn.Module):
    """TF1 GRUCell (reset-before) followed by two linear dense layers.

    torch.nn.GRU uses reset-after, so it is not an interchangeable backend.
    The upstream dense layers have no activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dense_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, dense_dim),
            nn.Linear(dense_dim, dense_dim),
        )
        for layer in (self.gates, self.candidate, *self.dense):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.gates.bias)

    def forward(self, x: torch.Tensor):
        state = x.new_zeros(x.shape[0], self.hidden_dim)
        outputs = []
        for value in x.unbind(dim=1):
            reset, update = torch.sigmoid(self.gates(torch.cat((value, state), dim=-1))).chunk(2, dim=-1)
            candidate = torch.tanh(self.candidate(torch.cat((value, reset * state), dim=-1)))
            state = update * state + (1 - update) * candidate
            outputs.append(state)
        return self.dense(torch.stack(outputs, dim=1))


class OmniAnomalyBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dense_dim: int,
                 latent_dim: int, n_flows: int, std_epsilon: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.std_epsilon = std_epsilon
        self.encoder = RecurrentNetwork(input_dim, hidden_dim, dense_dim)
        self.posterior_mean = nn.Linear(dense_dim + latent_dim, latent_dim)
        self.posterior_std = nn.Linear(dense_dim + latent_dim, latent_dim)
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])
        self.decoder = RecurrentNetwork(latent_dim, hidden_dim, dense_dim)
        self.output_mean = nn.Linear(dense_dim, input_dim)
        self.output_std = nn.Linear(dense_dim, input_dim)
        for head in (self.posterior_mean, self.posterior_std, self.output_mean, self.output_std):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def sample_posterior(self, x: torch.Tensor, n_samples: int):
        # Both upstream tf.scan calls use back_prop=False. Keep the encoder and
        # posterior detached, including the separately evaluated log density.
        with torch.no_grad():
            context = self.encoder(x)
            batch_size, win_len, _ = context.shape
            noise = x.new_empty(win_len, n_samples, self.latent_dim)
            nn.init.trunc_normal_(noise, a=-2., b=2.)
            previous = x.new_zeros(n_samples, batch_size, self.latent_dim)
            samples = []
            for t in range(win_len):
                h = context[:, t].unsqueeze(0).expand(n_samples, -1, -1)
                inputs = torch.cat((h, previous), dim=-1)
                mean = self.posterior_mean(inputs)
                std = F.softplus(self.posterior_std(inputs)) + self.std_epsilon
                # Upstream shares the same draw across batch members.
                previous = mean + std * noise[t].unsqueeze(1)
                samples.append(previous)
            z = torch.stack(samples, dim=2)  # [samples, batch, time, latent]
            # Preserve the upstream log_prob_step ordering [z_current, context],
            # which differs from sample_step's [context, z_previous].
            inputs = torch.cat((z, context.unsqueeze(0).expand(n_samples, -1, -1, -1)), dim=-1)
            mean = self.posterior_mean(inputs)
            std = F.softplus(self.posterior_std(inputs)) + self.std_epsilon
            log_q = (-0.5 * np.log(2 * np.pi) - std.log()
                     - 0.5 * ((z - mean).abs().clamp_max(1e8) / std).square()).sum(dim=-1)

        for flow in self.flows:
            z, log_det = flow(z)
            log_q = log_q - log_det
        return z, log_q

    def forward(self, x: torch.Tensor, n_samples: int = 1):
        z, log_q = self.sample_posterior(x, n_samples)
        # Official wrapper.rnn averages latent samples BEFORE decoding.
        hidden = self.decoder(z.mean(dim=0))
        mean = self.output_mean(hidden)
        std = F.softplus(self.output_std(hidden)) + self.std_epsilon
        return mean, std, log_q


class OmniAnomaly(BaseModel):
    HP = {
        "batch_size": 50,
        "mc_samples": 1,
        "hidden_dim": 500,
        "dense_dim": 500,
        "latent_dim": 3,
        "n_flows": 20,
        "std_epsilon": 1e-4,
        "win_len": REQUIRED,
        "scale_mode": "zscore",
        "lr": 1e-3,
        "epochs": 10,
        "scheduler_step_size": 40,
        "scheduler_gamma": 0.5,
        "grad_clip": 10.0,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.mc_samples = int(self.params["mc_samples"])
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        self.backend = OmniAnomalyBackbone(
            input_dim=1,
            hidden_dim=int(self.params["hidden_dim"]),
            dense_dim=int(self.params["dense_dim"]),
            latent_dim=int(self.params["latent_dim"]),
            n_flows=int(self.params["n_flows"]),
            std_epsilon=float(self.params["std_epsilon"]),
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

    def _loss(self, x: torch.Tensor):
        mean, std, log_q = self.backend(x)
        nll = -torch.distributions.Normal(mean, std).log_prob(x).sum(dim=-1)
        return (nll.unsqueeze(0) + log_q).mean()

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
        optimizer = Adam(self.backend.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.params["scheduler_step_size"]),
            gamma=float(self.params["scheduler_gamma"]),
        )

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
                loss = self._loss(x)
                optimizer.zero_grad()
                loss.backward()
                # Official trainer clips each variable separately, not globally.
                if self.params["grad_clip"] > 0:
                    for parameter in self.backend.parameters():
                        if parameter.grad is not None:
                            nn.utils.clip_grad_norm_([parameter], float(self.params["grad_clip"]))
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

    def _valid_loss(self, x_val: np.ndarray):
        loader = self._get_dataloader(x_val, flag="val")
        losses = []
        self.backend.eval()
        with torch.no_grad():
            for x in loader:
                losses.append(self._loss(x.to(self.device).unsqueeze(-1)).item())
        return float(np.mean(losses)) if losses else float("inf")

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        loader = self._get_dataloader(x_test, flag="val")
        recon_all, score_all = [], []
        self.backend.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device).unsqueeze(-1)
                mean, std, _ = self.backend(x, n_samples=self.mc_samples)
                point_nll = -torch.distributions.Normal(mean, std).log_prob(x).sum(dim=-1)
                recon_all.append(mean[:, -1, 0].cpu().numpy())
                score_all.append(point_nll[:, -1].cpu().numpy())
        recon = np.concatenate(recon_all)
        scores = np.concatenate(score_all)
        return DetectResult(scores=scores, output=recon, start_pos=int(self.params["win_len"]) - 1)
