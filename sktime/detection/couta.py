"""COUTA: Calibrated One-class classifier for Unsupervised TS Anomaly detection.

Refactored from Hongzuo Xu's Apache-2.0 implementation:
https://github.com/xuhongzuo/couta/tree/fe728a3625db7eb7269ff5ca3218da39a242533b
See COUTA.LICENSE for the upstream license.

Network, UMC/NAC, native anomalies, center initialization and scores follow
upstream. DevAD uses ReconDataset (stride 1, Min-Max, clipping to [-4, 5]),
including its constant-series handling and float64 scaling intermediates.
External validation UMC controls early stopping (no internal split); scores
are returned as the valid suffix instead of prepending zeros.
"""

from __future__ import annotations


import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping


def create_batch_neg(batch_seqs: torch.Tensor, max_cut_ratio: float = 0.5,
                     seed: int = 0, return_mul_label: bool = False):
    """Author FULL mode: tail 0/1, last-point local mean +/-0.5, or +/-2."""
    rng = np.random.RandomState(seed)
    batch_size, length, dim = batch_seqs.shape
    cut_start = length - rng.randint(1, int(max_cut_ratio * length), size=batch_size)
    n_cut_dim = rng.randint(1, dim + 1, size=batch_size)
    cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]
    negative = batch_seqs.clone()
    labels = torch.empty(batch_size, dtype=torch.long)
    flags = rng.randint(100000, size=batch_size)
    for i, flag in enumerate(flags % 6):
        if flag == 0 or flag == 1:
            negative[i, cut_start[i]:, cut_dim[i]] = int(flag)
            labels[i] = 1
        elif flag == 2 or flag == 3:
            mean = torch.mean(negative[i, -10:, cut_dim[i]], dim=0)
            offset = 0.5 if flag == 2 else -0.5
            negative[i, -1, cut_dim[i]] = mean + offset
            labels[i] = 2
        else:
            negative[i, -1, cut_dim[i]] = 2 if flag == 4 else -2
            labels[i] = 3
    return negative, labels if return_mul_label else torch.ones(batch_size, dtype=torch.long)


class DSVDDUncLoss(nn.Module):
    def __init__(self, c: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, rep_dup):
        distance = torch.sum((rep - self.c) ** 2, dim=1)
        distance_dup = torch.sum((rep_dup - self.c) ** 2, dim=1)
        variance = (distance - distance_dup) ** 2
        loss = 0.5 * torch.exp(-variance) * (distance + distance_dup) + 0.5 * variance
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding,
                 bias=True, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, padding=padding, bias=bias, dilation=dilation,
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=padding, bias=bias, dilation=dilation,
        ))
        self.net = nn.Sequential(
            self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # Preserve the author's initialization and RNG draw order.
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.net(x) + residual


class COUTABackbone(nn.Module):
    def __init__(self, input_dim, hidden_dims=16, rep_hidden=16, pretext_hidden=16,
                 emb_dim=16, kernel_size=2, dropout=0.0, tcn_bias=True, linear_bias=True):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers = []
        for i, out_channels in enumerate(hidden_dims):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dims[i - 1]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dilation,
                padding=(kernel_size - 1) * dilation, dropout=dropout, bias=tcn_bias,
            ))
        self.network = nn.Sequential(*layers)
        self.l1 = nn.Linear(hidden_dims[-1], rep_hidden, bias=linear_bias)
        self.l2 = nn.Linear(rep_hidden, emb_dim, bias=linear_bias)
        self.act = nn.LeakyReLU()
        self.l1_dup = nn.Linear(hidden_dims[-1], rep_hidden, bias=linear_bias)
        self.pretext_l1 = nn.Linear(hidden_dims[-1], pretext_hidden, bias=linear_bias)
        self.pretext_l2 = nn.Linear(pretext_hidden, 1, bias=linear_bias)

    def forward(self, x):
        hidden = self.network(x.transpose(2, 1)).transpose(2, 1)[:, -1]
        rep = self.l2(self.act(self.l1(hidden)))
        score = self.pretext_l2(self.act(self.pretext_l1(hidden)))
        rep_dup = self.l2(self.act(self.l1_dup(hidden)))
        return rep, rep_dup, score


class COUTA(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        win_len=100,
        epochs=40,
        batch_size=64,
        lr=0.0001,
        hidden_dims=16,
        emb_dim=16,
        rep_hidden=16,
        pretext_hidden=16,
        kernel_size=2,
        dropout=0.0,
        bias=True,
        alpha=0.1,
        neg_batch_ratio=0.2,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.win_len = win_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.emb_dim = emb_dim
        self.rep_hidden = rep_hidden
        self.pretext_hidden = pretext_hidden
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.bias = bias
        self.alpha = alpha
        self.neg_batch_ratio = neg_batch_ratio
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.backend_ = COUTABackbone(
            input_dim=1,
            hidden_dims=self.hidden_dims,
            emb_dim=self.emb_dim,
            pretext_hidden=self.pretext_hidden,
            rep_hidden=self.rep_hidden,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            linear_bias=self.bias,
            tcn_bias=self.bias,
        )
        self.c_: torch.Tensor | None = None

        return self.backend_

    def _get_dataloader(self, x, flag="train"):
        dataset = ReconDataset(x, win_len=int(self.win_len))
        return DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=flag == "train",
            drop_last=flag == "train",
        )

    def _set_c(self, loader: DataLoader) -> None:
        """Fix the hypersphere center from initial training representations."""
        representations = []
        self.backend_.eval()
        with torch.no_grad():
            for batch in loader:
                rep = self.backend_(batch.to(self.device_))[0]
                representations.append(rep)
        c = torch.cat(representations).mean(dim=0)
        c[(abs(c) < 0.1) & (c < 0)] = -0.1
        c[(abs(c) < 0.1) & (c > 0)] = 0.1
        self.c_ = c

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        self._set_seed()
        train_loader = self._get_dataloader(X, flag="train")
        if len(train_loader) == 0:
            raise ValueError("COUTA requires at least one full training batch of windows")


        self._set_c(train_loader)
        optimizer = torch.optim.Adam(self.backend_.parameters(), lr=self.lr)
        criterion_umc = DSVDDUncLoss(self.c_)
        criterion_nac = nn.MSELoss()
        normal_targets = -torch.ones(self.batch_size, device=self.device_)
        neg_batch_size = int(self.neg_batch_ratio * self.batch_size)
        early_stopping = None
        if self.early_stop_patience > 0 and X_val is not None:
            early_stopping = EarlyStopping(
                inclusive=True,
                mode="min",
                patience=int(self.early_stop_patience),
                delta=float(self.early_stop_delta),
            )

        epochs = int(self.epochs)
        for epoch in range(epochs):
            self.backend_.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            seeds = np.random.RandomState(self.seed + epoch).randint(0, 1000000, len(train_loader))
            for batch, batch_seed in zip(train_loader, seeds):
                batch = batch.to(self.device_)
                rep, rep_dup, normal_pred = self.backend_(batch)
                loss_umc = criterion_umc(rep, rep_dup)
                indices = np.random.RandomState(batch_seed).randint(
                    0, self.batch_size, neg_batch_size,
                )
                negative, labels = create_batch_neg(batch[indices], seed=batch_seed)
                negative_pred = self.backend_(negative)[-1]
                targets = torch.hstack([normal_targets, labels.to(self.device_)])
                predictions = torch.cat([normal_pred, negative_pred]).view(-1)
                loss_nac = criterion_nac(predictions, targets)
                loss = loss_umc + self.alpha * loss_nac

                self.backend_.zero_grad()
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())

            val_loss = None
            if X_val is not None:
                val_loss = self._valid_loss(X_val)
            if early_stopping is not None:
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

    def _valid_loss(self, x_val):
        val_loader = self._get_dataloader(x=x_val, flag="val")
        criterion = DSVDDUncLoss(self.c_, reduction="sum")
        total = 0.0
        self.backend_.eval()
        with torch.no_grad():
            for batch in val_loader:
                rep, rep_dup, _ = self.backend_(batch.to(self.device_))
                total += criterion(rep, rep_dup).item()
        return total / len(val_loader.dataset)

    def _transform_scores(self, x_test: np.ndarray, **kwargs):
        win_len = self.win_len
        if len(x_test) < win_len:
            raise ValueError(f"COUTA requires at least win_len={win_len} detection points")
        loader = self._get_dataloader(x_test, flag="test")
        representations, duplicates = [], []
        self.backend_.eval()
        with torch.no_grad():
            for batch in loader:
                rep, rep_dup, _ = self.backend_(batch.to(self.device_))
                representations.append(rep)
                duplicates.append(rep_dup)
        # Preserve upstream predict's arithmetic and CPU conversion order.
        scores = torch.sum((torch.cat(representations) - self.c_) ** 2, dim=1).cpu().numpy()
        scores_dup = torch.sum((torch.cat(duplicates) - self.c_) ** 2, dim=1).cpu().numpy()
        return np.pad(
            np.asarray(scores + scores_dup, dtype=float),
            (win_len - 1, 0), constant_values=0,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
