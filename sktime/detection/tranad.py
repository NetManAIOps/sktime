from __future__ import annotations

# Adapted from imperial-qore/TranAD, commit 7ffb98d0c18189cc3d9ab732b4cb0278200a0af0.
# Copyright (c) 2022, Shreshth Tuli. BSD-3-Clause; see TranAD.LICENSE.

import math

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.sin(position * div_term) + torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor):
        return self.dropout(x + self.pe[: x.size(0)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Upstream LeakyReLU(True) passes True as negative_slope, not inplace.
        self.activation = nn.LeakyReLU(negative_slope=1.0)

    def forward(self, src, **kwargs):
        attended = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(attended)
        hidden = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(hidden)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=1.0)

    def forward(self, tgt, memory, **kwargs):
        attended = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(attended)
        attended = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(attended)
        hidden = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt + self.dropout3(hidden)


class TranADBackbone(nn.Module):
    """Official two-pass TranAD network with configurable window and FFN size."""

    def __init__(
        self,
        win_len: int,
        feats: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.n_feats = feats
        self.pos_encoder = PositionalEncoding(2 * feats, dropout=dropout, max_len=win_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 1, enable_nested_tensor=False)
        decoder_layer1 = TransformerDecoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1, 1)
        decoder_layer2 = TransformerDecoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src: torch.Tensor, focus: torch.Tensor, tgt: torch.Tensor):
        src = torch.cat((src, focus), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        focus = torch.zeros_like(src)
        tgt1, memory1 = self.encode(src, focus, tgt)
        out1 = self.fcn(self.transformer_decoder1(tgt1, memory1))

        focus = (out1 - src).pow(2)
        tgt2, memory2 = self.encode(src, focus, tgt)
        out2 = self.fcn(self.transformer_decoder2(tgt2, memory2))
        return out1, out2


class TranAD(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        batch_size=128,
        win_len=10,
        d_ff=16,
        dropout_rate=0.1,
        lr=0.0001,
        epochs=5,
        weight_decay=1e-05,
        scheduler_step_size=5,
        scheduler_gamma=0.9,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.win_len = win_len
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.backend_ = TranADBackbone(
            win_len=int(self.win_len),
            feats=1,
            dim_feedforward=int(self.d_ff),
            dropout=float(self.dropout_rate),
        )

        return self.backend_

    def _get_dataloader(self, x, flag="train"):
        dataset = ReconDataset(x, win_len=int(self.win_len))
        return DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=flag == "train",
            drop_last=False,
        )

    @staticmethod
    def _to_official_inputs(x: torch.Tensor):
        src = x.permute(1, 0, 2)
        tgt = src[-1:, :, :]
        return src, tgt

    @staticmethod
    def _epoch_loss(out1: torch.Tensor, out2: torch.Tensor, target: torch.Tensor, epoch: int):
        weight = 1.0 / max(epoch, 1)
        loss1 = (out1 - target).pow(2)
        loss2 = (out2 - target).pow(2)
        return (weight * loss1 + (1.0 - weight) * loss2).mean()

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        self._set_seed()
        train_loader = self._get_dataloader(X, flag="train")
        lr = float(self.lr)
        epochs = int(self.epochs)
        optimizer = AdamW(self.backend_.parameters(), lr=lr, weight_decay=float(self.weight_decay))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(self.scheduler_step_size),
            gamma=float(self.scheduler_gamma),
        )

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
                src, target = self._to_official_inputs(x)
                out1, out2 = self.backend_(src, target)
                loss = self._epoch_loss(out1, out2, target, epoch + 1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())
            scheduler.step()
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
                src, target = self._to_official_inputs(x)
                _, out2 = self.backend_(src, target)
                losses.append(torch.mean((out2 - target).pow(2)).cpu().item())
        return float(np.mean(losses)) if losses else float("inf")

    def _transform_scores(self, x_test: np.ndarray, **kwargs):
        loader = self._get_dataloader(x_test, flag="val")
        recon_all, score_all = [], []
        self.backend_.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device_)
                src, target = self._to_official_inputs(x)
                _, out2 = self.backend_(src, target)
                point_errors = (out2 - target).pow(2).squeeze(0).squeeze(-1)
                recon_all.append(out2.squeeze(0).squeeze(-1).cpu().numpy())
                score_all.append(point_errors.cpu().numpy())
        recons = np.concatenate(recon_all)
        scores = np.concatenate(score_all)
        return np.pad(
            np.asarray(scores, dtype=float),
            (int(self.win_len) - 1, 0), constant_values=0,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
