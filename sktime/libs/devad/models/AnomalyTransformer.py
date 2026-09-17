from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .Base import BaseModel, DetectResult
from ..utils.dataset import ReconDataset
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


class TriangularCausalMask:
    def __init__(self, batch_size, seq_len, device="cpu"):
        mask_shape = [batch_size, 1, seq_len, seq_len]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        distances = torch.arange(win_size).unsqueeze(0) - torch.arange(win_size).unsqueeze(1)
        self.register_buffer("distances", distances.abs().float(), persistent=False)

    def forward(self, queries, keys, values, sigma, attn_mask):
        bsz, seq_len, n_heads, d_head = queries.shape
        scale = self.scale or 1.0 / math.sqrt(d_head)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(bsz, seq_len, device=queries.device)
            scores = scores.masked_fill(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)
        sigma = torch.sigmoid(sigma * 5.0) + 1e-5
        sigma = torch.pow(3.0, sigma) - 1.0
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, seq_len)
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-(prior ** 2) / (2 * sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        value_out = torch.einsum("bhls,bshd->blhd", series, values)
        if self.output_attention:
            return value_out.contiguous(), series, prior, sigma
        return value_out.contiguous(), None, None, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        bsz, seq_len, _ = queries.shape
        _, src_len, _ = keys.shape
        n_heads = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(bsz, seq_len, n_heads, -1)
        keys = self.key_projection(keys).view(bsz, src_len, n_heads, -1)
        values = self.value_projection(values).view(bsz, src_len, n_heads, -1)
        sigma = self.sigma_projection(x).view(bsz, seq_len, n_heads)

        out, series, prior, sigma = self.inner_attention(queries, keys, values, sigma, attn_mask)
        out = out.view(bsz, seq_len, -1)
        return self.out_projection(out), series, prior, sigma


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, prior, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, prior, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list


class AnomalyTransformerBackbone(nn.Module):
    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
    ):
        super().__init__()
        self.output_attention = output_attention
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        if self.output_attention:
            return enc_out, series, prior, sigmas
        return enc_out


def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr):
    new_lr = lr * (0.5 ** max(epoch - 1, 0))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


class _DualEarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = int(patience)
        self.verbose = verbose
        self.delta = float(delta)
        self.counter = 0
        self.best_score1 = None
        self.best_score2 = None
        self.early_stop = False

    def __call__(self, loss1, loss2, model, ckpt_path, epoch=None):
        score1 = -float(loss1)
        score2 = -float(loss2)
        if self.best_score1 is None:
            self.best_score1 = score1
            self.best_score2 = score2
            if epoch is not None:
                model.best_epoch = int(epoch)
            model.save(ckpt_path)
            return
        if score1 < self.best_score1 + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return
        self.best_score1 = score1
        self.best_score2 = score2
        self.counter = 0
        if epoch is not None:
            model.best_epoch = int(epoch)
        model.save(ckpt_path)


class AnomalyTransformer(BaseModel):
    HP = {
        "batch_size": 256,
        "win_len": 100,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 3,
        "d_ff": 512,
        "dropout_rate": 0.0,
        "activation": "gelu",
        "scale_mode": "zscore",
        "epochs": 10,
        "lr": 1e-4,
        "k": 3.0,
        "temperature": 50.0,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.criterion = nn.MSELoss(reduction="none")
        self.backend = AnomalyTransformerBackbone(
            win_size=int(self.params["win_len"]),
            enc_in=1,
            c_out=1,
            d_model=int(self.params["d_model"]),
            n_heads=int(self.params["n_heads"]),
            e_layers=int(self.params["e_layers"]),
            d_ff=int(self.params["d_ff"]),
            dropout=float(self.params["dropout_rate"]),
            activation=self.params["activation"],
            output_attention=True,
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
            dataset = ReconDataset(values, labels=None, win_len=win_len, flag="val", scale_cfg=self.scale_cfg, scale_mode=scale_mode)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _association_losses(self, prior, series, win_size):
        series_loss = 0.0
        prior_loss = 0.0
        for idx in range(len(prior)):
            prior_norm = prior[idx] / torch.unsqueeze(torch.sum(prior[idx], dim=-1), dim=-1).repeat(1, 1, 1, win_size)
            series_loss += (
                torch.mean(my_kl_loss(series[idx], prior_norm.detach()))
                + torch.mean(my_kl_loss(prior_norm.detach(), series[idx]))
            )
            prior_loss += (
                torch.mean(my_kl_loss(prior_norm, series[idx].detach()))
                + torch.mean(my_kl_loss(series[idx].detach(), prior_norm))
            )
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        return series_loss, prior_loss

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
        k = float(self.params["k"])
        win_size = int(self.params["win_len"])
        optimizer = torch.optim.Adam(self.backend.parameters(), lr=lr)

        early_stopping = None
        if x_val is not None:
            x_val = self.check_array(x_val, dtype=np.float32, name="x_val")
        if self.params["early_stop_patience"] > 0 and checkpoint_path and x_val is not None:
            early_stopping = _DualEarlyStopping(
                patience=int(self.params["early_stop_patience"]),
                delta=float(self.params["early_stop_delta"]),
            )

        for epoch in range(epochs):
            self.backend.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for x in train_loader:
                x = x.float().to(self.device).unsqueeze(-1)
                optimizer.zero_grad()
                output, series, prior, _ = self.backend(x)
                series_loss, prior_loss = self._association_losses(prior, series, win_size)
                rec_loss = torch.mean(self.criterion(output, x))

                loss1 = rec_loss - k * series_loss
                loss2 = rec_loss + k * prior_loss

                loss1.backward(retain_graph=True)
                loss2.backward()
                optimizer.step()
                reporter.step(loss1.item())

            adjust_learning_rate(optimizer, epoch + 1, lr)

            val_loss1 = None
            if early_stopping is not None:
                val_loss1, val_loss2 = self._valid_losses(x_val)
                early_stopping(val_loss1, val_loss2, self, checkpoint_path, epoch=epoch + 1)
            stopped_early = early_stopping is not None and early_stopping.early_stop
            reporter.end_epoch(
                val_loss=val_loss1,
                best_epoch=self.best_epoch,
                early_stopping=early_stopping,
            )
            if stopped_early:
                break

        if early_stopping is not None and checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    def _valid_losses(self, x_val: np.ndarray):
        val_loader = self._get_dataloader(x_val, flag="val")
        loss_1 = []
        loss_2 = []
        k = float(self.params["k"])
        win_size = int(self.params["win_len"])
        self.backend.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.float().to(self.device).unsqueeze(-1)
                output, series, prior, _ = self.backend(x)
                series_loss, prior_loss = self._association_losses(prior, series, win_size)
                rec_loss = torch.mean(self.criterion(output, x))
                loss_1.append((rec_loss - k * series_loss).item())
                loss_2.append((rec_loss + k * prior_loss).item())
        if not loss_1:
            return float("inf"), float("inf")
        return float(np.mean(loss_1)), float(np.mean(loss_2))

    def _predict_energy(self, x_test: np.ndarray):
        loader = self._get_dataloader(x_test, flag="val")
        temperature = float(self.params["temperature"])
        win_size = int(self.params["win_len"])
        score_all, recon_last_all = [], []

        self.backend.eval()
        with torch.no_grad():
            for x in loader:
                x = x.float().to(self.device).unsqueeze(-1)
                output, series, prior, _ = self.backend(x)
                loss = torch.mean(self.criterion(x, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for idx in range(len(prior)):
                    prior_norm = prior[idx] / torch.unsqueeze(torch.sum(prior[idx], dim=-1), dim=-1).repeat(1, 1, 1, win_size)
                    series_term = my_kl_loss(series[idx], prior_norm.detach()) * temperature
                    prior_term = my_kl_loss(prior_norm, series[idx].detach()) * temperature
                    if idx == 0:
                        series_loss = series_term
                        prior_loss = prior_term
                    else:
                        series_loss += series_term
                        prior_loss += prior_term

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss

                score_all.append(cri[:, -1].detach().cpu().numpy())
                recon_last_all.append(output[:, -1, 0].cpu().numpy())

        recon_last = np.concatenate(recon_last_all, axis=0)
        scores = np.concatenate(score_all, axis=0)
        return recon_last, scores

    def _score_window(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        temperature = float(self.params["temperature"])
        win_size = int(self.params["win_len"])
        output, series, prior, _ = self.backend(x.float())
        loss = torch.mean(self.criterion(x, output), dim=-1)

        series_loss = 0.0
        prior_loss = 0.0
        for idx in range(len(prior)):
            prior_norm = prior[idx] / torch.unsqueeze(torch.sum(prior[idx], dim=-1), dim=-1).repeat(1, 1, 1, win_size)
            series_term = my_kl_loss(series[idx], prior_norm.detach()) * temperature
            prior_term = my_kl_loss(prior_norm, series[idx].detach()) * temperature
            if idx == 0:
                series_loss = series_term
                prior_loss = prior_term
            else:
                series_loss += series_term
                prior_loss += prior_term

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        score = metric * loss
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        output, scores = self._predict_energy(x_test)
        return DetectResult(
            scores=scores,
            output=output,
            start_pos=int(self.params["win_len"]) - 1,
        )
