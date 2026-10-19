import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ReconDataset
from sktime.detection.utils.early_stopping import EarlyStopping


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        kernels = []
        for i in range(num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [kernel(x) for kernel in self.kernels]
        return torch.stack(outputs, dim=-1).mean(dim=-1)


def fft_for_period(x: torch.Tensor, top_k: int = 2):
    xf = torch.fft.rfft(x, dim=1)
    frequency = xf.abs().mean(dim=0).mean(dim=-1)
    if frequency.numel() <= 1:
        periods = torch.ones(1, dtype=torch.long, device=x.device)
        weights = xf.abs().mean(dim=-1)[:, :1]
        return periods, weights

    frequency = frequency.clone()
    frequency[0] = 0
    max_k = min(top_k, max(1, frequency.numel() - 1))
    top_list = torch.topk(frequency, k=max_k).indices
    top_list = torch.clamp(top_list, min=1)
    periods = torch.div(x.size(1), top_list, rounding_mode="floor")
    periods = torch.clamp(periods, min=1)
    weights = xf.abs().mean(dim=-1)[:, top_list]
    return periods, weights


class TimesBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, top_k: int, d_model: int, d_ff: int, num_kernels: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, total_len, channels = x.size()
        periods, period_weight = fft_for_period(x, self.top_k)

        outputs = []
        target_len = self.seq_len + self.pred_len
        for period in periods.tolist():
            if target_len % period != 0:
                padded_len = ((target_len // period) + 1) * period
                padding = torch.zeros(batch_size, padded_len - target_len, channels, device=x.device, dtype=x.dtype)
                padded = torch.cat([x, padding], dim=1)
            else:
                padded_len = target_len
                padded = x

            out = padded.reshape(batch_size, padded_len // period, period, channels)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
            outputs.append(out[:, :target_len, :])

        stacked = torch.stack(outputs, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, total_len, channels, 1)
        result = torch.sum(stacked * period_weight, dim=-1)
        return result + x


class TimesNetBackbone(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int = 0,
        d_model: int = 16,
        d_ff: int = 32,
        enc_in: int = 1,
        c_out: int = 1,
        e_layers: int = 2,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    top_k=top_k,
                    d_model=d_model,
                    d_ff=d_ff,
                    num_kernels=num_kernels,
                )
                for _ in range(e_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev

        enc_out = self.enc_embedding(normalized)
        for block in self.blocks:
            enc_out = self.layer_norm(block(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        return dec_out

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.anomaly_detection(x_enc)


class TimesNet(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        batch_size=128,
        win_len=96,
        d_model=16,
        d_ff=32,
        e_layers=2,
        top_k=3,
        num_kernels=6,
        dropout_rate=0.1,
        lr=0.0001,
        epochs=20,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.win_len = win_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.backend_ = TimesNetBackbone(
            seq_len=int(self.win_len),
            pred_len=0,
            d_model=int(self.d_model),
            d_ff=int(self.d_ff),
            enc_in=1,
            c_out=1,
            e_layers=int(self.e_layers),
            top_k=int(self.top_k),
            num_kernels=int(self.num_kernels),
            dropout=float(self.dropout_rate),
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

        self.backend_.to(self.device_)
        self.backend_.train()

        optimizer = Adam(self.backend_.parameters(), lr=lr)
        criterion = nn.MSELoss()

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
                output = self.backend_(x)
                loss = criterion(output, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())

            val_loss = None
            if early_stopping is not None:
                val_loss = self._valid_loss(X_val)
                improved = early_stopping.update(val_loss)
                if improved:
                    self._save_best_weights(epoch + 1)
                self.backend_.train()
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
        criterion = nn.MSELoss()
        losses = []
        self.backend_.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device_)
                output = self.backend_(x)
                losses.append(criterion(output, x).cpu().item())
        return float(np.mean(losses)) if losses else float("inf")

    def _score_window(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        output = self.backend_(x)
        score = (x - output).pow(2).squeeze(-1)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def _transform_scores(self, x_test: np.ndarray, **kwargs):
        loader = self._get_dataloader(x_test, flag="val")
        outputs, scores = [], []
        self.backend_.eval()

        with torch.no_grad():
            for x in loader:
                x = x.to(self.device_)
                output = self.backend_(x)

                raw_last = x[:, -1, 0]
                est_last = output[:, -1, 0]
                score = (raw_last - est_last).pow(2)

                scores.append(score.cpu().numpy())
                outputs.append(est_last.cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        output = np.concatenate(outputs, axis=0)
        return np.pad(
            np.asarray(scores, dtype=float),
            (int(self.win_len) - 1, 0), constant_values=0,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
