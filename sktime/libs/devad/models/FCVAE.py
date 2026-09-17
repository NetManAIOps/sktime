from __future__ import annotations

from types import SimpleNamespace
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from .Base import BaseModel, DetectResult
from ..utils.dataset import ReconDataset
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter


# 注入缺失值的数据增强
def _missing_data_injection(x, y, z, rate: float):
    miss_size = int(rate * x.shape[0] * x.shape[1] * x.shape[2])
    if miss_size <= 0:
        return x, y, z
    row = torch.randint(low=0, high=x.shape[0], size=(miss_size,), device=x.device)
    col = torch.randint(low=0, high=x.shape[2], size=(miss_size,), device=x.device)
    x[row, :, col] = 0
    z[row, col] = 1
    return x, y, z

# 注入点异常值的数据增强
def _point_ano(x, y, z, rate: float):
    aug_size = int(rate * x.shape[0])
    if aug_size <= 0:
        return x, y, z
    id_x = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    x_aug = x[id_x].clone()
    y_aug = y[id_x].clone()
    z_aug = z[id_x].clone()
    if x_aug.shape[1] == 1:
        half = int(aug_size / 2)
        ano_noise1 = torch.randint(low=1, high=20, size=(half,), device=x.device)
        ano_noise2 = torch.randint(low=-20, high=-1, size=(aug_size - half,), device=x.device)
        ano_noise = (torch.cat((ano_noise1, ano_noise2), dim=0) / 2.0).to(x.device)
        x_aug[:, 0, -1] += ano_noise
        y_aug[:, -1] = torch.logical_or(y_aug[:, -1], torch.ones_like(y_aug[:, -1]))
    return x_aug, y_aug, z_aug

# 注入片段异常的数据增强
def _seg_ano(x, y, z, rate: float, method: str = "swap"):
    aug_size = int(rate * x.shape[0])
    if aug_size <= 0:
        return x, y, z
    idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    while torch.any(idx_1 == idx_2):
        idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()
    time_start = torch.randint(low=7, high=x.shape[2], size=(aug_size,), device=x.device)
    for i in range(len(idx_2)):
        if method == "swap":
            x_aug[i, :, time_start[i] :] = x[idx_2[i], :, time_start[i] :]
            y_aug[i, time_start[i] :] = torch.logical_or(
                y_aug[i, time_start[i] :], torch.ones_like(y_aug[i, time_start[i] :])
            )
    return x_aug, y_aug, z_aug


def _rfft(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    dim = dim if dim >= 0 else input_tensor.dim() + dim
    out_shape = list(input_tensor.shape)
    out_shape[dim] = input_tensor.shape[dim] // 2 + 1
    out_dtype = torch.complex128 if input_tensor.dtype == torch.float64 else torch.complex64
    out = torch.empty(tuple(out_shape), dtype=out_dtype, device=input_tensor.device)
    return torch.fft.rfft(input_tensor, dim=dim, out=out)


class FCVAE(BaseModel):
    HP = {
        "batch_size": 256,
        "win_len": 64,
        "latent_dim": 8,
        "condition_emb_dim": 16,
        "d_model": 256,
        "d_inner": 512,
        "n_head": 8,
        "kernel_size": 16,
        "stride": 8,
        "dropout_rate": 0.05,
        "mcmc_rate": 0.2,
        "mcmc_value": -5,
        "mcmc_mode": 2,
        "scale_mode": "zscore",
        "sliding_window_size": 1,
        "lr": 5e-4,
        "epochs": 30,
        "missing_data_rate": 0.0,
        "point_ano_rate": 0.0,
        "seg_ano_rate": 0.0,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        hp = SimpleNamespace(
            window=int(self.params["win_len"]),
            latent_dim=int(self.params["latent_dim"]),
            condition_emb_dim=int(self.params["condition_emb_dim"]),
            d_model=int(self.params["d_model"]),
            d_inner=int(self.params["d_inner"]),
            n_head=int(self.params["n_head"]),
            kernel_size=int(self.params["kernel_size"]),
            stride=int(self.params["stride"]),
            dropout_rate=float(self.params["dropout_rate"]),
            mcmc_rate=float(self.params["mcmc_rate"]),
            mcmc_value=float(self.params["mcmc_value"]),
            mcmc_mode=int(self.params["mcmc_mode"]),
        )
        self.hp = hp
        self.backend = CVAE(hp).to(self.device)

    def _get_dataloader(self, x: np.ndarray, flag: str):
        win_len = int(self.params["win_len"])
        scale_mode = self.params["scale_mode"]
        assert flag in ["train", "val"]

        sliding_window_size = int(self.params["sliding_window_size"])

        values = x
        mask = np.isnan(values).astype(int)
        values = pd.Series(values).bfill().fillna(0).to_numpy()

        if sliding_window_size > 1:
            kernel = np.ones((sliding_window_size,)) / sliding_window_size
            values = np.convolve(values, kernel, mode="valid")
            mask = mask[sliding_window_size - 1 :]

        if flag == "train":
            dataset = ReconDataset(values, labels=None, win_len=win_len, flag="train", scale_cfg=self.scale_cfg, scale_mode=scale_mode, mask=mask)
            self.scale_cfg = dataset.get_scale_cfg()
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        if flag == "val":
            dataset = ReconDataset(values, labels=None, win_len=win_len, flag="val", scale_cfg=self.scale_cfg, scale_mode=scale_mode, mask=mask)
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

        self.backend.to(self.device)
        self.backend.train()

        optimizer = Adam(self.backend.parameters(), lr=lr)
        missing_data_rate = float(self.params["missing_data_rate"])
        point_ano_rate = float(self.params["point_ano_rate"])
        seg_ano_rate = float(self.params["seg_ano_rate"])

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
            for x, z_all in train_loader:
                x = x.to(self.device).unsqueeze(1)
                z_all = z_all.to(self.device)
                y_all = torch.zeros_like(z_all, dtype=torch.bool)

                if point_ano_rate > 0:
                    x_a, y_a, z_a = _point_ano(x, y_all, z_all, point_ano_rate)
                    x = torch.cat((x, x_a), dim=0)
                    y_all = torch.cat((y_all, y_a), dim=0)
                    z_all = torch.cat((z_all, z_a), dim=0)
                if seg_ano_rate > 0:
                    x_a, y_a, z_a = _seg_ano(x, y_all, z_all, seg_ano_rate, method="swap")
                    x = torch.cat((x, x_a), dim=0)
                    y_all = torch.cat((y_all, y_a), dim=0)
                    z_all = torch.cat((z_all, z_a), dim=0)
                if missing_data_rate > 0:
                    x, y_all, z_all = _missing_data_injection(x, y_all, z_all, missing_data_rate)

                mask = torch.logical_not(torch.logical_or(y_all, z_all))
                _, _, _, _, _, loss = self.backend.forward(x, "train", mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())

            val_loss = None
            if early_stopping is not None:
                val_loss = self._valid_loss(x_val)
                early_stopping(val_loss, self, checkpoint_path, epoch=epoch + 1)
                self.backend.train()
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
        val_loader = self._get_dataloader(x_val, flag="val")
        losses = []
        self.backend.eval()
        with torch.no_grad():
            for x, z_all in val_loader:
                x = x.to(self.device).unsqueeze(1)
                z_all = z_all.to(self.device)
                mask = torch.logical_not(z_all)
                _, _, _, _, _, loss = self.backend.forward(x, "train", mask)
                losses.append(loss.cpu().item())

        return float(np.mean(losses)) if losses else float("inf")

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        loader = self._get_dataloader(x_test, flag="val")
        outputs, scores = [], []
        self.backend.eval()
        with torch.no_grad():
            for x, z_all in loader:
                x = x.to(self.device).unsqueeze(1)
                z_all = z_all.to(self.device)

                imputed_x, recon_log_prob = self.backend.forward(x, "test", z_all)
                score = -recon_log_prob[:, 0, -1]

                scores.append(score.cpu().numpy())
                # MCMC2 returns the imputed window, not the MC-averaged decoder mean.
                outputs.append(imputed_x[:, 0, -1].cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        output = np.concatenate(outputs, axis=0)
        start_pos = int(self.params["win_len"]) + int(self.params["sliding_window_size"]) - 2
        return DetectResult(scores=scores, output=output, start_pos=start_pos)
    
    def get_latent(self, input: torch.Tensor, sample_latent: bool = True):
        condition = self.backend.get_conditon(input)
        condition = self.backend.dropout(condition)
        mu, var = self.backend.encode(torch.cat((input, condition), dim=2))
        z = self.backend.reparameterize(mu, var) if sample_latent else mu
        return {
            "condition": condition,
            "mu": mu,
            "var": var,
            "z": z,
        }

class EncoderLayer_selfattn(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        output, attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class CVAE(nn.Module):
    def __init__(
        self,
        hp,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "C",
    ):
        super().__init__()
        self.hp = hp
        self.num_iter = 0
        self.step_max = 0
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        modules = []
        in_channels = self.hp.window + 2 * self.hp.condition_emb_dim
        self.hidden_dims = [100, 100]
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
            nn.Softplus(),
        )
        modules = []
        self.decoder_input = nn.Linear(
            self.hp.latent_dim + 2 * self.hp.condition_emb_dim, self.hidden_dims[-1]
        )
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    nn.Tanh(),
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.hp.window),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        self.fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.window), nn.Softplus()
        )
        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.hp.d_model,
                    self.hp.d_inner,
                    self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )
        self.emb_local = nn.Sequential(
            nn.Linear(2 + self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.condition_emb_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.emb_global = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.condition_emb_dim),
            nn.Tanh(),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        var = self.fc_var(result).clamp_min(1e-6)
        return [mu, var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 1, self.hidden_dims[0])
        result = self.decoder(result)
        mu_x = self.fc_mu_x(result)
        var_x = self.fc_var_x(result).clamp_min(1e-6)
        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y):
        if mode == "train" or mode == "valid":
            condition = self.get_conditon(input)
            condition = self.dropout(condition)
            mu, var = self.encode(torch.cat((input, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            rec_x = self.reparameterize(mu_x, var_x)
            loss = self.loss_func(mu_x, var_x, input, mu, var, y, z)
            return [mu_x, var_x, rec_x, mu, var, loss]
        y = y.unsqueeze(1)
        return self.MCMC2(input)

    def get_conditon(self, x):
        x_g = x
        f_global = _rfft(x_g[:, :, :-1].contiguous(), dim=-1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        f_global = self.emb_global(f_global)
        x_g = x_g.view(x.shape[0], 1, 1, -1)
        x_l = x_g.clone()
        x_l[:, :, :, -1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.hp.stride),
        )
        unfold_x = unfold(x_l)
        unfold_x = unfold_x.transpose(1, 2)
        f_local = _rfft(unfold_x.contiguous(), dim=-1)
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)
        f_local = self.emb_local(f_local)
        for enc_layer in self.atten:
            f_local, enc_slf_attn = enc_layer(f_local)
        f_local = self.out_linear(f_local)
        f_local = f_local[:, -1, :].unsqueeze(1)
        output = torch.cat((f_global, f_local), -1)
        return output

    def MCMC2(self, x):
        condition = self.get_conditon(x)
        origin_x = x.clone()
        for i in range(10):
            mu, var = self.encode(torch.cat((x, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            recon = -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)
            temp = (
                torch.from_numpy(np.percentile(recon.cpu(), self.hp.mcmc_rate, axis=-1))
                .unsqueeze(2)
                .repeat(1, 1, self.hp.window)
            ).to(x.device)
            if self.hp.mcmc_mode == 0:
                l = (temp < recon).int()
                x = mu_x * (1 - l) + origin_x * l
            if self.hp.mcmc_mode == 1:
                l = (self.hp.mcmc_value < recon).int()
                x = origin_x * l + mu_x * (1 - l)
            if self.hp.mcmc_mode == 2:
                l = torch.ones_like(origin_x)
                l[:, :, -1] = 0
                x = origin_x * l + (1 - l) * mu_x
        prob_all = 0
        mu, var = self.encode(torch.cat((x, condition), dim=2))
        for i in range(128):
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            prob_all += -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)
        return x, prob_all / 128

    def loss_func(self, mu_x, var_x, input, mu, var, y, z, mode="nottrain"):
        if mode == "train":
            self.num_iter += 1
            self.num_iter = self.num_iter % 100
        kld_weight = 0.005
        mu_x = mu_x.squeeze(1)
        var_x = var_x.squeeze(1).clamp_min(1e-6)
        input = input.squeeze(1)
        
        # 带mask的重构损失
        recon_term = torch.log(var_x) + (input - mu_x).pow(2) / var_x
        recon_term = torch.where(y.bool(), recon_term, torch.zeros_like(recon_term))
        recon_loss = torch.mean(
            0.5 * torch.mean(recon_term, dim=1),
            dim=0,
        )

        # KL-loss
        m = (torch.sum(y, dim=1, keepdim=True) / self.hp.window).repeat(
            1, self.hp.latent_dim
        )
        var = var.clamp_min(1e-6)
        kld_loss = torch.mean(
            0.5 * torch.mean(m * (z**2) - torch.log(var) - (z - mu).pow(2) / var, dim=1),
            dim=0,
        )

        # 决定了两个项的加权方式
        if self.loss_type == "B":
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recon_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        elif self.loss_type == "C":
            loss = recon_loss + kld_loss
        elif self.loss_type == "D":
            loss = recon_loss + self.num_iter / 100 * kld_loss
        else:
            raise ValueError("Undefined loss type.")
        return loss
