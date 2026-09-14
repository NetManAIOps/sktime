"""ModernTCN single-stage anomaly detector, ported from the author's detection code.

Source: luodhhh/ModernTCN, commit 56a9a2c018385cd5acef015378cae7f084d1b11c.
MIT License: see ModernTCN.LICENSE.

Uses the single-stage architecture in the author's detection scripts. Unused
forecasting heads, multi-task wrappers and inference kernel fusion are omitted.
The patch decoder intentionally concatenates then truncates, as upstream does.
DevAD uses train-only scaling, external validation and stride-1 windows.
A whole-window MSE is assigned to its endpoint (not upstream's flattened scores).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .Base import REQUIRED, BaseModel, DetectResult
from ..utils.dataset import ReconDataset
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


def conv_bn(channels: int, kernel_size: int) -> nn.Sequential:
    layers = nn.Sequential()
    layers.add_module("conv", nn.Conv1d(
        channels, channels, kernel_size, padding=kernel_size // 2,
        groups=channels, bias=False,
    ))
    layers.add_module("bn", nn.BatchNorm1d(channels))
    return layers


class ReparamLargeKernelConv(nn.Module):
    """Keep both training branches; no deployment-time kernel merging."""

    def __init__(self, channels, kernel_size, small_kernel):
        super().__init__()
        self.lkb_origin = conv_bn(channels, kernel_size)
        if small_kernel is not None:
            self.small_conv = conv_bn(channels, small_kernel)

    def forward(self, x):
        out = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out += self.small_conv(x)
        return out


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(nvars * dmodel, large_size, small_size)
        self.norm = nn.BatchNorm1d(dmodel)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self,x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        x = self.dw(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x

class ModernTCNBackbone(nn.Module):
    """Author's single-stage detection path, with [B, W, C] input/output."""

    def __init__(self, win_len, patch_size, patch_stride, num_blocks, d_model,
                 large_size, small_size, ffn_ratio, dropout, revin, affine,
                 subtract_last, channels=1):
        super().__init__()
        self.win_len = win_len
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(channels, affine=affine, subtract_last=subtract_last)
        self.downsample_layers = nn.ModuleList([nn.Linear(patch_size, d_model)])
        stage = nn.Module()
        stage.blocks = nn.ModuleList([
            Block(large_size, small_size, d_model, d_model * ffn_ratio,
                  channels, drop=dropout)
            for _ in range(num_blocks)
        ])
        self.stages = nn.ModuleList([stage])
        self.head_dection1 = nn.Linear(d_model, patch_size)

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)
        batch, channels, _ = x.shape
        # Keep the author's [B*C, 1, W] stem and tensor layout.
        x = x.reshape(batch * channels, 1, -1)
        pad_len = self.patch_size - self.patch_stride
        if pad_len:
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
        x = x.reshape(batch, channels, -1)
        x = x.unfold(-1, self.patch_size, self.patch_stride)
        x = self.downsample_layers[0](x).permute(0, 1, 3, 2)
        for block in self.stages[0].blocks:
            x = block(x)
        x = self.head_dection1(x.permute(0, 1, 3, 2))
        # Not overlap-add: this is the author's actual detection head.
        x = x.reshape(batch, channels, -1)[:, :, :self.win_len].permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x


class ModernTCN(BaseModel):
    # Architecture/training defaults follow the official MSL detection script.
    HP = {
        "win_len": REQUIRED,
        "batch_size": 128,
        "patch_size": 8,
        "patch_stride": 4,
        "num_blocks": 1,
        "d_model": 8,
        "large_size": 51,
        "small_size": 5,
        "ffn_ratio": 1,
        "dropout": 0.1,
        "revin": True,
        "affine": False,
        "subtract_last": False,
        "scale_mode": "zscore",
        "epochs": 2,
        "lr": 5e-4,
        "pct_start": 0.3,
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.criterion = nn.MSELoss()

        backbone_hp = {
            key: self.params[key] for key in (
                "win_len", "patch_size", "patch_stride", "num_blocks", "d_model",
                "large_size", "small_size", "ffn_ratio", "dropout", "revin",
                "affine", "subtract_last",
            )
        } 
        self.backend = ModernTCNBackbone(**backbone_hp).to(self.device)

    def _get_dataloader(self, x, flag="train"):

        dataset = ReconDataset(
            x, win_len=self.params["win_len"], flag=flag, scale_cfg=self.scale_cfg,
            scale_mode=self.params["scale_mode"], clip=False,
        )
        if flag == "train":
            self.scale_cfg = dataset.get_scale_cfg()
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=flag == "train")

    def _make_optimizer(self, train_steps):
        optimizer = torch.optim.Adam(self.backend.parameters(), lr=self.params["lr"])
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.params["lr"], steps_per_epoch=train_steps,
            epochs=self.params["epochs"], pct_start=self.params["pct_start"],
        )
        return optimizer

    def _fit(
        self,
        x_train,
        y=None,
        x_val=None,
        checkpoint_path=None,
        reporter: TrainingReporter | None = None,
    ):
        reporter = reporter or NullTrainingReporter()
        train_loader = self._get_dataloader(x_train)
        if x_val is not None:
            x_val = self.check_array(x_val, dtype=np.float32, name="x_val")
        optimizer = self._make_optimizer(len(train_loader))
        early_stopping = None
        if self.params["early_stop_patience"] > 0 and checkpoint_path and x_val is not None:
            early_stopping = EarlyStopping(
                patience=self.params["early_stop_patience"],
                delta=self.params["early_stop_delta"],
            )
        epochs = self.params["epochs"]
        for epoch in range(epochs):
            self.backend.train()
            reporter.begin_epoch(epoch + 1, epochs, len(train_loader))
            for batch in train_loader:
                batch = batch.to(self.device).unsqueeze(-1)
                optimizer.zero_grad()
                reconstruction = self.backend(batch)
                loss = self.criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                reporter.step(loss.item())
            val_loss = None
            if x_val is not None:
                val_loss = self._valid_loss(x_val)
            if early_stopping is not None:
                early_stopping(val_loss, self, checkpoint_path, epoch=epoch + 1)
            stopped_early = early_stopping is not None and early_stopping.early_stop
            reporter.end_epoch(
                val_loss=val_loss,
                best_epoch=self.best_epoch,
                early_stopping=early_stopping,
            )
            if stopped_early:
                break
            # Official adjust_learning_rate(..., epoch + 1), lradj='type1'.
            for group in optimizer.param_groups:
                group["lr"] = self.params["lr"] * 0.5 ** epoch
        if early_stopping is not None and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    def _valid_loss(self, x_val):
        self.backend.eval()
        losses = []
        with torch.no_grad():
            for batch in self._get_dataloader(x_val, flag="val"):
                batch = batch.to(self.device).unsqueeze(-1)
                losses.append(self.criterion(self.backend(batch), batch).item())
        return float(np.mean(losses))

    def _detect(self, x_test):
        self.backend.eval()
        scores, outputs = [], []
        with torch.no_grad():
            for batch in self._get_dataloader(x_test, flag="test"):
                batch = batch.to(self.device).unsqueeze(-1)
                reconstruction = self.backend(batch)
                # Whole-window error, assigned only to the window endpoint.
                scores.append((reconstruction - batch).square().mean(dim=(1, 2)).cpu().numpy())
                outputs.append(reconstruction[:, -1, 0].cpu().numpy())
        return DetectResult(
            scores=np.concatenate(scores), output=np.concatenate(outputs),
            start_pos=self.params["win_len"] - 1,
        )
