from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .Base import BaseModel


class CAEBackbone(nn.Module):
    """Skeleton for a 1D convolutional autoencoder backbone."""

    def __init__(self, win_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.win_len = win_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("CAEBackbone.forward is not implemented yet.")


class CAE(BaseModel):
    """Convolutional autoencoder TSAD skeleton."""

    HP = {
        "batch_size": 128,
        "win_len": 128,
        "hidden_dim": 32,
        "latent_dim": 16,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.backend = CAEBackbone(
            win_len=int(self.params["win_len"]),
            hidden_dim=int(self.params["hidden_dim"]),
            latent_dim=int(self.params["latent_dim"]),
        ).to(self.device)

    def _fit(
        self,
        x_train: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("CAE._fit is not implemented yet.")

    def _detect(self, x: np.ndarray):
        raise NotImplementedError("CAE._detect is not implemented yet.")
