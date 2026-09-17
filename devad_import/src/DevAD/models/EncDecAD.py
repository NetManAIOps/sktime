from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .Base import BaseModel


class EncDecADBackbone(nn.Module):
    """Skeleton for an LSTM encoder-decoder reconstruction backbone."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("EncDecADBackbone.forward is not implemented yet.")


class EncDecAD(BaseModel):
    """LSTM encoder-decoder reconstruction TSAD skeleton."""

    HP = {
        "batch_size": 128,
        "input_dim": 1,
        "hidden_dim": 20,
        "num_layers": 2,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.backend = EncDecADBackbone(
            input_dim=int(self.params["input_dim"]),
            hidden_dim=int(self.params["hidden_dim"]),
            num_layers=int(self.params["num_layers"]),
        ).to(self.device)

    def _fit(
        self,
        x_train: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("EncDecAD._fit is not implemented yet.")

    def _detect(self, x: np.ndarray):
        raise NotImplementedError("EncDecAD._detect is not implemented yet.")
