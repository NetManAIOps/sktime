import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sktime.detection.base import BaseDeepDetector
from sktime.detection.utils.dataset import ForecastDataset
from sktime.detection.utils.early_stopping import EarlyStopping


class LSTMModel(nn.Module):
    def __init__(self, win_len, feats,
                 hidden_dim, pred_len, num_layers) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.feats = feats

        self.lstm_encoder = nn.LSTM(input_size=feats, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=feats, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.relu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, feats)

    def forward(self, src):
        # src: (bs, win_len, feats)
        _, decoder_hidden = self.lstm_encoder(src)
        bs = src.size(0)

        decoder_input = src[:, -1:, :].contiguous()         # (bs, 1, feats)
        outputs = src.new_zeros(bs, self.pred_len, self.feats)    # (bs, pred_len, feats)

        for t in range(self.pred_len):
            dec_out, decoder_hidden = self.lstm_decoder(decoder_input, decoder_hidden)  # (bs,1,h)
            dec_out = self.relu(dec_out)
            decoder_input = self.fc(dec_out)  # (bs,1,feats)

            outputs[:, t, :] = decoder_input.squeeze(1)  # 写入 (bs,feats)

        return outputs


class LSTMAD(BaseDeepDetector):
    """Native detector; input scaling belongs to an external transformer.

    Network, losses and valid-window scores retain the DevAD arithmetic.
    Scores are padded with zeros before the first valid window.
    """

    _tags = {"capability:multivariate": False, "capability:missing_values": False}

    def __init__(
        self,
        batch_size,
        win_len,
        h_dim,
        pred_len,
        num_layers,
        lr,
        epochs,
        early_stop_patience=7,
        early_stop_delta=0.0,
        seed=0,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.win_len = win_len
        self.h_dim = h_dim
        self.pred_len = pred_len
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        super().__init__(seed=seed, device=device)

    def _build_network(self):
        self.backend_ = LSTMModel(
            win_len=int(self.win_len),
            feats=1,
            hidden_dim=int(self.h_dim),
            pred_len=int(self.pred_len),
            num_layers=int(self.num_layers),
        )

        return self.backend_

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        self._set_seed()
        train_loader = self._get_dataloader(X, flag="train")
        lr = float(self.lr)
        epochs = int(self.epochs)

        self.backend_.to(self.device_)
        self.backend_.train()

        optimizer = Adam(self.backend_.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

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
            for x, target in train_loader:
                optimizer.zero_grad()
                # (bs, win, feat) (bs, pred_len, feat)
                x, target = x.to(self.device_), target.to(self.device_)
                # (bs, pred_len, feat)
                output = self.backend_(x)

                output = output.view(output.size(0), -1)
                target = target.view(target.size(0), -1)

                loss = loss_fn(output, target)
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
            for x, target in val_loader:
                x, target = x.to(self.device_), target.to(self.device_)
                output = self.backend_(x)
                output = output.view(output.size(0), -1)
                target = target.view(target.size(0), -1)
                losses.append(criterion(output, target).cpu().item())
        return float(np.mean(losses))

    def _transform_scores(self, x_test: np.ndarray, **kwargs):
        loader = self._get_dataloader(x_test, flag="val")
        outputs, scores = [], []
        self.backend_.eval()
        with torch.no_grad():
            for x, target in loader:
                x, target = x.to(self.device_), target.to(self.device_)
                output = self.backend_(x)

                output = output.view(output.size(0), -1)
                target = target.view(target.size(0), -1)

                mse = (output - target).pow(2).mean(dim=1)     # (bs,)
                scores.append(mse.cpu().numpy())
                outputs.append(output[:, -1].cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        output = np.concatenate(outputs, axis=0)
        start_pos = int(self.win_len) + int(self.pred_len) - 1
        return np.pad(
            np.asarray(scores, dtype=float),
            (start_pos, 0), constant_values=0,
        )

    def _get_dataloader(self, x, flag="train"):
        dataset = ForecastDataset(x, win_len=int(self.win_len), pred_len=int(self.pred_len))
        return DataLoader(
            dataset, batch_size=int(self.batch_size), shuffle=flag == "train",
            drop_last=False,
        )

    def _predict(self, X):
        raise NotImplementedError(
            "Use transform_scores; an anomaly threshold policy has not been defined."
        )
