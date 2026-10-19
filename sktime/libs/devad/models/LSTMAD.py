import numpy as np
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from ..utils.dataset import ForecastDataset
from .Base import BaseModel, DetectResult, REQUIRED
from torch.optim import Adam
from ..utils.train_utils import EarlyStopping
from ..utils.training_reporter import NullTrainingReporter, TrainingReporter

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
    
class LSTMAD(BaseModel):
    HP = {
        "batch_size": REQUIRED,
        "win_len": REQUIRED,
        "h_dim": REQUIRED,
        "pred_len": REQUIRED,
        "num_layers": REQUIRED,
        "lr": REQUIRED,
        "epochs": REQUIRED,
        "scale_mode": "zscore",
        "early_stop_patience": 7,
        "early_stop_delta": 0.0,
    }

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = int(self.params["batch_size"])
        self.backend = LSTMModel(
            win_len=int(self.params["win_len"]),
            feats=1,
            hidden_dim=int(self.params["h_dim"]),
            pred_len=int(self.params["pred_len"]),
            num_layers=int(self.params["num_layers"]),
        ).to(self.device)
        
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
        loss_fn = nn.MSELoss()
        
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
            for x, target in train_loader:
                optimizer.zero_grad()
                # (bs, win, feat) (bs, pred_len, feat)
                x, target = x.to(self.device), target.to(self.device)
                # (bs, pred_len, feat)
                output = self.backend(x)    

                output = output.view(output.size(0), -1)
                target = target.view(target.size(0), -1)

                loss = loss_fn(output, target)
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
        criterion = nn.MSELoss()
        losses = []
        self.backend.eval()
        with torch.no_grad():
            for x, target in val_loader:
                x, target = x.to(self.device), target.to(self.device)
                output = self.backend(x)
                output = output.view(output.size(0), -1)
                target = target.view(target.size(0), -1)
                losses.append(criterion(output, target).cpu().item())
        return float(np.mean(losses))

    def _detect(self, x_test: np.ndarray) -> DetectResult:
        loader = self._get_dataloader(x_test, flag="val")
        outputs, scores = [], []
        self.backend.eval()
        with torch.no_grad():
            for x, target in loader:
                x, target = x.to(self.device), target.to(self.device)
                output = self.backend(x)
                
                output = output.view(output.size(0), -1)
                target = target.view(target.size(0), -1)

                mse = (output - target).pow(2).mean(dim=1)     # (bs,)
                scores.append(mse.cpu().numpy())
                outputs.append(output[:, -1].cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        output = np.concatenate(outputs, axis=0)
        start_pos = int(self.params["win_len"]) + int(self.params["pred_len"]) - 1
        return DetectResult(scores=scores, output=output, start_pos=start_pos)

    def _get_dataloader(self, x: np.ndarray, flag="train"):
        win_len = int(self.params["win_len"])
        pred_len = int(self.params["pred_len"])
        scale_mode = self.params["scale_mode"]
        assert flag in ["train", "val"]

        if flag == "train":
            dataset = ForecastDataset(
                raw_seqs=x,
                win_len=win_len,
                pred_len=pred_len,
                scale_mode=scale_mode,
                scale_cfg=None,
                flag="train",
            )
            self.scale_cfg = dataset.get_scale_cfg()
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        dataset = ForecastDataset(
            raw_seqs=x,
            win_len=win_len,
            pred_len=pred_len,
            scale_mode=scale_mode,
            scale_cfg=self.scale_cfg,
            flag="val",
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
