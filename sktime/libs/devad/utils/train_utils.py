import os
import numpy as np
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, mode="min", patience=7, verbose=False, delta=0.0):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.patience = int(patience)
        self.verbose = verbose
        self.delta = float(delta)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = np.inf if mode == "min" else -np.inf

    def __call__(self, metric, model, ckpt_path, epoch=None):
        metric = float(metric)
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            self.best_metric = metric
            self.save_checkpoint(metric, model, ckpt_path, epoch=epoch)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.best_metric = metric
        self.counter = 0
        self.save_checkpoint(metric, model, ckpt_path, epoch=epoch)

    def save_checkpoint(self, metric, model, ckpt_path, epoch=None):
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        if self.verbose:
            print(f"Validation metric improved to {metric:.6f}. Saving model to {ckpt_path}")
        if epoch is not None:
            model.best_epoch = int(epoch)
        if hasattr(model, "save") and callable(model.save):
            model.save(ckpt_path)
        else:
            torch.save(model.state_dict(), ckpt_path)
