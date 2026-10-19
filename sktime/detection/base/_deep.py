"""Common lifecycle for PyTorch detection models."""

import random
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from sktime.detection.base._base import BaseDetector

__all__ = ["BaseDeepDetector"]


class BaseDeepDetector(BaseDetector, ABC):
    """Base class for PyTorch detectors.

    Parameters
    ----------
    seed : int, default=0
        Seed used before network construction, training and scoring.
    device : str, default="cpu"
        PyTorch execution device, for example "cpu", "cuda" or "mps".

    Notes
    -----
    Subclasses implement ``_get_dataloader``, ``_build_network``, ``_fit`` and
    ``_transform_scores``. ``_get_dataloader`` is a required abstract method;
    each model chooses its own dataset, shuffling and batching behavior.
    The constructor only stores configuration. Each fit creates a fresh network
    in ``backend_`` and records the resolved device in ``device_``.
    Training loops, validation losses and early stopping belong to subclasses.
    Subclasses can call ``_save_best_weights(epoch)`` on improvement; fit restores
    the saved weights after training and removes the temporary snapshot.
    Scores are returned unchanged; this base does not impose a result format.
    Whole-object save/load and parameter management are inherited from sktime.
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "np.ndarray",
        "fit_is_empty": False,
        "capability:update": False,
        "python_dependencies": ["torch", "rich"],
        "property:randomness": "stochastic",
    }

    def __init__(self, seed=0, device="cpu"):
        self.seed = seed
        self.device = device
        super().__init__()

    def fit(self, X, y=None, *, X_val=None, show_progress=False, **kwargs):
        """Train a fresh network, optionally using separate validation data.

        X and X_val follow the same sktime input conversion rules. y retains its
        meaning as optional training labels and is passed unchanged to the model.
        show_progress controls terminal progress; no log file is created here.
        A temporary reporter is passed to the subclass's _fit method.
        Additional kwargs, including preprocessing masks, are forwarded to _fit.
        Returns self only after training succeeds.
        """
        self.reset()

        import torch
        from sktime.detection.utils.reporter import (
            NullTrainingReporter,
            TrainingReporter,
        )

        X_inner = self._check_X(X)
        X_val_inner = self._check_X(X_val) if X_val is not None else None
        self._X = X
        self._y = y

        self._set_seed()
        self.device_ = torch.device(self.device)
        backend = self._build_network()
        if not isinstance(backend, torch.nn.Module):
            raise TypeError("_build_network() must return a torch.nn.Module")
        self.backend_ = backend.to(self.device_)
        self.backend_.train()
        self.best_epoch_ = None

        if show_progress:
            reporter = TrainingReporter(
                info={
                    "Model": type(self).__name__,
                    "Device": str(self.device_),
                    "Train points": len(X_inner),
                    "Validation points": 0 if X_val_inner is None else len(X_val_inner),
                    "Parameters": self.get_params(deep=False),
                },
            )
        else:
            reporter = NullTrainingReporter()

        with reporter:
            self._fit(X_inner, y=y, X_val=X_val_inner, reporter=reporter, **kwargs)
            if hasattr(self, "_best_state"):
                self._restore_best_weights()
            reporter.finish(best_epoch=self.best_epoch_)
        self._is_fitted = True
        return self

    def transform_scores(self, X, **kwargs):
        import torch

        self.check_is_fitted()
        X_inner = self._check_X(X)

        self._set_seed()
        self.backend_.eval()
        with torch.no_grad():
            return self._transform_scores(X_inner, **kwargs)

    def _save_best_weights(self, epoch):
        """Keep a weight snapshot when the subclass finds an improvement."""
        self._best_state = deepcopy(self.backend_.state_dict())
        self.best_epoch_ = epoch

    def _restore_best_weights(self):
        """Restore and discard the temporary weight snapshot."""
        self.backend_.load_state_dict(self._best_state)
        del self._best_state

    def _set_seed(self):
        """Seed the RNGs also used by the existing DevAD models."""
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @abstractmethod
    def _get_dataloader(self, X, **kwargs):
        """Build the model's DataLoader; no default implementation is provided."""
        raise NotImplementedError("Subclasses must implement _get_dataloader().")

    def _build_network(self):
        """Return a new nn.Module; called after seeding on every fit."""
        raise NotImplementedError("Subclasses must implement _build_network().")

    def _fit(self, X, y=None, *, X_val=None, reporter=None, **kwargs):
        """Train the network and notify reporter of epoch/batch progress."""
        raise NotImplementedError("Subclasses must implement _fit")

    def _transform_scores(self, X, **kwargs):
        """Compute scores with backend_ in eval mode and gradients disabled."""
        raise NotImplementedError("Subclasses must implement _transform_scores().")
