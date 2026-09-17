# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Default-constructible DevAD point-anomaly detectors.

Thin wrappers binding one family each from the vendored DevAD model zoo
(``sktime.libs.devad``) as an sktime detector, so the families are
constructible without arguments and show up in registry discovery (and hence
in the Playground/labts catalog).

DevAD models share a ``fit(x_train)`` / ``detect(x_test)`` API returning
per-point anomaly scores with a ``start_pos`` offset (windowed models cannot
score the first ``win_len - 1`` points). The wrappers here threshold the
scores at a quantile and report the exceedance positions as point anomalies.
"""

__all__ = [
    "DevADBeatGANDetector",
    "DevADCOUTADetector",
    "DevADDonutDetector",
    "DevADFCVAEDetector",
    "DevADFITSDetector",
    "DevADIForestDetector",
    "DevADKANADDetector",
    "DevADKMeansADDetector",
    "DevADLSTMADDetector",
    "DevADModernTCNDetector",
    "DevADOmniAnomalyDetector",
    "DevADSubLOFDetector",
    "DevADSubOCSVMDetector",
    "DevADSubPCADetector",
    "DevADTimesNetDetector",
    "DevADTranADDetector",
    "DevADUSADDetector",
]

__author__ = ["xiezhe", "Tinyfire27"]

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _safe_import

ModelRegistry = _safe_import("sktime.libs.devad.models.registry.ModelRegistry")


def scores_to_point_ilocs(scores, start_pos, threshold_quantile=0.99):
    """Positions where the anomaly score exceeds its own quantile.

    ``start_pos`` shifts score indices back to input positions: DevAD's
    right-aligned suffix contract places ``scores[i]`` at input position
    ``start_pos + i``.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    quantile = min(max(float(threshold_quantile), 0.0), 1.0)
    threshold = np.quantile(scores, quantile)
    return np.nonzero(scores > threshold)[0] + int(start_pos)


def _to_1d_values(X):
    """Flatten a univariate Series/DataFrame to the 1-D array DevAD expects."""
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    return np.asarray(X, dtype=float).ravel()


class _DevADDetectorBase(BaseDetector):
    """Base adapter wrapping one DevAD model family as a point detector.

    Parameters
    ----------
    threshold_quantile : float, default=0.99
        Scores above this quantile of the detected series are reported as
        point anomalies. Lower values flag more points.
    win_len : int or None, default=None
        Sliding-window length forwarded to the DevAD model. None keeps the
        family default declared in its ``HP`` table.
    epochs : int or None, default=None
        Training epochs for gradient-based families. None keeps the family
        default. Ignored by non-trainable families.
    batch_size : int or None, default=None
        Training batch size for gradient-based families. None keeps the
        family default.
    seed : int, default=2026
        Random seed (DevAD seeds numpy/torch/sklearn via ``setup_seed``).
    device : str, default="cpu"
        Torch device string, e.g. "cpu", "cuda", "mps".
    params : dict, JSON-object string, or None; default=None
        Extra DevAD hyperparameters keyed by the family's ``HP`` names
        (e.g. ``{"h_dim": 64}``). A string is parsed as JSON, so the CLI can
        pass ``--param params='{"h_dim": 64}'``. Unknown keys raise at fit
        time; explicit arguments above take precedence over entries here.
    """

    _tags = {
        "authors": ["xiezhe", "Tinyfire27"],
        "maintainers": ["xiezhe"],
        "python_dependencies": ["rich"],
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
    }

    # DevAD registry key, set by each subclass.
    family = None
    # Constructor values for families whose HP table has REQUIRED entries.
    _default_overrides: dict = {}

    def __init__(
        self,
        threshold_quantile=0.99,
        win_len=None,
        epochs=None,
        batch_size=None,
        seed=2026,
        device="cpu",
        params=None,
    ):
        self.threshold_quantile = threshold_quantile
        self.win_len = win_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.params = params
        super().__init__()

    def _devad_params(self):
        extra = self.params
        if isinstance(extra, str):
            import json

            extra = json.loads(extra)
        params = {**self._default_overrides, **(extra or {})}
        for key, value in (
            ("win_len", self.win_len),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
        ):
            if value is not None:
                params[key] = value
        return params

    def _fit(self, X, y=None):
        model = ModelRegistry.create_model(
            family=self.family,
            params=self._devad_params(),
            seed=int(self.seed),
            device=str(self.device),
        )
        model.fit(_to_1d_values(X))
        self._model = model
        return self

    def _predict(self, X):
        result = self._model.detect(_to_1d_values(X))
        ilocs = scores_to_point_ilocs(
            result.scores, result.start_pos, self.threshold_quantile
        )
        return pd.Series(ilocs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Small, fast configuration for sktime's estimator checks."""
        return {
            "threshold_quantile": 0.9,
            "win_len": 8,
            "epochs": 1,
            "batch_size": 16,
        }


class DevADBeatGANDetector(_DevADDetectorBase):
    """BeatGAN: reconstruction anomaly detection with adversarial training
    (beat-regularized GAN over sliding windows) [1]_.

    References
    ----------
    .. [1] Zhou, B. et al. BeatGAN: Anomalous Rhythm Detection using
       Adversarially Generated Time Series. IJCAI 2019.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "beatgan"


class DevADCOUTADetector(_DevADDetectorBase):
    """COUTA: calibrated one-class classification with uncertainty
    regularization for time-series anomaly detection [1]_.

    References
    ----------
    .. [1] Xu, H. et al. Calibrated One-class Classification for Unsupervised
       Time Series Anomaly Detection. TKDE 2023.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "couta"


class DevADDonutDetector(_DevADDetectorBase):
    """Donut: seasonal VAE reconstruction for web KPI anomaly detection [1]_.

    References
    ----------
    .. [1] Xu, H. et al. Unsupervised Anomaly Detection via Variational
       Auto-Encoder for Seasonal KPIs in Web Applications. WWW 2018.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "donut"
    _default_overrides = {"win_len": 120, "h_dim": 64, "z_dim": 8}


class DevADFCVAEDetector(_DevADDetectorBase):
    """FCVAE: frequency-conditioned VAE for time-series anomaly detection [1]_.

    References
    ----------
    .. [1] Liu, Y. et al. FCVAE: Frequency Conditioned Variational Autoencoder
       for Time Series Anomaly Detection. ICDE 2024.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "fcvae"


class DevADFITSDetector(_DevADDetectorBase):
    """FITS: lightweight frequency-interpolation reconstruction model [1]_.

    References
    ----------
    .. [1] Xu, Z., Zeng, A. and Xu, Q. FITS: Modeling Time Series with
       10k Parameters. ICLR 2024.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "fits"


class DevADIForestDetector(_DevADDetectorBase):
    """Isolation Forest over sliding-window features with z-score/min-max
    normalization [1]_.

    References
    ----------
    .. [1] Liu, F.T., Ting, K.M. and Zhou, Z.H. Isolation Forest. ICDM 2008.
    """

    family = "iforest"


class DevADKANADDetector(_DevADDetectorBase):
    """KAN-AD: Kolmogorov-Arnold-network predictor over differenced,
    scaled windows; prediction error is the anomaly score [1]_.

    References
    ----------
    .. [1] Zamanzadeh Darban, Z. et al. KAN-AD: Kolmogorov-Arnold Network
       for Online Time Series Anomaly Detection. ICDM 2024.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "kan_ad"
    _default_overrides = {"win_len": 100}


class DevADKMeansADDetector(_DevADDetectorBase):
    """K-Means over sliding windows; distance to the assigned centroid is
    the anomaly score."""

    family = "kmeans_ad"
    _default_overrides = {"win_len": 24}


class DevADLSTMADDetector(_DevADDetectorBase):
    """LSTM-AD: stacked-LSTM predictor; prediction error is the anomaly
    score [1]_.

    References
    ----------
    .. [1] Malhotra, P. et al. Long Short Term Memory Networks for Anomaly
       Detection in Time Series. ESANN 2015.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "lstm_ad"
    _default_overrides = {
        "batch_size": 64,
        "win_len": 30,
        "h_dim": 32,
        "pred_len": 1,
        "num_layers": 1,
        "lr": 1e-3,
        "epochs": 10,
    }


class DevADModernTCNDetector(_DevADDetectorBase):
    """ModernTCN: modernized temporal convolution network reconstruction
    model [1]_.

    References
    ----------
    .. [1] Luo, D. and Wang, X. ModernTCN: A Modern Pure Convolution
       Structure for General Time Series Analysis. ICLR 2024.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "modern_tcn"
    _default_overrides = {"win_len": 100}


class DevADOmniAnomalyDetector(_DevADDetectorBase):
    """OmniAnomaly: stochastic recurrent VAE with normalizing-flow
    reconstruction scores [1]_.

    References
    ----------
    .. [1] Su, Y. et al. Robust Anomaly Detection for Multivariate Time
       Series through Stochastic Recurrent Neural Network. KDD 2019.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "omni_anomaly"
    _default_overrides = {"win_len": 100}


class DevADSubLOFDetector(_DevADDetectorBase):
    """Subsequence LOF: local outlier factor over sliding windows [1]_.

    References
    ----------
    .. [1] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J. LOF:
       Identifying Density-based Local Outliers. SIGMOD 2000.
    """

    family = "sub_lof"
    _default_overrides = {"win_len": 24}


class DevADSubOCSVMDetector(_DevADDetectorBase):
    """Subsequence one-class SVM over standardized sliding windows [1]_.

    References
    ----------
    .. [1] Schölkopf, B. et al. Estimating the Support of a High-Dimensional
       Distribution. Neural Computation 2001.
    """

    family = "sub_ocsvm"
    _default_overrides = {"win_len": 24}


class DevADSubPCADetector(_DevADDetectorBase):
    """Subsequence PCA: distance to the least-variance principal components
    over sliding windows."""

    family = "sub_pca"
    _default_overrides = {"win_len": 24}


class DevADTimesNetDetector(_DevADDetectorBase):
    """TimesNet: inception-block reconstruction over period-folded
    time series [1]_.

    References
    ----------
    .. [1] Wu, H. et al. TimesNet: Temporal 2D-Variation Modeling for
       General Time Series Analysis. ICLR 2023.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "timesnet"


class DevADTranADDetector(_DevADDetectorBase):
    """TranAD: transformer with adversarial self-conditioning for anomaly
    detection [1]_.

    References
    ----------
    .. [1] Tuli, S. et al. TranAD: Deep Transformer Networks for Anomaly
       Detection in Multivariate Time Series Data. VLDB 2022.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "tranad"


class DevADUSADDetector(_DevADDetectorBase):
    """USAD: autoencoder with adversarially-trained twin decoders [1]_.

    References
    ----------
    .. [1] Audibert, J. et al. USAD: UnSupervised Anomaly Detection on
       Multivariate Time Series. KDD 2020.
    """

    _tags = {"python_dependencies": ["rich", "torch"]}

    family = "usad"
