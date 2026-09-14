# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Default-constructible PyOD point-anomaly detectors.

Thin wrappers around ``PyODDetector`` that bind a specific pyod model with
sensible defaults, so the detectors are constructible without arguments and
show up in registry discovery (and hence in the Playground/labts catalog).
"""

__all__ = [
    "PyODCBLOFDetector",
    "PyODCOPODDetector",
    "PyODECODDetector",
    "PyODHBOSDetector",
    "PyODIForestDetector",
    "PyODKNNDetector",
    "PyODLOFDetector",
    "PyODMCDDetector",
    "PyODOCSVMDetector",
]

__author__ = ["xiezhe"]

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _safe_import

import numpy as np
import pandas as pd

ECOD = _safe_import("pyod.models.ecod.ECOD")
COPOD = _safe_import("pyod.models.copod.COPOD")
LOF = _safe_import("pyod.models.lof.LOF")
IForest = _safe_import("pyod.models.iforest.IForest")
KNN = _safe_import("pyod.models.knn.KNN")
HBOS = _safe_import("pyod.models.hbos.HBOS")
CBLOF = _safe_import("pyod.models.cblof.CBLOF")
MCD = _safe_import("pyod.models.mcd.MCD")
OCSVM = _safe_import("pyod.models.ocsvm.OCSVM")


class _PyODPointsMixin:
    """Return anomaly ilocs, not anomaly values, from _predict.

    Upstream ``PyODDetector._predict`` returns a Series of the anomalous
    points' label values (all ones, on a fresh RangeIndex); BaseDetector's
    sparse contract expects the *positions* of the anomalous points. Without
    this fix every detection collapses onto iloc 1.
    """

    def _predict(self, X):
        X_np = X.to_numpy()
        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)
        y_np = self.estimator_.predict(X_np)
        return pd.Series(np.where(y_np)[0])


class PyODECODDetector(_PyODPointsMixin, PyODDetector):
    """ECOD: unsupervised outlier detection using empirical cumulative
    distribution functions. Parameter-free and fast [1]_.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, G.H.
       ECOD: Unsupervised Outlier Detection Using Empirical Cumulative
       Distribution Functions. TKDE 2022.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, contamination=0.1):
        self.contamination = contamination
        super().__init__(ECOD(contamination=contamination))


class PyODCOPODDetector(_PyODPointsMixin, PyODDetector):
    """COPOD: copula-based outlier detection. Parameter-free [1]_.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X.
       COPOD: Copula-Based Outlier Detection. ICDM 2020.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, contamination=0.1):
        self.contamination = contamination
        super().__init__(COPOD(contamination=contamination))


class PyODLOFDetector(_PyODPointsMixin, PyODDetector):
    """Local Outlier Factor: density-deviation based outlier detection [1]_.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors used for the local density estimation.
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J. LOF:
       Identifying Density-based Local Outliers. SIGMOD 2000.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        super().__init__(LOF(n_neighbors=n_neighbors, contamination=contamination))


class PyODIForestDetector(_PyODPointsMixin, PyODDetector):
    """Isolation Forest: isolation-based outlier detection [1]_.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of isolation trees.
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Liu, F.T., Ting, K.M. and Zhou, Z.H. Isolation Forest. ICDM 2008.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, n_estimators=100, contamination=0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        super().__init__(
            IForest(n_estimators=n_estimators, contamination=contamination)
        )


class PyODKNNDetector(_PyODPointsMixin, PyODDetector):
    """k-Nearest-Neighbors distance based outlier detection [1]_.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors whose distance is aggregated.
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Ramaswamy, S., Rastogi, R. and Shim, K. Efficient Algorithms for
       Mining Outliers from Large Data Sets. SIGMOD 2000.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, n_neighbors=5, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        super().__init__(KNN(n_neighbors=n_neighbors, contamination=contamination))


class PyODHBOSDetector(_PyODPointsMixin, PyODDetector):
    """HBOS: histogram-based outlier score. Fast, linear-time [1]_.

    Parameters
    ----------
    n_bins : int, default=10
        Number of histogram bins.
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Goldstein, M. and Dengel, A. Histogram-based Outlier Score (HBOS):
       A fast Unsupervised Anomaly Detection Algorithm. KI 2012.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, n_bins=10, contamination=0.1):
        self.n_bins = n_bins
        self.contamination = contamination
        super().__init__(HBOS(n_bins=n_bins, contamination=contamination))


class PyODCBLOFDetector(_PyODPointsMixin, PyODDetector):
    """CBLOF: cluster-based local outlier factor [1]_.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters formed before scoring.
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] He, Z., Xu, X. and Deng, S. Discovering Cluster-based Local
       Outliers. Pattern Recognition Letters 2003.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, n_clusters=8, contamination=0.1):
        self.n_clusters = n_clusters
        self.contamination = contamination
        super().__init__(CBLOF(n_clusters=n_clusters, contamination=contamination))


class PyODMCDDetector(_PyODPointsMixin, PyODDetector):
    """MCD: minimum covariance determinant, robust Gaussian outlier
    detection [1]_.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Rousseeuw, P.J. and Van Driessen, K. A Fast Algorithm for the
       Minimum Covariance Determinant Estimator. Technometrics 1999.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, contamination=0.1):
        self.contamination = contamination
        super().__init__(MCD(contamination=contamination))


class PyODOCSVMDetector(_PyODPointsMixin, PyODDetector):
    """One-Class SVM outlier detection with an RBF kernel [1]_.

    Parameters
    ----------
    nu : float, default=0.5
        Upper bound on the fraction of training errors.
    contamination : float, default=0.1
        Expected fraction of outliers, sets the decision threshold.

    References
    ----------
    .. [1] Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. and
       Williamson, R.C. Estimating the Support of a High-Dimensional
       Distribution. Neural Computation 2001.
    """

    _tags = {"authors": ["xiezhe", "pyod"], "maintainers": ["xiezhe"]}

    def __init__(self, nu=0.5, contamination=0.1):
        self.nu = nu
        self.contamination = contamination
        super().__init__(OCSVM(nu=nu, contamination=contamination))
