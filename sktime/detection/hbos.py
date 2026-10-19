# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""HBOS anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["HBOS"]
__author__ = ["xiezhe"]


class HBOS(PyODDetector):
    """HBOS: histogram-based outlier score. Fast, linear-time [1]_.

    This detector wraps ``pyod.models.hbos.HBOS`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.hbos import HBOS as _HBOS

        self.n_bins = n_bins
        self.contamination = contamination
        super().__init__(_HBOS(n_bins=n_bins, contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"n_bins": 4}
