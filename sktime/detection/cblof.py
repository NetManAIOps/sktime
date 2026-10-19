# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""CBLOF anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["CBLOF"]
__author__ = ["xiezhe"]


class CBLOF(PyODDetector):
    """CBLOF: cluster-based local outlier factor [1]_.

    This detector wraps ``pyod.models.cblof.CBLOF`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.cblof import CBLOF as _CBLOF

        self.n_clusters = n_clusters
        self.contamination = contamination
        super().__init__(_CBLOF(n_clusters=n_clusters, contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"n_clusters": 2}
