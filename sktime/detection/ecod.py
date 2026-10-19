# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ECOD anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["ECOD"]
__author__ = ["xiezhe"]


class ECOD(PyODDetector):
    """ECOD: unsupervised outlier detection using empirical cumulative
    distribution functions. Parameter-free and fast [1]_.

    This detector wraps ``pyod.models.ecod.ECOD`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.ecod import ECOD as _ECOD

        self.contamination = contamination
        super().__init__(_ECOD(contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {}
