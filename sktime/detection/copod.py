# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""COPOD anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["COPOD"]
__author__ = ["xiezhe"]


class COPOD(PyODDetector):
    """COPOD: copula-based outlier detection. Parameter-free [1]_.

    This detector wraps ``pyod.models.copod.COPOD`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.copod import COPOD as _COPOD

        self.contamination = contamination
        super().__init__(_COPOD(contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {}
