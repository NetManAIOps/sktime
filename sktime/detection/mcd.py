# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""MCD anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["MCD"]
__author__ = ["xiezhe"]


class MCD(PyODDetector):
    """MCD: minimum covariance determinant, robust Gaussian outlier
    detection [1]_.

    This detector wraps ``pyod.models.mcd.MCD`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.mcd import MCD as _MCD

        self.contamination = contamination
        super().__init__(_MCD(contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {}
