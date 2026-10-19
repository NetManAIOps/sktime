# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""OCSVM anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["OCSVM"]
__author__ = ["xiezhe"]


class OCSVM(PyODDetector):
    """One-Class SVM outlier detection with an RBF kernel [1]_.

    This detector wraps ``pyod.models.ocsvm.OCSVM`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.ocsvm import OCSVM as _OCSVM

        self.nu = nu
        self.contamination = contamination
        super().__init__(_OCSVM(nu=nu, contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"nu": 0.25}
