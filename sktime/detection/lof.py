# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""LOF anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["LOF"]
__author__ = ["xiezhe"]


class LOF(PyODDetector):
    """Local Outlier Factor: density-deviation based outlier detection [1]_.

    This detector wraps ``pyod.models.lof.LOF`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.lof import LOF as _LOF

        self.n_neighbors = n_neighbors
        self.contamination = contamination
        super().__init__(_LOF(n_neighbors=n_neighbors, contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"n_neighbors": 3}
