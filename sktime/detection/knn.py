# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""KNN anomaly detector backed by PyOD."""

from sktime.detection.adapters._pyod import PyODDetector
from sktime.utils.dependencies import _check_soft_dependencies

__all__ = ["KNN"]
__author__ = ["xiezhe"]


class KNN(PyODDetector):
    """k-Nearest-Neighbors distance based outlier detection [1]_.

    This detector wraps ``pyod.models.knn.KNN`` from the PyOD library.
    Inputs are observations, not sliding windows; preprocessing is external.

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
        _check_soft_dependencies("pyod")
        from pyod.models.knn import KNN as _KNN

        self.n_neighbors = n_neighbors
        self.contamination = contamination
        super().__init__(_KNN(n_neighbors=n_neighbors, contamination=contamination))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"n_neighbors": 3}
