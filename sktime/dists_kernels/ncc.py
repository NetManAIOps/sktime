"""NCC kernel for time series."""

__author__ = ["GitHub Copilot"]

from sktime.distances import pairwise_distance
from sktime.dists_kernels.base import BasePairwiseTransformerPanel


class NCCKernel(BasePairwiseTransformerPanel):
    """Normalized cross-correlation (NCC) kernel for equal-length time series.

    The kernel value is defined as ``max(NCC)`` across all lags, i.e.,
    ``1 - SBD`` where SBD is shape-based distance.
    """

    _tags = {
        "authors": [""],
        "symmetric": True,
        "X_inner_mtype": "numpy3D",
        "capability:unequal_length": False,
        "pwtrafo_type": "kernel",
    }

    def _transform(self, X, X2=None):
        """Compute NCC kernel matrix."""
        distmat = pairwise_distance(X, X2, metric="sbd")
        return 1.0 - distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{}]

    @staticmethod
    def ncc_profile(x, y):
        """Compute full normalized cross-correlation (NCC) profile.

        Parameters
        ----------
        x (1d or 2d array)
            First time series.
        y: np.ndarray (1d or 2d array)
            Second time series.

        Returns
        -------
        np.ndarray
            NCC profile across lags from ``-(m-1)`` to ``+(m-1)`` where ``m`` is
            the time series length. For multivariate input, profile is averaged
            across dimensions.
        """
        from sktime.distances._numba_utils import to_numba_timeseries
        from sktime.distances._sbd_numba import _numba_ncc_profile

        _x = to_numba_timeseries(x)
        _y = to_numba_timeseries(y)

        if _x.shape[0] != _y.shape[0]:
            raise ValueError(
                "NCC requires the same number of dimensions in x and y, "
                f"but found {_x.shape[0]} and {_y.shape[0]}."
            )

        if _x.shape[1] != _y.shape[1]:
            raise ValueError(
                "NCC requires equal length time series, "
                f"but found lengths {_x.shape[1]} and {_y.shape[1]}."
            )

        return _numba_ncc_profile(_x, _y)
