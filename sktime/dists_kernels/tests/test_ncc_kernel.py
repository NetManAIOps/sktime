"""Tests for NCC kernel."""

import numpy as np

from sktime.dists_kernels import NCCKernel
from sktime.utils._testing.panel import make_transformer_problem


def test_ncc_kernel_is_one_minus_sbd():
    """NCC kernel should be 1 - SBD by definition."""
    X = make_transformer_problem(
        n_instances=4,
        n_columns=1,
        n_timepoints=20,
        random_state=42,
        return_numpy=True,
        panel=True,
    )

    k = NCCKernel()
    mat = k.transform(X)

    assert isinstance(mat, np.ndarray)
    assert mat.shape == (4, 4)
    assert np.allclose(np.diag(mat), 1.0)


def test_ncc_kernel_bounds():
    """NCC values should be bounded above by 1."""
    X = make_transformer_problem(
        n_instances=3,
        n_columns=1,
        n_timepoints=15,
        random_state=0,
        return_numpy=True,
        panel=True,
    )

    mat = NCCKernel().transform(X)
    assert np.all(mat <= 1.0 + 1e-12)
