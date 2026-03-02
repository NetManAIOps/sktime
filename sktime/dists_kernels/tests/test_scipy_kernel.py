"""Tests for scipy kernel interface."""

import numpy as np
import pytest

from sktime.dists_kernels.scipy_kernel import ScipyKernel
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_transformer_problem


@pytest.fixture
def X1():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=1,
        return_numpy=True,
        panel=False,
    )


@pytest.fixture
def X2():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=2,
        return_numpy=True,
        panel=False,
    )


@pytest.fixture
def X1_df():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=1,
        return_numpy=False,
        panel=False,
    )


@pytest.fixture
def X2_df():
    return make_transformer_problem(
        n_instances=5,
        n_columns=5,
        n_timepoints=5,
        random_state=2,
        return_numpy=False,
        panel=False,
    )


METRIC_VALUES = ["pearson", "kendalltau", "spearman"]
COLALIGN_VALUES = ["intersect", "force-align", "none"]


@pytest.mark.skipif(
    not run_test_for_class(ScipyKernel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_scipykernel(X1, X2, X1_df, X2_df):
    """Test runner for numpy and dataframe tests."""
    _run_scipy_kernel_test(X1, X2)
    _run_scipy_kernel_test(X1_df, X2_df)


def _run_scipy_kernel_test(x, y):
    default_params = ScipyKernel()
    default_transformation = default_params.transform(x, y)

    assert isinstance(default_transformation, np.ndarray)
    assert default_transformation.shape == (len(x), len(y))

    for metric in METRIC_VALUES:
        for colalign in COLALIGN_VALUES:
            metric_params = ScipyKernel(metric=metric, colalign=colalign)
            metric_params_transformation = metric_params.transform(x, y)

            assert isinstance(metric_params_transformation, np.ndarray), (
                f"Error occurred testing parameters metric={metric}, colalign={colalign}"
            )
            assert metric_params_transformation.shape == (len(x), len(y))


def test_scipykernel_diagonal_is_one_for_self_similarity(X1):
    """Diagonal should be one for rank/correlation similarities on same vectors."""
    for metric in METRIC_VALUES:
        k = ScipyKernel(metric=metric)
        mat = k.transform(X1)
        assert np.allclose(np.diag(mat), 1.0)


def test_scipykernel_callable_metric(X1, X2):
    """Callable metric should be accepted."""

    def dot_norm(x, y):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx == 0 or ny == 0:
            return 0.0
        return float(np.dot(x, y) / (nx * ny))

    k = ScipyKernel(metric=dot_norm)
    mat = k.transform(X1, X2)

    assert isinstance(mat, np.ndarray)
    assert mat.shape == (len(X1), len(X2))
