"""Test functions for loose loaders."""

__author__ = ["fkiraly"]

__all__ = []


import pytest

import pandas as pd

from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_load_macroeconomic():
    """Test that load_macroeconomic runs."""
    from sktime.datasets import load_macroeconomic

    load_macroeconomic()


def test_load_causal_bnlearn_dataset():
    """Test that bundled causal benchmark dataset loaders run and return expected types."""
    from sktime.datasets import load_alarm, load_asia, load_sachs

    for loader in [load_sachs, load_alarm, load_asia]:
        X = loader()
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] > 0
        assert X.shape[1] > 0

        X_graph, edges = loader(return_true_graph=True)
        assert isinstance(X_graph, pd.DataFrame)
        assert isinstance(edges, list)
        assert len(edges) > 0
