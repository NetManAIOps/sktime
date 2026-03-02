"""Tests for MovingAverageTransformer."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.moving_average import MovingAverageTransformer


def _k_ma(sequence, k):
    """Original k_ma from SFbasedSmoothing notebook for regression test."""
    return np.convolve(sequence, np.ones(k) / k, mode="valid")


@pytest.mark.skipif(
    not run_test_for_class(MovingAverageTransformer),
    reason="run test only if MovingAverageTransformer is available",
)
def test_moving_average_matches_k_ma():
    """Verify MovingAverageTransformer matches original k_ma behavior."""
    np.random.seed(42)
    series = np.random.randn(128) + np.sin(np.linspace(0, 10, 128))
    y = pd.Series(series)

    transformer = MovingAverageTransformer(window_length=7, mode="valid")
    result = transformer.fit_transform(y)

    expected = _k_ma(series, 7)
    np.testing.assert_array_almost_equal(result.values, expected)


@pytest.mark.skipif(
    not run_test_for_class(MovingAverageTransformer),
    reason="run test only if MovingAverageTransformer is available",
)
def test_moving_average_dataframe():
    """Test MovingAverageTransformer on DataFrame (multivariate)."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "a": np.random.randn(50),
            "b": np.random.randn(50) + 1,
        }
    )
    transformer = MovingAverageTransformer(window_length=5, mode="valid")
    result = transformer.fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 50 - 5 + 1
    assert list(result.columns) == ["a", "b"]
    np.testing.assert_array_almost_equal(
        result["a"].values, _k_ma(df["a"].values, 5)
    )
    np.testing.assert_array_almost_equal(
        result["b"].values, _k_ma(df["b"].values, 5)
    )


@pytest.mark.skipif(
    not run_test_for_class(MovingAverageTransformer),
    reason="run test only if MovingAverageTransformer is available",
)
def test_moving_average_mode_same():
    """Test mode='same' preserves length."""
    y = pd.Series(np.arange(20))
    transformer = MovingAverageTransformer(window_length=5, mode="same")
    result = transformer.fit_transform(y)
    assert len(result) == len(y)


@pytest.mark.skipif(
    not run_test_for_class(MovingAverageTransformer),
    reason="run test only if MovingAverageTransformer is available",
)
def test_moving_average_invalid_params():
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="window_length must be >= 2"):
        MovingAverageTransformer(window_length=1)
    with pytest.raises(ValueError, match="mode must be one of"):
        MovingAverageTransformer(window_length=5, mode="invalid")
