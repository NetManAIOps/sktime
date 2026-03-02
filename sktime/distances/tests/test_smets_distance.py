"""Tests for SMETS distance."""

import numpy as np

from sktime.distances import pairwise_distance, smets_distance


def test_smets_distance_zero_for_identical_series():
    """SMETS should be zero when two MTS are identical."""
    x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert np.isclose(smets_distance(x, x), 0.0)


def test_smets_distance_default_matches_manual_simple_case():
    """SMETS with no unmatched dimensions matches manual L2 aggregation."""
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0, 0.0], [2.0, 2.0]])

    # Pairwise euclidean distances by dimension:
    # [[0.0, sqrt(8)], [sqrt(2), sqrt(2)]]
    # Greedy picks 0.0 then sqrt(2), with no unmatched penalties.
    expected = np.sqrt(2.0)
    actual = smets_distance(x, y, p_norm=2)

    assert np.isclose(actual, expected)


def test_smets_distance_supports_linear_alias_and_metric_switch():
    """SMETS should accept 'linear' alias and different uts_metric names."""
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0, 0.0], [2.0, 2.0]])

    d_euclid = smets_distance(x, y, uts_metric="euclidean")
    d_linear = smets_distance(x, y, uts_metric="linear")
    d_squared = smets_distance(x, y, uts_metric="squared")

    assert np.isclose(d_euclid, d_linear)
    assert d_squared >= d_euclid


def test_smets_pairwise_distance_runs():
    """SMETS should be usable through pairwise_distance with metric string."""
    X = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [2.0, 2.0]],
            [[1.0, 1.0], [3.0, 3.0]],
        ]
    )

    mat = pairwise_distance(X, metric="smets")

    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 0.0)
