"""Tests for SBD distance."""

import numpy as np

from sktime.distances import sbd_distance


def test_sbd_distance_matches_ncc_definition():
    """SBD should equal 1 - max normalized cross-correlation."""
    y1 = np.array([1.0, 2.0, 3.0, 2.0])
    y2 = np.array([0.0, 1.0, 2.0, 3.0])

    cc = np.correlate(y1, y2, mode="full")
    ac1 = np.correlate(y1, y1, mode="full")
    ac2 = np.correlate(y2, y2, mode="full")
    ncc = cc / np.sqrt(ac1[len(ac1) // 2] * ac2[len(ac2) // 2])

    expected = 1.0 - np.max(ncc)
    actual = sbd_distance(y1, y2)

    assert np.isclose(actual, expected)


def test_sbd_distance_identical_is_zero():
    """SBD should be zero on identical series."""
    y = np.array([1.0, -2.0, 3.0, -4.0])
    assert np.isclose(sbd_distance(y, y), 0.0)
