"""Tests for the detector base class."""

import unittest

import numpy as np

from sktime.detection.naive import ThresholdDetector


class TestBaseDetector(unittest.TestCase):
    def test_fit_transform_numpy(self):
        """NumPy inputs work with the public fit/transform interface (#8325)."""
        data = np.array([0, 0.5, 2, 0.1, 0, 0, 0, 2, 0, 0, 0.3, -1, 0, 2, 0.2])
        model = ThresholdDetector(upper=1.5, mode="points")
        prediction = model.fit_transform(data)

        self.assertEqual(prediction.shape, (len(data), 1))
        expected = np.zeros(len(data), dtype=int)
        expected[[2, 7, 13]] = 1
        np.testing.assert_array_equal(prediction["labels"], expected)

        model = ThresholdDetector(upper=1.5, mode="points").fit(data)
        np.testing.assert_array_equal(model.transform(data), prediction)
