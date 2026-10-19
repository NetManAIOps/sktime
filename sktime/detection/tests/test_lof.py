"""Regression tests for pointwise PyOD LOF and sliding-window SubLOF."""

import unittest

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from sktime.detection.lof import LOF
from sktime.detection.sub_lof import SubLOF
from sktime.utils.dependencies import _check_soft_dependencies


@unittest.skipUnless(
    _check_soft_dependencies("pyod", severity="none"), "PyOD is not installed"
)
class TestLOF(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(np.linspace(-1, 1, 40))
        self.test = pd.DataFrame([0.0, 0.1, 8.0, -0.1, -8.0, 0.2])

    def test_scores_and_positions_match_pyod(self):
        """Expose full scores, sparse positions and dense labels from PyOD."""
        from pyod.models.lof import LOF as PyODLOF

        model = LOF(n_neighbors=3).fit(self.train)
        reference = PyODLOF(n_neighbors=3).fit(self.train.to_numpy())
        scores = model.transform_scores(self.test)
        np.testing.assert_array_equal(
            scores, reference.decision_function(self.test.to_numpy())
        )
        self.assertEqual(scores.shape, (len(self.test),))
        self.assertEqual(scores.dtype, np.float64)
        labels = reference.predict(self.test.to_numpy())
        np.testing.assert_array_equal(model.predict(self.test)["ilocs"], [2, 4])
        np.testing.assert_array_equal(
            model.predict(self.test)["ilocs"], np.flatnonzero(labels)
        )
        np.testing.assert_array_equal(model.transform(self.test)["labels"], labels)

    def test_inputs_are_unchanged(self):
        train_before, test_before = self.train.copy(), self.test.copy()
        model = LOF(n_neighbors=3).fit(self.train)
        model.predict(self.test)
        model.transform_scores(self.test)
        pd.testing.assert_frame_equal(self.train, train_before)
        pd.testing.assert_frame_equal(self.test, test_before)

    def test_clone_and_save_load(self):
        model = LOF(n_neighbors=3).fit(self.train)
        self.assertEqual(model.clone().get_params(), model.get_params())
        loaded = LOF.load_from_serial(model.save()[1])
        np.testing.assert_array_equal(
            loaded.transform_scores(self.test), model.transform_scores(self.test)
        )
        pd.testing.assert_frame_equal(loaded.predict(self.test), model.predict(self.test))


class TestSubLOF(unittest.TestCase):
    def setUp(self):
        self.train = np.sin(np.linspace(0, 12, 80))
        self.test = np.cos(np.linspace(0, 3, 24))
        self.test[12] = 8.0

    def test_window_scores_match_sklearn(self):
        """Score stride-one windows at their last point; pad the warm-up only."""
        win_len = 4
        model = SubLOF(win_len=win_len, n_neighbors=3, n_jobs=1).fit(self.train)
        reference = LocalOutlierFactor(n_neighbors=3, n_jobs=1, novelty=True)
        reference.fit(np.lib.stride_tricks.sliding_window_view(self.train, win_len))
        expected = -reference.decision_function(
            np.lib.stride_tricks.sliding_window_view(self.test, win_len)
        )
        scores = model.transform_scores(self.test)
        np.testing.assert_array_equal(scores[: win_len - 1], 0)
        np.testing.assert_array_equal(scores[win_len - 1 :], expected)
        self.assertEqual(scores.shape, self.test.shape)
        self.assertEqual(scores.dtype, np.float64)

    def test_inputs_are_unchanged(self):
        train, test = pd.DataFrame(self.train), pd.DataFrame(self.test)
        train_before, test_before = train.copy(), test.copy()
        model = SubLOF(win_len=4, n_neighbors=3, n_jobs=1).fit(train)
        model.transform_scores(test)
        pd.testing.assert_frame_equal(train, train_before)
        pd.testing.assert_frame_equal(test, test_before)

    def test_predict_has_no_implicit_threshold(self):
        model = SubLOF(win_len=4, n_neighbors=3, n_jobs=1).fit(self.train)
        with self.assertRaisesRegex(NotImplementedError, "transform_scores"):
            model.predict(self.test)

    def test_clone_and_save_load(self):
        model = SubLOF(win_len=4, n_neighbors=3, n_jobs=1).fit(self.train)
        self.assertEqual(model.clone().get_params(), model.get_params())
        loaded = SubLOF.load_from_serial(model.save()[1])
        np.testing.assert_array_equal(
            loaded.transform_scores(self.test), model.transform_scores(self.test)
        )
