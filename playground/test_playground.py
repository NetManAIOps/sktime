"""Lightweight tests for the local TSBox Sandbox Playground."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from catalog import build_catalog
from runners import PlaygroundError, run_experiment


class CatalogTests(unittest.TestCase):
    def test_catalog_shape(self):
        catalog = build_catalog(include_registered=False)
        self.assertIn("tasks", catalog)
        self.assertIn("algorithms", catalog)
        self.assertIn("datasets", catalog)
        self.assertIn("compatibility", catalog)
        self.assertTrue(catalog["compatibility"])

    def test_all_enabled_algorithms_have_dataset(self):
        catalog = build_catalog(include_registered=False)
        enabled = [item for item in catalog["algorithms"] if item["enabled"]]
        pairs = {(row["algorithm_id"], row["dataset_id"]) for row in catalog["compatibility"]}
        for algorithm in enabled:
            self.assertTrue(any(pair[0] == algorithm["id"] for pair in pairs))


class RunnerTests(unittest.TestCase):
    def _run_or_skip_missing_deps(self, spec):
        try:
            return run_experiment(spec)
        except PlaygroundError as exc:
            if "Missing dependency" in str(exc):
                self.skipTest(str(exc))
            raise

    def test_forecasting_runner(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "naive-seasonal-last",
                "params": {"horizon": 6, "seasonal_period": 12},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("MAE", result["metrics"])
        self.assertIn("code", result)

    def test_classification_runner(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "classification",
                "dataset_id": "unit-test",
                "algorithm_id": "summary-random-forest",
                "params": {"n_estimators": 5, "random_state": 7},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("Accuracy", result["metrics"])

    def test_anomaly_runner(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "anomaly_detection",
                "dataset_id": "yahoo",
                "algorithm_id": "threshold-detector",
                "params": {"threshold": 2.0, "window": 24},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("F1", result["metrics"])
        # Guard against the regression where the threshold was applied to the
        # raw, strongly trending series and never matched a ground-truth label.
        self.assertGreater(result["metrics"]["F1"], 0)
        self.assertGreater(result["metrics"]["Detected"], 0)

    def test_discovered_forecaster_runs(self):
        # PolynomialTrendForecaster is discovered via the registry (no soft
        # deps) and exercises the generic, non-curated runner path.
        try:
            result = run_experiment(
                {
                    "task": "forecasting",
                    "dataset_id": "airline",
                    "algorithm_id": "registered-forecasting-PolynomialTrendForecaster",
                    "params": {"horizon": 6, "degree": 2},
                }
            )
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["status"], "ok")
        self.assertIn("MAE", result["metrics"])
        self.assertIn("code", result)


if __name__ == "__main__":
    unittest.main()
