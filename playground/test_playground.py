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

    def test_regression_runner(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "regression",
                "dataset_id": "covid-3month",
                "algorithm_id": "summary-random-forest-regressor",
                "params": {"n_estimators": 5, "random_state": 7},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("MAE", result["metrics"])
        self.assertIn("R²", result["metrics"])
        self.assertIn("code", result)

    def test_clustering_runner(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "clustering",
                "dataset_id": "unit-test-cl",
                "algorithm_id": "ts-kmeans",
                "params": {"n_clusters": 2, "random_state": 7},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("ARI", result["metrics"])
        self.assertEqual(result["metrics"]["Clusters"], 2)
        self.assertIn("code", result)

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

    def test_tslib_forecaster_runs(self):
        # TSLib adapter (sktime.forecasting.tslib) exercises the generic
        # runner with a torch-backed deep forecaster; skipped without torch.
        try:
            result = run_experiment(
                {
                    "task": "forecasting",
                    "dataset_id": "airline",
                    "algorithm_id": "registered-forecasting-DLinearForecaster",
                    "params": {
                        "horizon": 6,
                        "seq_len": 24,
                        "num_epochs": 2,
                        "batch_size": 16,
                    },
                }
            )
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["status"], "ok")
        self.assertIn("MAE", result["metrics"])
        self.assertIn("code", result)

    def test_pyod_detector_runs(self):
        # PyOD wrapper (sktime.detection.adapters.pyod) exercises the generic
        # anomaly runner; guards the sparse-ilocs contract of _predict.
        try:
            result = run_experiment(
                {
                    "task": "anomaly_detection",
                    "dataset_id": "yahoo",
                    "algorithm_id": "registered-anomaly_detection-PyODECODDetector",
                    "params": {"contamination": 0.01},
                }
            )
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["status"], "ok")
        self.assertIn("F1", result["metrics"])
        self.assertGreater(result["metrics"]["Detected"], 1)
        self.assertGreater(result["metrics"]["F1"], 0)


class UserPluginTests(unittest.TestCase):
    def test_user_plugins_discovered(self):
        from user_algos import discover_user_algorithms

        entries = discover_user_algorithms()
        ids = {entry["id"] for entry in entries}
        self.assertIn("user-ecod_direct", ids)
        for entry in entries:
            if entry["id"] == "user-ecod_direct":
                self.assertTrue(entry["enabled"])
                self.assertEqual(entry["task"], "anomaly_detection")

    def test_broken_plugin_only_disables_itself(self):
        from user_algos import EXPERIMENTS_DIR, discover_user_algorithms

        broken = EXPERIMENTS_DIR / "_broken_tmp.py"
        broken.write_text("import definitely_not_a_real_module_xyz\n", encoding="utf-8")
        try:
            entries = discover_user_algorithms()
            by_id = {entry["id"]: entry for entry in entries}
            self.assertIn("user-_broken_tmp", by_id)
            self.assertFalse(by_id["user-_broken_tmp"]["enabled"])
            self.assertIn("disabled_reason", by_id["user-_broken_tmp"])
            # other plugins are unaffected
            self.assertTrue(by_id["user-ecod_direct"]["enabled"])
        finally:
            broken.unlink(missing_ok=True)

    def test_user_plugin_runs(self):
        try:
            result = run_experiment(
                {
                    "task": "anomaly_detection",
                    "dataset_id": "yahoo",
                    "algorithm_id": "user-ecod_direct",
                    "params": {"contamination": 0.01},
                }
            )
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["metrics"]["Detected"], 1)
        self.assertIn("code", result)


if __name__ == "__main__":
    unittest.main()
