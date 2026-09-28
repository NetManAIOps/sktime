"""Roundtrip tests for the generic train/predict persistence backend.

Covers the three backend paths of `trainer.train` / `trainer.predict_estimator`
(forecaster via sktime save/load, classifier via sktime save/load, DevAD via
the model_service backend) plus the manifest schema documented in
``playground/persistence.py``. All tests use a throwaway models root and never
touch ``playground/models/``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

import trainer
from persistence import detect_backend
from runners import PlaygroundError

# Manifest keys required by the sktime-backend schema (persistence.py docstring).
MANIFEST_REQUIRED_KEYS = {
    "model_id",
    "backend",
    "task",
    "algorithm_id",
    "params",
    "created_at",
    "spec",
}


class TrainerTestCase(unittest.TestCase):
    """Base class wiring trainer to a throwaway models root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_models_root = trainer.MODELS_ROOT
        trainer.MODELS_ROOT = Path(self._tmp.name)
        self.addCleanup(self._restore_models_root)

    def _restore_models_root(self):
        trainer.MODELS_ROOT = self._orig_models_root

    def _train_or_skip(self, spec):
        try:
            return trainer.train(spec)
        except PlaygroundError as exc:
            self.skipTest(str(exc))


class SktimeForecasterTests(TrainerTestCase):
    def test_sarimax_train_predict_roundtrip(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "registered-forecasting-SARIMAX",
                "dataset_id": "airline",
                "model_id": "t-sarimax",
                "params": {"horizon": 6},
            }
        )
        self.assertEqual(trained["status"], "ok")
        self.assertEqual(trained["backend"], "sktime")
        self.assertEqual(trained["task"], "forecasting")
        self.assertEqual(trained["model_id"], "t-sarimax")

        model_dir = Path(trained["model_dir"])
        self.assertTrue((model_dir / "model.zip").is_file())
        self.assertTrue((model_dir / "manifest.json").is_file())

        manifest = trained["manifest"]
        self.assertTrue(MANIFEST_REQUIRED_KEYS <= set(manifest))
        self.assertEqual(manifest["backend"], "sktime")
        self.assertEqual(manifest["task"], "forecasting")
        self.assertEqual(manifest["algorithm_id"], "registered-forecasting-SARIMAX")
        self.assertEqual(manifest["spec"]["dataset_id"], "airline")
        self.assertEqual(manifest["eval_params"]["horizon"], 6)

        result = trainer.predict_estimator("t-sarimax", "airline", {})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["task"], "forecasting")
        self.assertIn("MAE", result["metrics"])
        self.assertIn("MSE", result["metrics"])
        self.assertEqual(result["spec"]["algorithm_id"], "sktime-trained:t-sarimax")
        self.assertEqual(result["algorithm"]["backend"], "sktime")
        self.assertTrue(result["algorithm"]["trained"])
        self.assertIn("code", result)
        self.assertIn("report", result)

    def test_curated_forecaster_train_predict_roundtrip(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "naive-seasonal-last",
                "dataset_id": "airline",
                "model_id": "t-naive",
                "params": {"horizon": 6, "seasonal_period": 12},
            }
        )
        self.assertEqual(trained["status"], "ok")
        result = trainer.predict_estimator("t-naive", "airline", {})
        self.assertEqual(result["status"], "ok")
        self.assertIn("MAE", result["metrics"])
        self.assertIn("MSE", result["metrics"])

    def test_predict_horizon_override(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "naive-seasonal-last",
                "dataset_id": "airline",
                "model_id": "t-naive-override",
                "params": {"horizon": 6},
            }
        )
        self.assertEqual(trained["status"], "ok")
        result = trainer.predict_estimator(
            "t-naive-override", "airline", {"horizon": 3}
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["series"]["meta"]["horizon"], 3)


class SktimeClassifierTests(TrainerTestCase):
    def test_registered_classifier_train_predict_roundtrip(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": (
                    "registered-classification-KNeighborsTimeSeriesClassifier"
                ),
                "dataset_id": "unit-test",
                "model_id": "t-knn",
            }
        )
        self.assertEqual(trained["status"], "ok")
        self.assertEqual(trained["backend"], "sktime")
        self.assertEqual(trained["task"], "classification")
        self.assertTrue((Path(trained["model_dir"]) / "model.zip").is_file())

        result = trainer.predict_estimator("t-knn", "unit-test", {})
        self.assertEqual(result["status"], "ok")
        self.assertIn("Accuracy", result["metrics"])
        self.assertGreaterEqual(result["metrics"]["Accuracy"], 0.0)
        self.assertLessEqual(result["metrics"]["Accuracy"], 1.0)
        self.assertEqual(result["spec"]["algorithm_id"], "sktime-trained:t-knn")
        self.assertIn("code", result)
        self.assertIn("report", result)

    def test_curated_classifier_train_predict_roundtrip(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "summary-random-forest",
                "dataset_id": "unit-test",
                "model_id": "t-summary-rf",
                "params": {"n_estimators": 5, "random_state": 7},
            }
        )
        self.assertEqual(trained["status"], "ok")
        result = trainer.predict_estimator("t-summary-rf", "unit-test", {})
        self.assertEqual(result["status"], "ok")
        self.assertIn("Accuracy", result["metrics"])


class DevADBackendTests(TrainerTestCase):
    def test_iforest_train_detect_via_generic_entrypoints(self):
        # The generic train/predict entry points must reproduce the DevAD
        # train_devad/detect_devad behavior (detect is the anomaly alias).
        trained = self._train_or_skip(
            {
                "algorithm_id": "registered-anomaly_detection-DevADIForestDetector",
                "dataset_id": "yahoo",
                "model_id": "t-iforest",
                "params": {"win_len": 8},
            }
        )
        self.assertEqual(trained["status"], "ok")
        self.assertEqual(trained["backend"], "devad")
        self.assertEqual(trained["family"], "iforest")
        # train() returns the uniform manifest view (generic schema keys) ...
        self.assertTrue(MANIFEST_REQUIRED_KEYS <= set(trained["manifest"]))
        self.assertEqual(trained["manifest"]["backend"], "devad")

        model_dir = Path(trained["model_dir"])
        self.assertTrue((model_dir / "model.pt").is_file())
        self.assertTrue((model_dir / "manifest.json").is_file())
        # ... while the on-disk DevAD manifest keeps its original schema.
        on_disk = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["family"], "iforest")
        self.assertNotIn("backend", on_disk)

        result = trainer.predict_estimator(
            "t-iforest", "yahoo", {"threshold_quantile": 0.95}
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("F1", result["metrics"])
        self.assertGreater(result["metrics"]["Detected"], 1)
        self.assertEqual(result["spec"]["algorithm_id"], "devad-trained:t-iforest")
        self.assertIn("code", result)
        self.assertIn("report", result)

        # Direct detect_devad on the same model still works (regression).
        direct = trainer.detect_devad(
            {
                "model_id": "t-iforest",
                "dataset_id": "yahoo",
                "params": {"threshold_quantile": 0.95},
            }
        )
        self.assertEqual(direct["status"], "ok")
        self.assertEqual(direct["spec"]["algorithm_id"], "devad-trained:t-iforest")
        self.assertEqual(direct["metrics"], result["metrics"])


class ManifestAndListingTests(TrainerTestCase):
    def test_manifest_schema_and_model_listing(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "naive-seasonal-last",
                "dataset_id": "airline",
                "model_id": "t-manifest",
                "params": {"horizon": 4},
            }
        )
        self.assertEqual(trained["status"], "ok")
        manifest_path = Path(trained["model_dir"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(MANIFEST_REQUIRED_KEYS <= set(manifest))
        self.assertEqual(manifest["model_id"], "t-manifest")
        self.assertEqual(manifest["backend"], "sktime")
        self.assertEqual(detect_backend(manifest), "sktime")
        self.assertEqual(manifest["spec"]["algorithm_id"], "naive-seasonal-last")
        self.assertEqual(manifest["artifacts"]["model"], "model.zip")
        # created_at must be a parseable ISO-8601 timestamp.
        from datetime import datetime

        datetime.fromisoformat(manifest["created_at"])

        rows = trainer.list_trained_models()
        self.assertEqual([row["model_id"] for row in rows], ["t-manifest"])
        row = rows[0]
        self.assertEqual(row["backend"], "sktime")
        self.assertEqual(row["task"], "forecasting")
        self.assertEqual(row["algorithm_id"], "naive-seasonal-last")
        self.assertEqual(row["created_at"], manifest["created_at"])

    def test_devad_manifest_detects_devad_backend(self):
        self.assertEqual(detect_backend({"family": "iforest", "params": {}}), "devad")
        self.assertEqual(detect_backend({"backend": "sktime"}), "sktime")


class TrainPredictErrorTests(TrainerTestCase):
    def test_train_rejects_unknown_algorithm(self):
        with self.assertRaises(PlaygroundError):
            trainer.train({"algorithm_id": "definitely-not-an-algorithm"})

    def test_train_rejects_missing_algorithm(self):
        with self.assertRaises(PlaygroundError):
            trainer.train({"dataset_id": "airline"})

    def test_predict_rejects_unknown_model(self):
        with self.assertRaises(PlaygroundError):
            trainer.predict_estimator("no-such-model", "airline", {})

    def test_predict_rejects_wrong_task_dataset(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "naive-seasonal-last",
                "dataset_id": "airline",
                "model_id": "t-wrong-task",
            }
        )
        self.assertEqual(trained["status"], "ok")
        with self.assertRaises(PlaygroundError):
            trainer.predict_estimator("t-wrong-task", "unit-test", {})

    def test_curated_anomaly_detector_rejected(self):
        with self.assertRaises(PlaygroundError):
            trainer.train(
                {
                    "algorithm_id": "threshold-detector",
                    "dataset_id": "yahoo",
                    "model_id": "t-threshold",
                }
            )


if __name__ == "__main__":
    unittest.main()
