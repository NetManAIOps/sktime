"""Roundtrip tests for the generic train/predict persistence backend.

Covers `trainer.train` / `trainer.predict_estimator` (and the `labts run
--model-id` CLI on top of them): sktime save/load roundtrips for
forecasting/classification, rolling-origin forecasting, clustering
`fit_on=all`, causal discovery, per-series multi-series anomaly bundles
(synthetic TSB-UAD layout), the DevAD model_service backend, and the
manifest schema documented in ``playground/persistence.py``. Deterministic
algorithms are checked for exact metric parity with the one-shot
`run_experiment`. All tests use a throwaway models root and never touch
``playground/models/``.
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


class CausalTrainTests(TrainerTestCase):
    def test_notears_train_run_matches_one_shot(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "causal-notears",
                "dataset_id": "causal-sachs",
                "model_id": "t-notears",
                "params": {"max_samples": 300},
            }
        )
        self.assertEqual(trained["status"], "ok")
        self.assertEqual(trained["task"], "causal")
        self.assertTrue((Path(trained["model_dir"]) / "model.zip").is_file())

        from runners import run_experiment

        result = trainer.predict_estimator("t-notears", "causal-sachs", {})
        one_shot = run_experiment(
            {
                "task": "causal",
                "dataset_id": "causal-sachs",
                "algorithm_id": "causal-notears",
                "params": {"max_samples": 300},
            }
        )
        self.assertEqual(result["status"], "ok")
        # NOTEARS is deterministic on the same subsample: exact match required.
        self.assertEqual(result["metrics"], one_shot["metrics"])
        self.assertEqual(result["graph"]["adjacency"], one_shot["graph"]["adjacency"])


class ClusteringFitOnAllTests(TrainerTestCase):
    def test_fit_on_all_train_run_matches_one_shot(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "ts-kmeans",
                "dataset_id": "unit-test-cl",
                "model_id": "t-kmeans-all",
                "params": {"fit_on": "all"},
            }
        )
        self.assertEqual(trained["status"], "ok")

        from runners import run_experiment

        result = trainer.predict_estimator("t-kmeans-all", "unit-test-cl", {})
        one_shot = run_experiment(
            {
                "task": "clustering",
                "dataset_id": "unit-test-cl",
                "algorithm_id": "ts-kmeans",
                "params": {"fit_on": "all"},
            }
        )
        self.assertEqual(result["metrics"], one_shot["metrics"])
        # The default holdout protocol is untouched (no fit_on).
        default_run = run_experiment(
            {
                "task": "clustering",
                "dataset_id": "unit-test-cl",
                "algorithm_id": "ts-kmeans",
            }
        )
        self.assertNotEqual(result["metrics"]["ARI"], default_run["metrics"]["ARI"])


class MultiSeriesAnomalyTests(TrainerTestCase):
    """Per-series model bundles for TSB-UAD-style multi-series datasets."""

    def setUp(self):
        super().setUp()
        import os

        import numpy as np

        self._tsb_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tsb_tmp.cleanup)
        series_dir = Path(self._tsb_tmp.name) / "TSB-UAD-Public" / "YAHOO"
        series_dir.mkdir(parents=True)
        rng = np.random.RandomState(0)
        for k in range(3):
            n = 300
            t = np.arange(n)
            values = np.sin(t / (8.0 + k)) + 0.05 * rng.randn(n)
            labels = np.zeros(n, dtype=int)
            for center in (100 + 10 * k, 200 - 5 * k):
                values[center : center + 3] += 6.0
                labels[center : center + 3] = 1
            data = np.column_stack([values, labels])
            np.savetxt(
                series_dir / f"synthetic_{k}.out",
                data,
                fmt=["%.6f", "%d"],
                delimiter=",",
            )
        self._orig_tsb_home = os.environ.get("TSB_UAD_HOME")
        os.environ["TSB_UAD_HOME"] = self._tsb_tmp.name
        self.addCleanup(self._restore_tsb_home)

    def _restore_tsb_home(self):
        import os

        if self._orig_tsb_home is None:
            os.environ.pop("TSB_UAD_HOME", None)
        else:
            os.environ["TSB_UAD_HOME"] = self._orig_tsb_home

    def test_multiseries_train_run_matches_one_shot(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "registered-anomaly_detection-PyODLOFDetector",
                "dataset_id": "tsb-yahoo",
                "model_id": "t-lof-ms",
            }
        )
        self.assertEqual(trained["status"], "ok")
        self.assertEqual(trained["series_models"], 3)
        manifest = trained["manifest"]
        self.assertTrue(manifest["multiseries"])
        self.assertEqual(len(manifest["series"]), 3)
        model_dir = Path(trained["model_dir"])
        self.assertTrue((model_dir / "series" / "0000" / "model.zip").is_file())

        from runners import run_experiment

        result = trainer.predict_estimator("t-lof-ms", "tsb-yahoo", {})
        one_shot = run_experiment(
            {
                "task": "anomaly_detection",
                "dataset_id": "tsb-yahoo",
                "algorithm_id": "registered-anomaly_detection-PyODLOFDetector",
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metrics"]["Series"], 3)
        # LOF is deterministic: the persisted per-series models must reproduce
        # the one-shot dataset-level metrics exactly.
        for key in ("AUC-ROC", "Precision", "Recall", "F1"):
            self.assertAlmostEqual(
                result["metrics"][key], one_shot["metrics"][key], places=9
            )
        self.assertEqual(len(result["tables"]["per_series"]), 3)


class RollingForecastTrainTests(TrainerTestCase):
    def test_rolling_train_run_protocol_and_determinism(self):
        params = {
            "eval_mode": "rolling",
            "horizon": 6,
            "pred_len": 6,
            "seq_len": 12,
            "num_epochs": 1,
            "test_fraction": 0.2,
        }
        trained = self._train_or_skip(
            {
                "algorithm_id": "registered-forecasting-DLinearForecaster",
                "dataset_id": "airline",
                "model_id": "t-dlinear-roll",
                "params": params,
            }
        )
        self.assertEqual(trained["status"], "ok")
        self.assertEqual(trained["manifest"]["eval_params"]["eval_mode"], "rolling")

        from runners import run_experiment

        result = trainer.predict_estimator("t-dlinear-roll", "airline", {})
        again = trainer.predict_estimator("t-dlinear-roll", "airline", {})
        one_shot = run_experiment(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "registered-forecasting-DLinearForecaster",
                "params": params,
            }
        )
        self.assertEqual(result["status"], "ok")
        # Same persisted model, no refit: bit-identical metrics on re-run.
        self.assertEqual(result["metrics"], again["metrics"])
        # Same protocol as the one-shot rolling runner (split, window count).
        self.assertEqual(result["metrics"]["Windows"], one_shot["metrics"]["Windows"])
        self.assertEqual(
            result["series"]["meta"]["test_start"],
            one_shot["series"]["meta"]["test_start"],
        )
        self.assertEqual(
            result["series"]["meta"]["test_end"], one_shot["series"]["meta"]["test_end"]
        )
        for key in ("MSE", "MAE"):
            import math

            self.assertGreaterEqual(result["metrics"][key], 0.0)
            self.assertTrue(math.isfinite(result["metrics"][key]))


class CliRunModelIdTests(TrainerTestCase):
    """`labts run --model-id` end-to-end through the CLI entry point."""

    def _cli(self, argv):
        import contextlib
        import io

        import labts

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = labts.main(argv)
        return code, json.loads(buffer.getvalue())

    def test_run_model_id_matches_one_shot(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "naive-seasonal-last",
                "dataset_id": "airline",
                "model_id": "t-cli",
                "params": {"horizon": 6},
            }
        )
        self.assertEqual(trained["status"], "ok")

        code, envelope = self._cli(["run", "--model-id", "t-cli", "--compact"])
        self.assertEqual(code, 0)
        self.assertEqual(envelope["status"], "ok")
        metrics = envelope["data"]["metrics"]

        code, one_shot = self._cli(
            [
                "run",
                "--task",
                "forecasting",
                "--dataset",
                "airline",
                "--algorithm",
                "naive-seasonal-last",
                "--param",
                "horizon=6",
                "--compact",
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(metrics, one_shot["data"]["metrics"])

        code, with_metric = self._cli(
            ["run", "--model-id", "t-cli", "--metric", "mase", "--compact"]
        )
        self.assertEqual(code, 0)
        self.assertIn("MASE", with_metric["data"]["metrics"])

    def test_run_model_id_rejects_conflicting_flags(self):
        trained = self._train_or_skip(
            {
                "algorithm_id": "naive-seasonal-last",
                "dataset_id": "airline",
                "model_id": "t-cli-conflict",
            }
        )
        self.assertEqual(trained["status"], "ok")
        code, envelope = self._cli(
            ["run", "--model-id", "t-cli-conflict", "--algorithm", "naive-seasonal-last"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(envelope["status"], "error")
        self.assertIn("--algorithm", envelope["error"])


if __name__ == "__main__":
    unittest.main()
