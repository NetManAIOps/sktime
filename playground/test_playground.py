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

    def test_devad_detector_runs(self):
        # DevAD adapter (sktime.detection.adapters.devad) exercises the
        # generic anomaly runner with a subsequence sklearn family.
        try:
            result = run_experiment(
                {
                    "task": "anomaly_detection",
                    "dataset_id": "yahoo",
                    "algorithm_id": "registered-anomaly_detection-DevADSubPCADetector",
                    "params": {"win_len": 8, "threshold_quantile": 0.95},
                }
            )
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["status"], "ok")
        self.assertIn("F1", result["metrics"])
        self.assertGreater(result["metrics"]["Detected"], 1)

    def test_devad_train_detect_cycle(self):
        # labts train/detect: persist an iforest model, reload it, and get a
        # run-compatible envelope back. Uses a throwaway models root.
        import tempfile

        import trainer

        try:
            with tempfile.TemporaryDirectory() as tmp:
                trainer.MODELS_ROOT = Path(tmp)
                trained = trainer.train_devad(
                    {
                        "algorithm_id": "registered-anomaly_detection-DevADIForestDetector",
                        "dataset_id": "yahoo",
                        "model_id": "t-iforest",
                        "params": {"win_len": 8},
                    }
                )
                self.assertEqual(trained["status"], "ok")
                self.assertEqual(trained["family"], "iforest")
                self.assertTrue((Path(tmp) / "t-iforest" / "model.pt").is_file())
                self.assertTrue((Path(tmp) / "t-iforest" / "manifest.json").is_file())

                models = trainer.list_trained_models()
                self.assertEqual([m["model_id"] for m in models], ["t-iforest"])

                result = trainer.detect_devad(
                    {
                        "model_id": "t-iforest",
                        "dataset_id": "yahoo",
                        "params": {"threshold_quantile": 0.95},
                    }
                )
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        finally:
            trainer.MODELS_ROOT = REPO_ROOT / "playground" / "models"
        self.assertEqual(result["status"], "ok")
        self.assertIn("F1", result["metrics"])
        self.assertGreater(result["metrics"]["Detected"], 1)
        self.assertEqual(result["spec"]["algorithm_id"], "devad-trained:t-iforest")
        self.assertIn("code", result)
        self.assertIn("report", result)

    def test_devad_train_rejects_non_devad_algorithm(self):
        import trainer

        with self.assertRaises(PlaygroundError):
            trainer.train_devad(
                {
                    "algorithm_id": "registered-anomaly_detection-PyODECODDetector",
                    "dataset_id": "yahoo",
                }
            )

    def test_devad_fits_train_detect_cycle(self):
        # Torch family end-to-end: tiny FITS training run + reload + detect.
        import tempfile

        import trainer

        try:
            with tempfile.TemporaryDirectory() as tmp:
                trainer.MODELS_ROOT = Path(tmp)
                trained = trainer.train_devad(
                    {
                        "algorithm_id": "registered-anomaly_detection-DevADFITSDetector",
                        "dataset_id": "yahoo",
                        "model_id": "t-fits",
                        "params": {"epochs": 1, "batch_size": 64},
                    }
                )
                self.assertEqual(trained["status"], "ok")
                result = trainer.detect_devad({"model_id": "t-fits", "dataset_id": "yahoo"})
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        finally:
            trainer.MODELS_ROOT = REPO_ROOT / "playground" / "models"
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["metrics"]["Detected"], 1)


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

    def test_catalog_exposes_new_sections(self):
        catalog = build_catalog(include_registered=False)
        task_ids = {task["id"] for task in catalog["tasks"]}
        self.assertIn("causal", task_ids)
        self.assertIn("metrics", catalog)
        self.assertIn("analyzers", catalog)
        self.assertIn("distances", catalog)
        self.assertTrue(catalog["analyzers"])
        self.assertTrue(catalog["distances"])
        causal_datasets = {d["id"] for d in catalog["datasets"] if d["task"] == "causal"}
        self.assertIn("causal-sachs", causal_datasets)
        causal_algos = {a["id"] for a in catalog["algorithms"] if a["task"] == "causal"}
        self.assertIn("causal-notears", causal_algos)
        self.assertIn("causal-pc", causal_algos)
        self.assertIn("causal-ges", causal_algos)


class MetricsRegistryTests(unittest.TestCase):
    def test_registry_metadata_shape(self):
        from metrics import all_metrics

        for entry in all_metrics():
            self.assertIn("id", entry)
            self.assertIn("name", entry)
            self.assertIn("task", entry)
            self.assertIn("requires", entry)
            self.assertTrue(set(entry["requires"]) <= {"scores", "labels", "values", "predictions"})

    def test_default_metric_sets_unchanged(self):
        from metrics import default_metric_ids

        self.assertEqual(default_metric_ids("forecasting"), ["mae", "mse", "mape"])
        self.assertEqual(default_metric_ids("classification"), ["accuracy", "macro_f1"])
        self.assertEqual(default_metric_ids("regression"), ["mae", "rmse", "r2"])
        self.assertEqual(default_metric_ids("clustering"), ["ari", "nmi"])
        self.assertEqual(default_metric_ids("anomaly_detection"), ["precision", "recall", "f1"])

    def test_devad_score_metrics_registered(self):
        from metrics import metrics_for_task

        ids = {entry["id"] for entry in metrics_for_task("anomaly_detection")}
        for expected in (
            "auc_roc", "ap", "point_f1", "pa_f1",
            "affiliation_f1", "delay_f1", "vus_pr", "vus_roc",
        ):
            self.assertIn(expected, ids)

    def test_unknown_metric_raises(self):
        from metrics import MetricError, resolve_metrics

        with self.assertRaises(MetricError):
            resolve_metrics(["not_a_metric"], "forecasting")


class RunnerMetricsTests(unittest.TestCase):
    def _run_or_skip_missing_deps(self, spec):
        try:
            return run_experiment(spec)
        except PlaygroundError as exc:
            if "Missing dependency" in str(exc):
                self.skipTest(str(exc))
            raise

    def test_forecasting_extra_metrics(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "naive-seasonal-last",
                "params": {"horizon": 6, "seasonal_period": 12},
                "metrics": ["mase", "rmse"],
            }
        )
        self.assertEqual(result["status"], "ok")
        # defaults are still there
        self.assertIn("MAE", result["metrics"])
        self.assertIn("MSE", result["metrics"])
        self.assertIn("MAPE", result["metrics"])
        # requested extras
        self.assertIn("MASE", result["metrics"])
        self.assertIn("RMSE", result["metrics"])

    def test_anomaly_extra_metrics_and_scores(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "anomaly_detection",
                "dataset_id": "yahoo",
                "algorithm_id": "threshold-detector",
                "params": {"threshold": 2.0, "window": 24},
                "metrics": ["pa_f1", "vus_roc", "auc_roc"],
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("PA-F1", result["metrics"])
        self.assertIn("VUS-ROC", result["metrics"])
        self.assertIn("AUC-ROC", result["metrics"])
        # continuous scores + evaluation payload for `evaluate --from`
        self.assertIn("scores", result)
        self.assertEqual(len(result["scores"]), 1000)
        self.assertIn("evaluation", result)
        self.assertEqual(len(result["evaluation"]["labels"]), 1000)

    def test_unknown_metric_blocked(self):
        with self.assertRaises(PlaygroundError):
            run_experiment(
                {
                    "task": "forecasting",
                    "dataset_id": "airline",
                    "algorithm_id": "naive-seasonal-last",
                    "metrics": ["not_a_metric"],
                }
            )


class EvaluateSavedRunTests(unittest.TestCase):
    def _run_or_skip_missing_deps(self, spec):
        try:
            return run_experiment(spec)
        except PlaygroundError as exc:
            if "Missing dependency" in str(exc):
                self.skipTest(str(exc))
            raise

    def test_evaluate_saved_anomaly_run(self):
        from runners import evaluate_saved_run

        result = self._run_or_skip_missing_deps(
            {
                "task": "anomaly_detection",
                "dataset_id": "yahoo",
                "algorithm_id": "threshold-detector",
                "params": {"threshold": 2.0, "window": 24},
                "metrics": ["pa_f1", "vus_roc"],
            }
        )
        scored = evaluate_saved_run(result, ["pa_f1", "vus_roc"])
        self.assertEqual(scored["status"], "ok")
        # re-scoring from the saved payload reproduces the run-time values
        self.assertAlmostEqual(scored["metrics"]["PA-F1"], result["metrics"]["PA-F1"], places=9)
        self.assertAlmostEqual(scored["metrics"]["VUS-ROC"], result["metrics"]["VUS-ROC"], places=9)

    def test_evaluate_saved_forecasting_run(self):
        from runners import evaluate_saved_run

        result = self._run_or_skip_missing_deps(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "naive-seasonal-last",
                "params": {"horizon": 6, "seasonal_period": 12},
            }
        )
        scored = evaluate_saved_run(result, ["mae", "rmse"])
        self.assertEqual(scored["status"], "ok")
        self.assertAlmostEqual(scored["metrics"]["MAE"], result["metrics"]["MAE"], places=6)
        self.assertIn("RMSE", scored["metrics"])

    def test_evaluate_rejects_compact_payload(self):
        from runners import evaluate_saved_run

        with self.assertRaises(PlaygroundError):
            evaluate_saved_run({"task": "anomaly_detection", "run_id": "x"}, ["pa_f1"])


class PreprocessorChainTests(unittest.TestCase):
    def _run_or_skip_missing_deps(self, spec):
        try:
            return run_experiment(spec)
        except PlaygroundError as exc:
            if "Missing dependency" in str(exc):
                self.skipTest(str(exc))
            raise

    def test_single_preprocessor_still_works(self):
        # legacy single-preprocessor spec shape must behave as before
        result = self._run_or_skip_missing_deps(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "naive-seasonal-last",
                "preprocessor_id": "registered-preprocessor-LogTransformer",
                "params": {"horizon": 6},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["preprocessors"]), 1)

    def test_chained_preprocessors_run_in_order(self):
        result = self._run_or_skip_missing_deps(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "naive-seasonal-last",
                "preprocessors": [
                    {"id": "registered-preprocessor-LogTransformer", "params": {}},
                    {"id": "registered-preprocessor-Detrender", "params": {}},
                ],
                "params": {"horizon": 6},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["preprocessors"]), 2)
        applied = [line for line in result["log"] if line.startswith("Applied preprocessor step")]
        self.assertEqual(len(applied), 2)
        self.assertIn("LogTransformer", applied[0])
        self.assertIn("Detrender", applied[1])

    def test_chain_rejects_disabled_preprocessor(self):
        with self.assertRaises(PlaygroundError):
            run_experiment(
                {
                    "task": "forecasting",
                    "dataset_id": "airline",
                    "algorithm_id": "naive-seasonal-last",
                    "preprocessors": [{"id": "registered-preprocessor-NoSuchTransformer", "params": {}}],
                }
            )

    def test_cli_step_param_parsing(self):
        from labts import UsageError, _parse_step_kv_pairs

        steps = _parse_step_kv_pairs(["1:degree=2", "2:method=box-cox"], 2, "--pre-param")
        self.assertEqual(steps, [{"degree": 2}, {"method": "box-cox"}])
        # bare key=value is fine with a single step (historic behaviour)
        self.assertEqual(_parse_step_kv_pairs(["degree=2"], 1, "--pre-param"), [{"degree": 2}])
        with self.assertRaises(UsageError):
            _parse_step_kv_pairs(["degree=2"], 2, "--pre-param")
        with self.assertRaises(UsageError):
            _parse_step_kv_pairs(["5:degree=2"], 2, "--pre-param")


class NestedSpecTests(unittest.TestCase):
    def _run_or_skip(self, spec):
        try:
            return run_experiment(spec)
        except PlaygroundError as exc:
            if "Missing dependency" in str(exc):
                self.skipTest(str(exc))
            raise

    def test_forecasting_pipeline_with_nested_sarimax(self):
        result = self._run_or_skip(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "registered-forecasting-ForecastingPipeline",
                "params": {
                    "horizon": 6,
                    "steps": [
                        {
                            "name": "sarimax",
                            "estimator": {
                                "algorithm_id": "registered-forecasting-SARIMAX",
                                "params": {},
                            },
                        }
                    ],
                },
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("MAE", result["metrics"])
        self.assertIn("ForecastingPipeline", result["code"])
        self.assertIn("SARIMAX", result["code"])

    def test_ensemble_forecaster_with_nested_members(self):
        result = self._run_or_skip(
            {
                "task": "forecasting",
                "dataset_id": "airline",
                "algorithm_id": "registered-forecasting-EnsembleForecaster",
                "params": {
                    "horizon": 6,
                    "forecasters": [
                        {
                            "name": "naive",
                            "estimator": {
                                "algorithm_id": "registered-forecasting-NaiveForecaster",
                                "params": {"strategy": "last"},
                            },
                        },
                        {
                            "name": "theta",
                            "estimator": {
                                "algorithm_id": "registered-forecasting-ThetaForecaster"
                            },
                        },
                    ],
                },
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("MAE", result["metrics"])
        # the generated script is executable Python with the nested imports
        self.assertIn("NaiveForecaster(strategy='last')", result["code"])
        self.assertIn("ThetaForecaster", result["code"])

    def test_nested_task_type_validation(self):
        # a classifier cannot be nested inside a forecasting pipeline
        with self.assertRaises(PlaygroundError):
            run_experiment(
                {
                    "task": "forecasting",
                    "dataset_id": "airline",
                    "algorithm_id": "registered-forecasting-ForecastingPipeline",
                    "params": {
                        "horizon": 6,
                        "steps": [
                            {
                                "name": "bad",
                                "estimator": {
                                    "algorithm_id": "summary-random-forest",
                                    "params": {},
                                },
                            }
                        ],
                    },
                }
            )

    def test_missing_required_param_blocked(self):
        with self.assertRaises(PlaygroundError) as ctx:
            run_experiment(
                {
                    "task": "forecasting",
                    "dataset_id": "airline",
                    "algorithm_id": "registered-forecasting-ForecastingPipeline",
                    "params": {"horizon": 6},
                }
            )
        self.assertIn("requires constructor params", str(ctx.exception))


class CausalTaskTests(unittest.TestCase):
    def test_causal_notears_on_sachs(self):
        try:
            result = run_experiment(
                {
                    "task": "causal",
                    "dataset_id": "causal-sachs",
                    "algorithm_id": "causal-notears",
                    "params": {"max_samples": 500},
                }
            )
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["status"], "ok")
        for key in ("SHD", "Edge Precision", "Edge Recall", "Edge F1", "Edges", "True Edges"):
            self.assertIn(key, result["metrics"])
        self.assertEqual(result["metrics"]["True Edges"], 17)
        self.assertGreaterEqual(result["metrics"]["SHD"], 0)
        self.assertEqual(result["series"]["kind"], "causal_graph")
        self.assertIn("NOTEARS", result["code"])
        self.assertIn("load_sachs", result["code"])

    def test_causal_disabled_algorithm_blocked(self):
        # PC needs causal-learn; when missing it must fail as blocked, not crash
        from catalog import get_enabled_algorithm

        pc = get_enabled_algorithm("causal-pc")
        if pc is not None and pc.get("enabled"):
            self.skipTest("causal-learn is installed; PC is enabled")
        with self.assertRaises(PlaygroundError):
            run_experiment(
                {
                    "task": "causal",
                    "dataset_id": "causal-sachs",
                    "algorithm_id": "causal-pc",
                }
            )


class AnalyzeDistPredictTests(unittest.TestCase):
    def test_analyze_seasonality_acf(self):
        from analyzer import run_analysis

        result = run_analysis({"analyzer_id": "seasonality-acf", "dataset_id": "airline"})
        self.assertEqual(result["status"], "ok")
        self.assertIn("sp", result["estimates"])
        self.assertIn("sp_significant", result["estimates"])
        self.assertIn(12, result["estimates"]["sp_significant"])

    def test_analyze_stationarity(self):
        from analyzer import run_analysis

        result = run_analysis({"analyzer_id": "stationarity-adf", "dataset_id": "airline"})
        self.assertEqual(result["status"], "ok")
        self.assertIn("stationary", result["estimates"])
        self.assertIn("pvalue", result["estimates"])

    def test_analyze_unknown_analyzer_blocked(self):
        from analyzer import run_analysis

        with self.assertRaises(PlaygroundError):
            run_analysis({"analyzer_id": "nope", "dataset_id": "airline"})

    def test_dist_pairwise_matrix(self):
        from catalog import get_dataset
        from domain_runners import compute_distance_matrix

        dataset = get_dataset("unit-test")
        try:
            result = compute_distance_matrix(dataset, "dtw", max_instances=6, log=[])
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["shape"], [6, 6])
        self.assertTrue(result["symmetric"])
        self.assertEqual(len(result["matrix"]), 6)
        self.assertEqual(len(result["matrix"][0]), 6)

    def test_dist_scipy_metric(self):
        from catalog import get_dataset
        from domain_runners import compute_distance_matrix

        dataset = get_dataset("unit-test")
        try:
            result = compute_distance_matrix(dataset, "scipy:cosine", max_instances=5, log=[])
        except PlaygroundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result["shape"], [5, 5])
        self.assertTrue(result["symmetric"])

    def test_dist_unknown_metric_blocked(self):
        from catalog import get_dataset
        from domain_runners import compute_distance_matrix

        with self.assertRaises(PlaygroundError):
            compute_distance_matrix(get_dataset("unit-test"), "not-a-distance")

    def test_predict_blocked_without_backend(self):
        # trainer.predict_estimator lands with M2; until then (and for unknown
        # model ids afterwards) `predict` must exit blocked, never crash.
        import io as _io
        from contextlib import redirect_stdout as _redirect

        import labts

        buffer = _io.StringIO()
        with _redirect(buffer):
            code = labts.main(["predict", "--model-id", "definitely-not-a-model"])
        self.assertEqual(code, 3)
        envelope = json.loads(buffer.getvalue())
        self.assertEqual(envelope["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
