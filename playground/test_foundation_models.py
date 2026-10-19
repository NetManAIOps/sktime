"""Offline smoke tests for the P0 foundation-model playground adapters.

Covers the two zero-shot wrappers in ``sktime/forecasting/fm_playground.py``:

- ``ChronosPlaygroundForecaster`` (default ``amazon/chronos-bolt-tiny``)
- ``TinyTimeMixerPlaygroundForecaster`` (default ``ibm/TTM``, zero-shot
  ``fit_strategy="minimal"`` with ``requires-fh-in-fit=False``)

Both are exercised through the playground runner contract — ``fit(y)``
*without* a forecasting horizon, then ``predict(fh=...)`` — on a short
airline split. Tests run fully offline against the prewarmed Hugging Face
cache (``HF_HUB_OFFLINE=1``); they skip when the soft dependencies or the
cached weights are unavailable.
"""

from __future__ import annotations

import os

# Must be set before huggingface_hub/transformers are imported anywhere in
# the process: offline mode makes every weight/config lookup resolve against
# the local HF cache instead of the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")

import importlib.util  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

HORIZON = [1, 2, 3]


def _missing_module():
    for module in ("torch", "transformers", "accelerate"):
        if importlib.util.find_spec(module) is None:
            return module
    return None


def _skip_if_weights_unavailable(exc):
    """Skip on errors that mean 'weights not in the local cache'."""
    text = str(exc).lower()
    markers = (
        "failed to load model configuration",
        "localentrynotfound",
        "couldn't connect",
        "could not connect",
        "connection error",
        "offline mode",
        "not found in the cache",
        "does not appear to have a file named",
    )
    if isinstance(exc, (OSError, ValueError)) and any(m in text for m in markers):
        return unittest.SkipTest(str(exc))
    return None


class FoundationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = _missing_module()
        if missing is not None:
            raise unittest.SkipTest(f"Missing dependency `{missing}`")
        from sktime.datasets import load_airline

        y = load_airline()
        cls.y_train = y.iloc[: -len(HORIZON)]
        cls.y_test = y.iloc[-len(HORIZON) :]

    def _run_or_skip(self, forecaster):
        """fit(y) without fh, then predict(fh) — the runner contract."""
        try:
            forecaster.fit(self.y_train)
            return forecaster.predict(fh=list(HORIZON))
        except Exception as exc:
            skip = _skip_if_weights_unavailable(exc)
            if skip is not None:
                raise skip from exc
            raise

    def test_chronos_default_constructible(self):
        from sktime.forecasting.fm_playground import ChronosPlaygroundForecaster

        forecaster = ChronosPlaygroundForecaster()
        self.assertEqual(forecaster.model_path, "amazon/chronos-bolt-tiny")
        self.assertFalse(forecaster.get_tag("requires-fh-in-fit"))

    def test_ttm_default_constructible(self):
        from sktime.forecasting.fm_playground import (
            TinyTimeMixerPlaygroundForecaster,
        )

        forecaster = TinyTimeMixerPlaygroundForecaster()
        self.assertEqual(forecaster.model_path, "ibm/TTM")
        self.assertEqual(forecaster.fit_strategy, "minimal")
        self.assertFalse(forecaster.get_tag("requires-fh-in-fit"))

    def test_chronos_fit_predict_airline(self):
        from sktime.forecasting.fm_playground import ChronosPlaygroundForecaster

        y_pred = self._run_or_skip(ChronosPlaygroundForecaster())
        values = y_pred.to_numpy().ravel()
        self.assertEqual(len(values), len(HORIZON))
        self.assertTrue(all(float(v) == float(v) for v in values))  # no NaN

        # Chronos-Bolt is deterministic: a second predict must match exactly.
        y_pred2 = self._run_or_skip(ChronosPlaygroundForecaster())
        self.assertTrue((y_pred.to_numpy() == y_pred2.to_numpy()).all())

    def test_ttm_fit_predict_airline(self):
        from sktime.forecasting.fm_playground import (
            TinyTimeMixerPlaygroundForecaster,
        )

        y_pred = self._run_or_skip(TinyTimeMixerPlaygroundForecaster())
        values = y_pred.to_numpy().ravel()
        self.assertEqual(len(values), len(HORIZON))
        self.assertTrue(all(float(v) == float(v) for v in values))  # no NaN

    def test_catalog_discovers_both_enabled(self):
        from catalog import discover_registered_algorithms

        entries = {
            item["id"]: item
            for item in discover_registered_algorithms()
            if item["task"] == "forecasting"
        }
        for name in (
            "ChronosPlaygroundForecaster",
            "TinyTimeMixerPlaygroundForecaster",
        ):
            entry = entries.get(f"registered-forecasting-{name}")
            self.assertIsNotNone(entry, f"{name} not discovered")
            self.assertTrue(
                entry["enabled"],
                f"{name} disabled: {entry.get('disabled_reason')}",
            )


if __name__ == "__main__":
    unittest.main()
