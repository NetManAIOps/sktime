"""Tests for the DevAD adapter detectors (sktime.detection.adapters.devad).

Covers every family registered in the vendored DevAD ``ModelRegistry``:
default constructibility of the adapters and a small-data fit/predict
smoke run, plus the ``BaseModel.evaluate``/``metric_all`` path.
"""

__author__ = ["xiezhe"]

import numpy as np
import pandas as pd
import pytest

from sktime.detection.adapters.devad import (
    DevADAnomalyTransformerDetector,
    DevADBeatGANDetector,
    DevADCOUTADetector,
    DevADDAGMMDetector,
    DevADDonutDetector,
    DevADFCVAEDetector,
    DevADFITSDetector,
    DevADIForestDetector,
    DevADKANADDetector,
    DevADKMeansADDetector,
    DevADLSTMADDetector,
    DevADModernTCNDetector,
    DevADOmniAnomalyDetector,
    DevADSubLOFDetector,
    DevADSubOCSVMDetector,
    DevADSubPCADetector,
    DevADTimesNetDetector,
    DevADTranADDetector,
    DevADUSADDetector,
)
from sktime.tests.test_switch import run_test_for_class

DEVAD_DETECTORS = [
    DevADAnomalyTransformerDetector,
    DevADBeatGANDetector,
    DevADCOUTADetector,
    DevADDAGMMDetector,
    DevADDonutDetector,
    DevADFCVAEDetector,
    DevADFITSDetector,
    DevADIForestDetector,
    DevADKANADDetector,
    DevADKMeansADDetector,
    DevADLSTMADDetector,
    DevADModernTCNDetector,
    DevADOmniAnomalyDetector,
    DevADSubLOFDetector,
    DevADSubOCSVMDetector,
    DevADSubPCADetector,
    DevADTimesNetDetector,
    DevADTranADDetector,
    DevADUSADDetector,
]


def _make_series(n=120, seed=0):
    """Short synthetic series with one injected anomaly segment."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 24.0) + 0.05 * rng.standard_normal(n)
    x[60:63] += 3.0
    return pd.DataFrame(x)


@pytest.mark.skipif(
    not run_test_for_class(DEVAD_DETECTORS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_registry_adapter_consistency():
    """Every registered DevAD family has exactly one adapter and vice versa."""
    from sktime.libs.devad.models.registry import ModelRegistry

    registry_families = set(ModelRegistry.list_families())
    adapter_families = {cls.family for cls in DEVAD_DETECTORS}
    assert registry_families == adapter_families


@pytest.mark.parametrize("cls", DEVAD_DETECTORS, ids=lambda cls: cls.__name__)
def test_default_constructible(cls):
    """All DevAD adapters are constructible without arguments."""
    if not run_test_for_class(cls):
        pytest.skip("run test only if softdeps are present")
    from sktime.libs.devad.models.registry import ModelRegistry

    detector = cls()
    info = ModelRegistry.get_model_info(detector.family)
    required = {
        name for name, param in info["params"].items() if param["required"]
    }
    # REQUIRED hyperparameters must be covered by the adapter's overrides,
    # otherwise the default-constructed detector cannot fit.
    assert required <= set(detector._default_overrides)


@pytest.mark.parametrize("cls", DEVAD_DETECTORS, ids=lambda cls: cls.__name__)
def test_fit_predict_smoke(cls):
    """Small-data fit/predict smoke run with the fast test configuration."""
    if not run_test_for_class(cls):
        pytest.skip("run test only if softdeps are present")
    X = _make_series()
    detector = cls(**cls.get_test_params())
    y = detector.fit_predict(X)

    assert isinstance(y, pd.DataFrame)
    assert list(y.columns) == ["ilocs"]
    ilocs = y["ilocs"].to_numpy()
    assert len(ilocs) > 0
    assert ilocs.min() >= 0
    assert ilocs.max() < len(X)


@pytest.mark.skipif(
    not run_test_for_class(DevADDAGMMDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_model_evaluate_smoke():
    """BaseModel.evaluate returns all label/score metrics via metric_all."""
    from sktime.libs.devad.models.registry import ModelRegistry

    X = _make_series().to_numpy(dtype=np.float32).ravel()
    y = np.zeros(len(X), dtype=np.int64)
    y[60:63] = 1

    model = ModelRegistry.create_model(
        "dagmm",
        params={"win_len": 8, "epochs": 1, "batch_size": 16},
        seed=2026,
        device="cpu",
    )
    model.fit(X)
    results = model.evaluate(X, y)

    expected = {
        "AUC-ROC",
        "AP",
        "Point-F1",
        "PA-F1",
        "Affiliation-F1",
        "Delay-F1",
        "VUS-PR",
        "VUS-ROC",
    }
    assert expected <= set(results)
    for name in expected:
        assert np.isfinite(results[name]), f"{name} is not finite"
