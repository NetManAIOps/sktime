"""Fork of registered-anomaly_detection-DevADSubPCADetector (`sktime.detection.adapters.devad.DevADSubPCADetector`).

Edit this file freely, then:
    labts.py check playground/experiments/my_subpca.py
    labts.py run --algorithm user-my_subpca --task anomaly_detection ...

The original implementation lives in `sktime.detection.adapters.devad.DevADSubPCADetector`; this subclass starts as an
exact copy of its behavior. Override methods, change defaults in PARAMS, or
replace the body entirely — anything matching the `anomaly_detection` plugin contract
(see playground/experiments/__init__.py) runs in the Playground.
"""

TASK = 'anomaly_detection'
NAME = 'my_subpca'
FORKED_FROM = 'registered-anomaly_detection-DevADSubPCADetector'
PARAMS = {'seed': 2026, 'threshold_quantile': 0.99}

from sktime.detection.adapters.devad import DevADSubPCADetector as _Base


class Algorithm(_Base):
    """Fork of DevADSubPCADetector; starts identical to the original."""

    pass
