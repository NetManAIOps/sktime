"""Fork of registered-anomaly_detection-PyODECODDetector (`sktime.detection.adapters.pyod.PyODECODDetector`).

Edit this file freely, then:
    labts.py check playground/experiments/my_ecod.py
    labts.py run --algorithm user-my_ecod --task anomaly_detection ...

The original implementation lives in `sktime.detection.adapters.pyod.PyODECODDetector`; this subclass starts as an
exact copy of its behavior. Override methods, change defaults in PARAMS, or
replace the body entirely — anything matching the `anomaly_detection` plugin contract
(see playground/experiments/__init__.py) runs in the Playground.
"""

TASK = 'anomaly_detection'
NAME = 'my_ecod'
FORKED_FROM = 'registered-anomaly_detection-PyODECODDetector'
PARAMS = {'contamination': 0.02}

from sktime.detection.adapters.pyod import PyODECODDetector as _Base


class Algorithm(_Base):
    """Fork of PyODECODDetector; starts identical to the original."""

    pass
