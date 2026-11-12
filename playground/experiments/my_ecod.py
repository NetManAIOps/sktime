"""Fork of registered-anomaly_detection-ECOD (`sktime.detection.ecod.ECOD`).

Edit this file freely, then:
    libraryts.py check playground/experiments/my_ecod.py
    libraryts.py run --algorithm user-my_ecod --task anomaly_detection ...

The original implementation lives in `sktime.detection.ecod.ECOD`; this subclass starts as an
exact copy of its behavior. Override methods, change defaults in PARAMS, or
replace the body entirely — anything matching the `anomaly_detection` plugin contract
(see playground/experiments/__init__.py) runs in the Playground.
"""

TASK = 'anomaly_detection'
NAME = 'my_ecod'
FORKED_FROM = 'registered-anomaly_detection-ECOD'
PARAMS = {'contamination': 0.02}

from sktime.detection.ecod import ECOD as _Base


class Algorithm(_Base):
    """Fork of ECOD; starts identical to the original."""

    pass
