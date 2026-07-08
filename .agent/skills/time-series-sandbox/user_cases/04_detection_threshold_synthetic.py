"""Threshold-based anomaly detection on a small synthetic signal.

Run from the repository root:
python3 .agent/skills/time-series-sandbox/user_cases/04_detection_threshold_synthetic.py
"""

import pandas as pd

from sktime.detection.naive import ThresholdDetector


def main():
    y = pd.DataFrame(
        {
            "value": [
                0.1,
                0.2,
                -0.1,
                0.0,
                3.5,
                3.7,
                0.2,
                -3.1,
                -3.4,
                0.1,
            ]
        }
    )

    detector = ThresholdDetector(upper=3.0, lower=-3.0, mode="points")
    anomalies = detector.fit_predict(y)

    print("algorithm=ThresholdDetector(upper=3.0, lower=-3.0, mode='points')")
    print("signal=")
    print(y["value"].to_string(index=False))
    print("anomaly_ilocs=")
    print(anomalies.to_string(index=False))


if __name__ == "__main__":
    main()
