"""Seasonal naive forecasting on the built-in Airline dataset.

Run from the repository root:
python3 .agent/skills/time-series-sandbox/user_cases/01_forecasting_naive_airline.py
"""

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


def main():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)

    forecaster = NaiveForecaster(strategy="last", sp=12)
    forecaster.fit(y_train)

    fh = list(range(1, len(y_test) + 1))
    y_pred = forecaster.predict(fh=fh)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("algorithm=NaiveForecaster(strategy='last', sp=12)")
    print(f"train_length={len(y_train)} test_length={len(y_test)}")
    print("predictions=")
    print(y_pred.to_string())
    print(f"mape={mape:.4f}")


if __name__ == "__main__":
    main()
