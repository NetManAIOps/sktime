"""Distance-based time-series classification on the built-in UnitTest dataset.

Run from the repository root:
python3 .agent/skills/time-series-sandbox/user_cases/02_classification_knn_unit_test.py
"""

from sklearn.metrics import accuracy_score

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_unit_test


def main():
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)

    classifier = KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        distance="euclidean",
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("algorithm=KNeighborsTimeSeriesClassifier(distance='euclidean')")
    print(f"train_shape={X_train.shape} test_shape={X_test.shape}")
    print(f"accuracy={accuracy_score(y_test, y_pred):.4f}")
    print(f"first_10_predictions={list(y_pred[:10])}")


if __name__ == "__main__":
    main()
