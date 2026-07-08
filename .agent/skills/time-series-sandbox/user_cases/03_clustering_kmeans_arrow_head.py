"""Time-series k-means clustering on the built-in ArrowHead dataset.

Run from the repository root:
python3 .agent/skills/time-series-sandbox/user_cases/03_clustering_kmeans_arrow_head.py
"""

from collections import Counter

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datasets import load_arrow_head


def main():
    X_train, _ = load_arrow_head(split="train", return_X_y=True)

    clusterer = TimeSeriesKMeans(
        n_clusters=3,
        metric="euclidean",
        averaging_method="mean",
        n_init=2,
        max_iter=10,
        random_state=1,
    )
    labels = clusterer.fit_predict(X_train)

    print("algorithm=TimeSeriesKMeans(n_clusters=3, metric='euclidean')")
    print(f"train_shape={X_train.shape}")
    print(f"cluster_counts={dict(Counter(labels))}")
    print(f"first_10_labels={list(labels[:10])}")


if __name__ == "__main__":
    main()
