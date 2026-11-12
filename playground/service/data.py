"""CSV loading shared by training, prediction and evaluation services."""

from pathlib import Path

import pandas as pd


def read_csv(data_path: str | Path, *, float_precision=None) -> pd.DataFrame:
    """Read a headered CSV, using its optional timestamp column as the index."""
    path = Path(data_path).expanduser().resolve()
    if path.suffix.lower() != ".csv":
        raise ValueError("Time series must be provided as a .csv file")
    frame = pd.read_csv(path, float_precision=float_precision)
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    return frame


def load_csv(data_path: str | Path) -> pd.DataFrame:
    """Return float64 value columns, without timestamps or ground-truth labels."""
    values = read_csv(data_path).drop(columns=["label"], errors="ignore")
    if values.shape[1] == 0:
        raise ValueError("CSV must contain at least one numerical value column")
    return values.astype("float64")


def resolve_data_path(data: dict, split: str) -> Path:
    """Resolve a local split or a registered single-series anomaly dataset.

    Registered IDs currently point to an unsplit CSV, not a held-out split.
    Local paths resolve against cwd; registered paths resolve against the repo.
    """
    if "dataset_id" in data: # 目前默认不处理 split 而返回完整的单条时序
        if any(key in data for key in ("train", "val", "test")):
            raise ValueError("Use either dataset_id or local data paths, not both")
        from playground.catalog import REPO_ROOT, get_dataset

        dataset_id = data["dataset_id"]
        dataset = get_dataset(dataset_id)
        if dataset is None or not dataset.get("enabled"):
            raise ValueError(f"Dataset is not enabled: {dataset_id}")
        if dataset.get("task") != "anomaly_detection":
            raise ValueError("Data loading currently supports anomaly detection only")
        if not dataset.get("path"):
            raise ValueError(
                f"Dataset {dataset_id} does not identify a single CSV series; "
                "provide local split paths"
            )
        return (REPO_ROOT / dataset["path"]).expanduser().resolve()

    path = data.get(split)
    if path is None:
        raise ValueError(f"data.{split} is required")
    return Path(path).expanduser().resolve()


def load_training_data(data: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load train and optional val; never read test or automatically split data.

    Registered Yahoo/MITDB IDs load the full CSV as train, with no validation.
    No imputation, scaling, windowing, sorting or other preprocessing is done.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    X_train = load_csv(resolve_data_path(data, "train"))
    X_val = load_csv(data["val"]) if data.get("val") is not None else None
    return X_train, X_val


def load_test_data(data: dict) -> pd.DataFrame:
    """Load test values, excluding any label column from model input."""
    return load_csv(resolve_data_path(data, "test"))


def load_test_labels(data: dict) -> pd.Series:
    """Load ground truth only for evaluation, from the test CSV's label column."""
    frame = read_csv(resolve_data_path(data, "test"))
    if "label" not in frame.columns:
        raise ValueError("Evaluation requires a label column in the test CSV")
    return frame["label"]
