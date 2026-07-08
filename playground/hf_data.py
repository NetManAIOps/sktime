"""Hugging Face helpers for THU-ANM online time-series data."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from catalog import HF_DATASET_ID


def load_hf_series(config: str, series_name: str | None = None):
    """Load one THU-ANM series as a pandas Series.

    The primary path uses huggingface_hub + parquet. A lightweight fallback uses
    the datasets-server first rows API, which is enough for previews and demo
    runs when parquet support is not installed.
    """
    try:
        return _load_hf_series_from_parquet(config, series_name=series_name)
    except Exception as parquet_exc:
        try:
            return _load_hf_series_from_first_rows(config, series_name=series_name)
        except Exception as preview_exc:
            raise RuntimeError(
                "Unable to load Hugging Face dataset. Install optional deps with "
                "`python3 -m pip install huggingface-hub pyarrow`, or check network. "
                f"Parquet error: {parquet_exc}. Preview error: {preview_exc}"
            ) from preview_exc


def _load_hf_series_from_parquet(config: str, series_name: str | None = None):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    filename = f"{config}/train-00000-of-00001.parquet"
    path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=filename,
        repo_type="dataset",
    )
    frame = pd.read_parquet(path)
    return _row_to_series(frame, series_name)


def _load_hf_series_from_first_rows(config: str, series_name: str | None = None):
    import pandas as pd

    url = "https://datasets-server.huggingface.co/first-rows?" + urllib.parse.urlencode(
        {"dataset": HF_DATASET_ID, "config": config, "split": "train"}
    )
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = [item["row"] for item in payload.get("rows", [])]
    if not rows:
        raise ValueError(f"No preview rows returned for {config}")
    frame = pd.DataFrame(rows)
    return _row_to_series(frame, series_name)


def _row_to_series(frame, series_name: str | None = None):
    import pandas as pd

    if series_name and series_name in set(frame["series_name"].astype(str)):
        row = frame[frame["series_name"].astype(str) == series_name].iloc[0]
    else:
        row = frame.iloc[0]

    timestamps = _maybe_json(row["timestamps"])
    values = _maybe_json(row["values"])
    series_name = str(row["series_name"])

    index = pd.to_datetime(timestamps, errors="coerce")
    if index.isna().any():
        index = pd.RangeIndex(start=0, stop=len(values), step=1)
    series = pd.Series(values, index=index, name=series_name, dtype="float64")
    return series.dropna()


def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def cache_dir() -> Path:
    return Path.home() / ".cache" / "tsbox-sandbox-playground"

