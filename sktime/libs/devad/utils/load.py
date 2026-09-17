import numpy as np
from pathlib import Path


def load_np(
    data_path: str | Path,
    dtype=np.float32,
    name: str = "Time series",
) -> np.ndarray:

    path = Path(data_path).expanduser().resolve()

    if path.suffix.lower() != ".npy":
        raise ValueError(f"{name} must be provided as a .npy file")

    values = np.load(path, allow_pickle=False)

    if values.ndim != 1:
        raise ValueError(
            f"{name} must have shape (n,), got {values.shape}"
        )

    return values.astype(dtype)
