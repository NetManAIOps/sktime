#!/usr/bin/env python3
"""Verify PyTorch / pytorch-forecasting / sktime DeepAR setup.

1. Ensures **pip** packages: `torch`, `lightning`, `pytorch-forecasting>=1.0.0`, `scikit-base`
   (sktime needs `scikit-base` for the `skbase` module on many installs).

2. Imports `pytorch_forecasting` and checks **distribution** version is **>= 1.0.0**
   (sktime’s `PytorchForecastingDeepAR` uses this requirement, not only `import`).

3. If **Python >= 3.10**, imports `PytorchForecastingDeepAR` and constructs a tiny model
   (the same check that fails in the notebook if (1) is missing).

Run:
  python lectures/lec7/verify_deepar_env.py

Use the **same** `python` as your Jupyter kernel (e.g. `import sys; print(sys.executable)` in a cell).

**zsh / macOS:** if you `pip install` from the shell, quote the requirement, e.g.
`python -m pip install "pytorch-forecasting>=1.0.0"` — unquoted `>=` is parsed as shell redirection.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

PIP_PKGS = [
    "scikit-base",
    "torch",
    "lightning",
    "pytorch-forecasting>=1.0.0",
]


def _pip_install() -> None:
    cmd = [sys.executable, "-m", "pip", "install", *PIP_PKGS]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _distribution_ok() -> bool:
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version  # type: ignore

    try:
        v = version("pytorch-forecasting")
    except Exception:
        return False
    try:
        major = int(v.split(".")[0])
    except ValueError:
        return False
    print(f"Distribution pytorch-forecasting=={v}", flush=True)
    return major >= 1


def main() -> None:
    need_install = not _distribution_ok()
    if importlib.util.find_spec("torch") is None:
        need_install = True
    if importlib.util.find_spec("lightning") is None:
        need_install = True
    try:
        import skbase  # noqa: F401
    except ModuleNotFoundError:
        print("skbase missing — will pip install scikit-base.", flush=True)
        need_install = True

    if need_install:
        _pip_install()

    if not _distribution_ok():
        raise SystemExit("pytorch-forecasting>=1.0.0 still not satisfied after pip install.")

    import pytorch_forecasting  # noqa: F401

    print("Import pytorch_forecasting: OK", flush=True)

    if sys.version_info < (3, 10):
        print(
            "Python < 3.10: skip sktime PytorchForecastingDeepAR constructor "
            "(current sktime needs 3.10+). pytorch-forecasting part is OK.",
            flush=True,
        )
        return

    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    from sktime.forecasting.pytorchforecasting import PytorchForecastingDeepAR

    model = PytorchForecastingDeepAR(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 1,
            "logger": False,
            "enable_checkpointing": False,
        },
        model_params={
            "cell_type": "GRU",
            "rnn_layers": 1,
            "hidden_size": 4,
            "log_interval": -1,
        },
        dataset_params={"max_encoder_length": 6},
        train_to_dataloader_params={"batch_size": 4},
        broadcasting=True,
        deterministic=True,
    )
    print("OK:", model, flush=True)


if __name__ == "__main__":
    main()
