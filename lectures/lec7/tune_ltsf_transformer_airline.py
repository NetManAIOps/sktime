#!/usr/bin/env python3
"""
Tune LTSFTransformerForecaster on airline data using the same protocol as
lectures/lec7/deepar_seq2seq_airline.ipynb (1958 split, log1p-zscore, rolling OOS).

Usage:
  python lectures/lec7/tune_ltsf_transformer_airline.py           # screen + refine best
  python lectures/lec7/tune_ltsf_transformer_airline.py --quick    # fewer epochs / configs

Requires: sktime, torch (same env as the notebook).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Data & rolling (mirror notebook)
# -----------------------------------------------------------------------------


def load_airline_notebook_style():
    y = load_airline_module().astype(float)
    if isinstance(y.index, pd.PeriodIndex):
        y = y.copy()
        y.index = y.index.to_timestamp()
    _freq = pd.infer_freq(y.index)
    _series_freq = _freq or "ME"
    y = y.copy()
    y.index = pd.DatetimeIndex(y.index, freq=_series_freq)

    split = pd.Timestamp("1958-01-01")
    y_train = y.loc[y.index < split].copy()
    y_test = y.loc[y.index >= split].copy()
    y_train.index = pd.DatetimeIndex(y_train.index, freq=_series_freq)
    y_test.index = pd.DatetimeIndex(y_test.index, freq=_series_freq)

    lt = np.log1p(y_train.astype(float))
    mu = float(lt.mean())
    sig = float(lt.std(ddof=0))
    if sig < 1e-12:
        sig = 1.0
    y_train_scaled = (lt - mu) / sig
    y_train_scaled.name = y_train.name
    y_train_scaled.index = pd.DatetimeIndex(y_train_scaled.index, freq=_series_freq)

    return y_train, y_test, y_train_scaled, mu, sig, _series_freq


def load_airline_module():
    from sktime.datasets import load_airline

    return load_airline()


class RollingEnv:
    """Holds scaling constants + freq for rolling_oos."""

    def __init__(
        self,
        mu: float,
        sig: float,
        series_freq: str,
        y_train_scaled: pd.Series,
        y_test: pd.Series,
        train_pred_len: int,
    ):
        self.mu = mu
        self.sig = sig
        self.series_freq = series_freq
        self.y_train_scaled = y_train_scaled
        self.y_test = y_test
        self.train_pred_len = train_pred_len

        from sktime.forecasting.base import ForecastingHorizon

        self.fit_fh_roll = ForecastingHorizon(
            np.arange(1, train_pred_len + 1),
            is_relative=True,
            freq=series_freq,
        )


def inverse_scaled_log_to_passengers(y_scaled: float, env: RollingEnv) -> float:
    y_log = y_scaled * env.sig + env.mu
    return float(np.expm1(y_log))


def rolling_ltsf_oos(forecaster, env: RollingEnv) -> pd.Series:
    from sktime.forecasting.ltsf import LTSFTransformerForecaster

    assert isinstance(forecaster, LTSFTransformerForecaster)
    y_train_s = env.y_train_scaled
    y_test = env.y_test
    roll_fh = env.fit_fh_roll
    y_name = y_train_s.name

    forecaster.fit(y_train_s, fh=roll_fh)
    vals = []
    for idx in y_test.index:
        p = forecaster.predict(roll_fh)
        p0 = float(np.asarray(p.iloc[0]).ravel()[0])
        vals.append(inverse_scaled_log_to_passengers(p0, env))
        y_obs_s = (np.log1p(y_test.loc[[idx]].astype(float)) - env.mu) / env.sig
        y_obs_s.name = y_name
        _f = y_train_s.index.freq or pd.infer_freq(y_train_s.index) or env.series_freq
        y_obs_s.index = pd.DatetimeIndex(y_obs_s.index, freq=_f)
        forecaster.update(y_obs_s, update_params=False)
    return pd.Series(vals, index=y_test.index, name=y_name)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    from sktime.performance_metrics.forecasting import mean_absolute_error

    return float(mean_absolute_error(y_true, y_pred))


# -----------------------------------------------------------------------------
# Candidate configs (structure only; num_epochs set by caller)
# -----------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    # lectures/lec7/transformers.ipynb style (monthly ME, GRU-like simplicity)
    "transformers_nb": {
        "lr": 0.0005,
        "batch_size": 8,
        "d_model": 128,
        "n_heads": 8,
        "d_ff": 256,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.1,
        "factor": 5,
        "activation": "relu",
        "optimizer": None,
        "optimizer_kwargs": None,
        "criterion": None,
        "temporal_encoding": True,
        "temporal_encoding_type": "linear",
    },
    # Same trunk, higher lr (often helps Adam on small batches)
    "ref_lr001": {
        "lr": 0.001,
        "batch_size": 8,
        "d_model": 128,
        "n_heads": 8,
        "d_ff": 256,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.12,
        "factor": 5,
        "activation": "gelu",
        "optimizer": "AdamW",
        "optimizer_kwargs": {"weight_decay": 5e-5},
        "criterion": "SmoothL1",
        "temporal_encoding": True,
        "temporal_encoding_type": "linear",
    },
    # Smaller model + stronger reg (good for short series)
    "small_reg": {
        "lr": 0.001,
        "batch_size": 4,
        "d_model": 96,
        "n_heads": 4,
        "d_ff": 192,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.22,
        "factor": 4,
        "activation": "gelu",
        "optimizer": "AdamW",
        "optimizer_kwargs": {"weight_decay": 1e-4},
        "criterion": "SmoothL1",
        "temporal_encoding": True,
        "temporal_encoding_type": "linear",
    },
    # Reference trunk but SmoothL1 + AdamW only
    "ref_adamw_smooth": {
        "lr": 0.0005,
        "batch_size": 8,
        "d_model": 128,
        "n_heads": 8,
        "d_ff": 256,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.1,
        "factor": 5,
        "activation": "gelu",
        "optimizer": "AdamW",
        "optimizer_kwargs": {"weight_decay": 1e-4},
        "criterion": "SmoothL1",
        "temporal_encoding": True,
        "temporal_encoding_type": "linear",
    },
    # Even smaller + high dropout (anti-overfit)
    "tiny_high_do": {
        "lr": 0.0012,
        "batch_size": 4,
        "d_model": 64,
        "n_heads": 4,
        "d_ff": 128,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.3,
        "factor": 3,
        "activation": "gelu",
        "optimizer": "AdamW",
        "optimizer_kwargs": {"weight_decay": 2e-4},
        "criterion": "SmoothL1",
        "temporal_encoding": True,
        "temporal_encoding_type": "linear",
    },
}


def build_forecaster(
    preset: dict[str, Any],
    *,
    seq_len: int,
    context_len: int,
    pred_len: int,
    num_epochs: int,
    air_freq: str,
) -> Any:
    from sktime.forecasting.ltsf import LTSFTransformerForecaster

    kw = {**preset}
    return LTSFTransformerForecaster(
        seq_len=seq_len,
        context_len=context_len,
        pred_len=pred_len,
        num_epochs=num_epochs,
        freq=air_freq,
        **kw,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast screen (fewer epochs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--screen-epochs", type=int, default=None)
    parser.add_argument("--full-epochs", type=int, default=None)
    args = parser.parse_args()

    screen_epochs = args.screen_epochs or (35 if args.quick else 55)
    full_epochs = args.full_epochs or (160 if args.quick else 260)

    set_seed(args.seed)

    y_train, y_test, y_train_scaled, mu, sig, series_freq = load_airline_notebook_style()
    train_pred_len = 12
    seq_len = min(36, len(y_train) - train_pred_len)
    context_len = train_pred_len

    env = RollingEnv(mu, sig, series_freq, y_train_scaled, y_test, train_pred_len)

    presets_to_run = list(PRESETS.keys())
    if args.quick:
        presets_to_run = ["transformers_nb", "ref_lr001", "small_reg"]

    print(
        f"[data] train={len(y_train)} test={len(y_test)} seq_len={seq_len} "
        f"context={context_len} pred_len={train_pred_len} freq={series_freq}",
        flush=True,
    )
    print(f"[screen] epochs={screen_epochs} presets={presets_to_run}", flush=True)

    results: list[tuple[str, float, float]] = []

    for name in presets_to_run:
        preset = PRESETS[name]
        t0 = time.perf_counter()
        fcst = build_forecaster(
            preset,
            seq_len=seq_len,
            context_len=context_len,
            pred_len=train_pred_len,
            num_epochs=screen_epochs,
            air_freq=series_freq,
        )
        pred = rolling_ltsf_oos(fcst, env)
        err = mae(y_test, pred)
        dt = time.perf_counter() - t0
        results.append((name, err, dt))
        print(f"  [{name}] MAE={err:.3f} time={dt:.1f}s", flush=True)

    results.sort(key=lambda x: x[1])
    best_name, best_mae_screen, _ = results[0]
    print(f"\n[screen best] {best_name} MAE={best_mae_screen:.3f}", flush=True)

    print(f"\n[refine] training best preset '{best_name}' with epochs={full_epochs} ...", flush=True)
    t0 = time.perf_counter()
    fcst_final = build_forecaster(
        PRESETS[best_name],
        seq_len=seq_len,
        context_len=context_len,
        pred_len=train_pred_len,
        num_epochs=full_epochs,
        air_freq=series_freq,
    )
    pred_final = rolling_ltsf_oos(fcst_final, env)
    err_final = mae(y_test, pred_final)
    dt = time.perf_counter() - t0
    print(f"[refine] MAE={err_final:.3f} time={dt:.1f}s", flush=True)

    out = {
        "best_preset_name": best_name,
        "screen_mae": best_mae_screen,
        "full_epochs": full_epochs,
        "final_mae": err_final,
        "preset_kwargs": PRESETS[best_name],
        "seq_len": seq_len,
        "context_len": context_len,
        "train_pred_len": train_pred_len,
        "series_freq": series_freq,
        "num_epochs_refine": full_epochs,
    }
    out_path = Path(__file__).resolve().parent / "tune_ltsf_transformer_airline.best.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[wrote] {out_path}", flush=True)
    print("\nPaste-friendly constructor kwargs (num_epochs set separately in notebook):", flush=True)
    print(json.dumps(PRESETS[best_name], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
