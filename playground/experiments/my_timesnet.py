"""Fork of registered-forecasting-TimesNetForecaster (`sktime.forecasting.tslib.TimesNetForecaster`).

Edit this file freely, then:
    labts.py check playground/experiments/my_timesnet.py
    labts.py run --algorithm user-my_timesnet --task forecasting ...

The original implementation lives in `sktime.forecasting.tslib.TimesNetForecaster`; this subclass starts as an
exact copy of its behavior. Override methods, change defaults in PARAMS, or
replace the body entirely — anything matching the `forecasting` plugin contract
(see playground/experiments/__init__.py) runs in the Playground.
"""

TASK = 'forecasting'
NAME = 'my_timesnet'
FORKED_FROM = 'registered-forecasting-TimesNetForecaster'
PARAMS = {'horizon': 12, 'context_window': 36, 'batch_size': 32, 'channel_independence': 1, 'd_ff': 128, 'd_layers': 1, 'd_model': 32, 'dropout': 0.1, 'e_layers': 2, 'factor': 3, 'label_len': 18, 'lr': 0.001, 'moving_avg': 25, 'n_heads': 8, 'num_epochs': 2, 'num_kernels': 6, 'pred_len': 12, 'seq_len': 36, 'top_k': 3}

from sktime.forecasting.tslib import TimesNetForecaster as _Base


class Algorithm(_Base):
    """Fork of TimesNetForecaster; starts identical to the original."""

    pass
