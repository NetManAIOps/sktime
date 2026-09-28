# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Playground adapters for zero-shot foundation-model forecasters.

Thin, default-constructible wrappers around foundation models so the TSBox
Sandbox Playground (``playground/``) can discover and run them through its
``fit(y) -> predict(fh)`` runner contract:

- ``ChronosPlaygroundForecaster`` defaults ``ChronosForecaster`` to
  ``amazon/chronos-bolt-tiny`` (zero-shot, deterministic).
- ``TinyTimeMixerPlaygroundForecaster`` keeps the ``ibm/TTM`` defaults with
  ``fit_strategy="minimal"`` (zero-shot when the pretrained config is
  unchanged) and flips ``requires-fh-in-fit`` to False, following the TSLib
  adapter precedent (``sktime.forecasting.tslib.BaseTSLibForecaster``): when
  ``fit`` is called without ``fh`` the model falls back to the pretrained
  config's ``prediction_length`` (96 for ``ibm/TTM`` revision "main"), and
  ``predict`` serves any horizon up to that length.
"""

__all__ = [
    "ChronosPlaygroundForecaster",
    "TinyTimeMixerPlaygroundForecaster",
]

__author__ = ["xiezhe"]

from sktime.forecasting.chronos import ChronosForecaster
from sktime.forecasting.ttm import TinyTimeMixerForecaster


class ChronosPlaygroundForecaster(ChronosForecaster):
    """Chronos-Bolt zero-shot forecaster with playground-ready defaults.

    Same as ``sktime.forecasting.chronos.ChronosForecaster`` but defaulting
    ``model_path`` to ``amazon/chronos-bolt-tiny`` (~34 MB), so the estimator
    is constructible without arguments. Chronos-Bolt is zero-shot (no
    fine-tuning in ``fit``) and deterministic (median of the model's quantile
    outputs, no sampling), and its ``requires-fh-in-fit=False`` tag matches
    the playground runner contract ``fit(y) -> predict(fh)`` directly.

    Parameters
    ----------
    model_path : str, default="amazon/chronos-bolt-tiny"
        Path to the Chronos/Chronos-Bolt huggingface model.
    config : dict, optional, default=None
        Configuration overrides, see ``ChronosForecaster``.
    seed : int, optional, default=None
        Random seed (only affects sampling-based vanilla Chronos models).
    use_source_package : bool, optional, default=False
        If True, load the model from the source ``chronos`` package instead
        of the vendored ``sktime.libs.chronos``.
    ignore_deps : bool, optional, default=False
        If True, skip soft-dependency enforcement.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    .. [2] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, and others
           (2024). Chronos: Learning the Language of Time Series

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.fm_playground import ChronosPlaygroundForecaster
    >>> y = load_airline()
    >>> forecaster = ChronosPlaygroundForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "xiezhe",
            "abdulfatir",
            "lostella",
            "Z-Fran",
            "benheid",
            "geetu040",
        ],
        "maintainers": ["xiezhe"],
    }

    def __init__(
        self,
        model_path="amazon/chronos-bolt-tiny",
        config=None,
        seed=None,
        use_source_package=False,
        ignore_deps=False,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            seed=seed,
            use_source_package=use_source_package,
            ignore_deps=ignore_deps,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        return [{}]


class TinyTimeMixerPlaygroundForecaster(TinyTimeMixerForecaster):
    """TinyTimeMixer (Granite TTM) zero-shot forecaster for the playground.

    Same as ``sktime.forecasting.ttm.TinyTimeMixerForecaster`` with the
    ``ibm/TTM`` defaults (context_length 512, prediction_length 96, ~3 MB of
    weights) and ``fit_strategy="minimal"``, which reduces to zero-shot when
    the pretrained configuration is used unchanged.

    The only behavioral change is the ``requires-fh-in-fit=False`` tag,
    following the TSLib adapter precedent
    (``sktime.forecasting.tslib.BaseTSLibForecaster``): the playground runner
    calls ``fit(y)`` without a forecasting horizon. In that case the model
    keeps the pretrained config's ``prediction_length`` (96 for revision
    "main") instead of enlarging it to ``max(fh)``, and ``predict`` serves
    any horizon up to that length.

    Parameters
    ----------
    Same as ``sktime.forecasting.ttm.TinyTimeMixerForecaster``; key defaults:
    ``model_path="ibm/TTM"``, ``revision="main"``, ``fit_strategy="minimal"``.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer
    .. [2] Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S., Nguyen, N.H.,
           Gifford, W.M., Reddy, C. and Kalagnanam, J., 2024. Tiny Time
           Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot
           Forecasting of Multivariate Time Series. CoRR.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.fm_playground import (
    ...     TinyTimeMixerPlaygroundForecaster,
    ... )
    >>> y = load_airline()
    >>> forecaster = TinyTimeMixerPlaygroundForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """  # noqa: E501

    _tags = {
        "authors": ["xiezhe", "ajati", "wgifford", "vijaye12", "geetu040"],
        "maintainers": ["xiezhe"],
        # runner contract: fit(y) without fh; fall back to the pretrained
        # config's prediction_length (TSLib adapter precedent)
        "requires-fh-in-fit": False,
    }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        return [{}]
