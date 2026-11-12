# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Per-variable scaling using training-series statistics."""

__all__ = ["MinMaxScaler", "StandardScaler"]

import pandas as pd
from sklearn.preprocessing import MinMaxScaler as _SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler

from sktime.transformations.base import BaseTransformer


class StandardScaler(BaseTransformer):
    """Standardize each variable using its training mean and variance.

    Implements the sktime transformer interface and delegates numerical
    calculations to sklearn's StandardScaler. Statistics are fitted only in
    fit; transform and inverse_transform reuse them without updating them.
    DataFrame indices, columns and sequence length are preserved. In panel
    inputs, sktime fits a separate scaler per series instance.

    Parameters
    ----------
    with_mean : bool, default=True
        Subtract the training mean from each variable.
    with_std : bool, default=True
        Divide each variable by its training standard deviation (ddof=0).
        Constant variables use a scale of 1, as in sklearn.

    Attributes
    ----------
    mean_ : np.ndarray or None
        Per-variable training mean, following sklearn's flag semantics.
    var_ : np.ndarray or None
        Per-variable training variance, following sklearn's flag semantics.
    scale_ : np.ndarray or None
        Per-variable scale; None when with_std=False.
    scaler_ : sklearn.preprocessing.StandardScaler
        Fitted numerical implementation. Inputs are never modified in place.

    Notes
    -----
    NaNs are ignored when fitting statistics and preserved when transforming;
    this transformer does not impute missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.series.scale import StandardScaler
    >>> train = pd.DataFrame({"data": [1.0, 2.0, 3.0]})
    >>> scaler = StandardScaler().fit(train)
    >>> scaler.transform(train).shape
    (3, 1)
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:categorical_in_X": False,
        "capability:inverse_transform": True,
        "transform-returns-same-time-index": True,
    }

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        super().__init__()

    def _fit(self, X, y=None):
        """Fit per-column statistics on the training data only."""
        self.scaler_ = _SklearnStandardScaler(
            with_mean=self.with_mean, with_std=self.with_std
        ).fit(X)
        self.mean_ = self.scaler_.mean_
        self.var_ = self.scaler_.var_
        self.scale_ = self.scaler_.scale_
        return self

    def _transform(self, X, y=None):
        """Apply the fitted statistics without changing them."""
        return pd.DataFrame(
            self.scaler_.transform(X), index=X.index, columns=X.columns
        )

    def _inverse_transform(self, X, y=None):
        """Restore the original scale using the fitted statistics."""
        return pd.DataFrame(
            self.scaler_.inverse_transform(X), index=X.index, columns=X.columns
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return configurations covering centering and scaling switches."""
        return [
            {},
            {"with_mean": False},
            {"with_std": False},
            {"with_mean": False, "with_std": False},
        ]


class MinMaxScaler(BaseTransformer):
    """Rescale each variable using its minimum and maximum on the training data.

    The scaler preserves the time index, columns and missing positions.
    Values outside the training range can map outside the target interval.

    Parameters
    ----------
    feature_range : tuple or list of float, default=(0.0, 1.0)
        Lower and upper endpoints of the transformed training interval.
    clip : bool, default=False
        Clip transformed values to feature_range.
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:categorical_in_X": False,
        "capability:inverse_transform": True,
        "transform-returns-same-time-index": True,
    }

    def __init__(self, feature_range=(0.0, 1.0), clip=False):
        self.feature_range = feature_range
        self.clip = clip
        super().__init__()

    def _fit(self, X, y=None):
        self.scaler_ = _SklearnMinMaxScaler(
            feature_range=tuple(self.feature_range), clip=self.clip
        ).fit(X)
        self.data_min_ = self.scaler_.data_min_
        self.data_max_ = self.scaler_.data_max_
        return self

    def _transform(self, X, y=None):
        return pd.DataFrame(
            self.scaler_.transform(X), index=X.index, columns=X.columns
        )

    def _inverse_transform(self, X, y=None):
        return pd.DataFrame(
            self.scaler_.inverse_transform(X), index=X.index, columns=X.columns
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return a default and a clipped, non-unit interval configuration."""
        return [{}, {"feature_range": (-1.0, 1.0), "clip": True}]
