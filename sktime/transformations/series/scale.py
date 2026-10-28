# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Per-variable scaling using training-series statistics."""

__all__ = ["StandardScaler"]

import pandas as pd
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
