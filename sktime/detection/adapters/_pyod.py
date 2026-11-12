# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Adapt PyOD estimators to the sktime anomaly-detection interface."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _check_soft_dependencies

__author__ = ["mloning", "satya-pattnaik", "fkiraly"]
__all__ = ["PyODDetector"]


class PyODDetector(BaseDetector):
    """Wrap a PyOD estimator without changing its training or scoring algorithm.

    Parameters
    ----------
    estimator : PyOD estimator
        Unfitted estimator providing fit, predict and decision_function.

    Notes
    -----
    Each time point is an observation. Imputation, scaling and any feature
    construction are the caller's responsibility. Extra fit/score arguments,
    including validation data and preprocessing masks, are unused.
    ``predict`` returns anomaly positions using the fitted PyOD threshold.
    ``transform_scores`` returns one float64 score per observation, with larger
    values indicating greater abnormality, without applying a threshold.
    """

    _tags = {
        "authors": ["mloning", "satya-pattnaik", "fkiraly"],
        "maintainers": "satya-pattnaik",
        "python_dependencies": "pyod",
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "tests:core": True,
    }

    def __init__(self, estimator):
        self.estimator = estimator
        super().__init__()

    def _fit(self, X, y=None, **kwargs):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X.to_numpy())
        return self

    def _predict(self, X):
        labels = self.estimator_.predict(X.to_numpy())
        return pd.Series(np.flatnonzero(labels), dtype="int64")

    def _transform_scores(self, X, **kwargs):
        scores = self.estimator_.decision_function(X.to_numpy())
        return np.asarray(scores, dtype=np.float64).reshape(-1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        if _check_soft_dependencies("pyod", severity="none"):
            from pyod.models.knn import KNN

            params0 = {"estimator": KNN()}
            params1 = {"estimator": KNN(n_neighbors=3)}
        else:
            params0 = {"estimator": "foo"}
            params1 = {"estimator": "bar"}
        return [params0, params1]
