# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""All causal discovery estimators."""

from sktime.registry import all_estimators

est_tuples = all_estimators(estimator_types="causal_discoverer", return_names=True)

if len(est_tuples) > 0:
    est_names, ests = zip(*est_tuples)
    for i, x in enumerate(est_tuples):
        exec(f"{x[0]} = ests[{i}]")
    __all__ = list(est_names)
else:
    __all__ = []
