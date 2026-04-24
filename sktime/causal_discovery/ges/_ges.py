# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""GES algorithm wrapper for sktime causal discovery API."""

__author__ = ["sktime developers"]
__all__ = ["GES"]

import numpy as np

from sktime.causal_discovery.base import BaseCausalDiscoverer
from sktime.causal_discovery.pc._pc import _causallearn_graph_to_cpdag


class GES(BaseCausalDiscoverer):
    """Greedy Equivalence Search (GES) causal discovery algorithm.

    This estimator wraps the Python implementation from ``causal-learn`` and returns
    a CPDAG encoded in the sktime causal discovery interface.

    Parameters
    ----------
    score_func : str, default="local_score_BIC"
        Score function understood by ``causal-learn`` GES implementation.
    max_p : int, default=None
        Maximum number of parents allowed for each node.
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "capability:time_series": False,
        "capability:iid": True,
        "graph_type": "CPDAG",
        "python_dependencies": "causal-learn",
    }

    def __init__(self, score_func="local_score_BIC", max_p=None):
        self.score_func = score_func
        self.max_p = max_p
        super().__init__()

    def _fit(self, X):
        from causallearn.search.ScoreBased.GES import ges

        result = ges(
            X.to_numpy(),
            score_func=self.score_func,
            maxP=self.max_p,
        )

        graph_obj = result.get("G", result) if isinstance(result, dict) else result
        raw_graph = getattr(graph_obj, "graph", graph_obj)

        self.adjacency_matrix_ = _causallearn_graph_to_cpdag(np.asarray(raw_graph))
        self.variable_names_ = X.columns.tolist()
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"score_func": "local_score_BIC"}
