# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""PC algorithm wrapper for sktime causal discovery API."""

__author__ = ["sktime developers"]
__all__ = ["PC"]

import numpy as np

from sktime.causal_discovery.base import BaseCausalDiscoverer


def _causallearn_graph_to_cpdag(raw_graph):
    """Convert causal-learn edge endpoint matrix to sktime CPDAG encoding."""
    raw_graph = np.asarray(raw_graph)
    n_vars = raw_graph.shape[0]
    cpdag = np.zeros((n_vars, n_vars), dtype=int)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            ij = raw_graph[i, j]
            ji = raw_graph[j, i]

            if ij == -1 and ji == 1:
                cpdag[i, j] = 1
            elif ij == 1 and ji == -1:
                cpdag[j, i] = 1
            elif ij != 0 and ji != 0:
                cpdag[i, j] = -1
                cpdag[j, i] = -1

    return cpdag


class PC(BaseCausalDiscoverer):
    """PC causal discovery algorithm.

    This estimator wraps the Python implementation from ``causal-learn`` and returns
    a CPDAG encoded in the sktime causal discovery interface.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level used in conditional independence testing.
    indep_test : str, default="fisherz"
        Conditional independence test name understood by ``causal-learn``.
    stable : bool, default=True
        Whether to run the order-independent stable PC variant.
    uc_rule : int, default=0
        Unshielded collider orientation rule used by ``causal-learn``.
    uc_priority : int, default=2
        Priority policy for collider orientation in ``causal-learn``.
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "capability:time_series": False,
        "capability:iid": True,
        "graph_type": "CPDAG",
        "python_dependencies": "causal-learn",
    }

    def __init__(
        self,
        alpha=0.05,
        indep_test="fisherz",
        stable=True,
        uc_rule=0,
        uc_priority=2,
    ):
        self.alpha = alpha
        self.indep_test = indep_test
        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        super().__init__()

    def _fit(self, X):
        from causallearn.search.ConstraintBased.PC import pc

        result = pc(
            X.to_numpy(),
            alpha=self.alpha,
            indep_test=self.indep_test,
            stable=self.stable,
            uc_rule=self.uc_rule,
            uc_priority=self.uc_priority,
            show_progress=False,
        )

        graph_obj = getattr(result, "G", result)
        raw_graph = getattr(graph_obj, "graph", graph_obj)

        self.adjacency_matrix_ = _causallearn_graph_to_cpdag(raw_graph)
        self.variable_names_ = X.columns.tolist()
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"alpha": 0.1}
