# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""PCMCI algorithm wrapper for sktime causal discovery API."""

__author__ = ["sktime developers"]
__all__ = ["PCMCI"]

import numpy as np

from sktime.causal_discovery.base import BaseCausalDiscoverer


class PCMCI(BaseCausalDiscoverer):
    """PCMCI causal discovery algorithm for multivariate time series.

    This estimator wraps ``tigramite``'s PCMCI procedure and returns lagged
    adjacency matrices compatible with the sktime causal discovery API.

    Parameters
    ----------
    max_lag : int, default=1
        Maximum lag considered in the discovered graph.
    alpha_level : float, default=0.05
        Significance threshold used to include links.
    pc_alpha : float or None, default=None
        Significance level used in the PC stage of PCMCI.
        If None, ``tigramite`` defaults are used.
    cond_ind_test : object or None, default=None
        Conditional independence test instance from ``tigramite``.
        If None, uses ``ParCorr()``.
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "capability:time_series": True,
        "capability:iid": False,
        "graph_type": "lagged_DAG",
        "python_dependencies": "tigramite",
    }

    def __init__(self, max_lag=1, alpha_level=0.05, pc_alpha=None, cond_ind_test=None):
        self.max_lag = max_lag
        self.alpha_level = alpha_level
        self.pc_alpha = pc_alpha
        self.cond_ind_test = cond_ind_test
        super().__init__()

    def _fit(self, X):
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI as TigramitePCMCI

        var_names = X.columns.tolist()
        tigra_df = pp.DataFrame(data=X.to_numpy(), var_names=var_names)

        cond_ind_test = self.cond_ind_test
        if cond_ind_test is None:
            cond_ind_test = ParCorr(significance="analytic")

        pcmci = TigramitePCMCI(
            dataframe=tigra_df,
            cond_ind_test=cond_ind_test,
            verbosity=0,
        )
        results = pcmci.run_pcmci(
            tau_max=self.max_lag,
            pc_alpha=self.pc_alpha,
            alpha_level=self.alpha_level,
        )

        p_matrix = np.asarray(results["p_matrix"])
        adjacency = (p_matrix <= self.alpha_level).astype(int)

        # No self-links at lag 0 by convention.
        np.fill_diagonal(adjacency[:, :, 0], 0)

        self.adjacency_matrix_ = adjacency
        self.p_values_ = p_matrix
        self.link_coefficients_ = np.asarray(results.get("val_matrix"))
        self.variable_names_ = var_names
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"max_lag": 1, "alpha_level": 0.1}
