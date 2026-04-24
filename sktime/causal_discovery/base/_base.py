# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for causal discovery estimator scitype.

    class name: BaseCausalDiscoverer

Scitype defining methods:
    fitting              - fit(X, y=None)
    get graph            - get_graph() -> dict
    get adjacency matrix - get_adjacency_matrix() -> np.ndarray
    get edge list        - get_edge_list() -> list

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__author__ = ["sktime developers"]
__all__ = ["BaseCausalDiscoverer"]

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.utils.dependencies import _check_estimator_deps


class BaseCausalDiscoverer(BaseEstimator):
    """Base causal discovery estimator class.

    The base causal discoverer specifies the methods and method signatures
    that all causal discovery estimators have to implement.

    Causal discovery estimators learn a causal graph from data.
    The primary output is a graph structure, not predictions.

    Specific implementations of these methods is deferred to concrete instances.

    Attributes
    ----------
    _is_fitted : bool
        Flag indicating whether the estimator has been fitted.
    _X : pd.DataFrame
        Stored data from fit.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "causal_discoverer",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:X": "Series",
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:time_series": False,
        "capability:iid": True,
        "graph_type": "DAG",
        "python_version": None,
        "python_dependencies": None,
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "fit_is_empty": False,
        "tests:core": True,
    }

    def __init__(self):
        self._is_fitted = False
        self._X = None

        super().__init__()
        _check_estimator_deps(self)

    def fit(self, X, y=None):
        """Fit causal discovery estimator to data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes `X` to self._X.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data from which to discover the causal graph.

            For i.i.d. algorithms: shape ``(n_samples, n_variables)``
            For time series algorithms: shape ``(n_timepoints, n_variables)``

        y : ignored, exists for API consistency.

        Returns
        -------
        self : Reference to self.
        """
        self.reset()

        # convert to inner mtype
        X_inner = self._check_X(X)

        self._X = X_inner

        self._fit(X=X_inner)

        self._is_fitted = True

        return self

    def _check_X(self, X):
        """Check and convert input X to inner mtype.

        Parameters
        ----------
        X : pd.DataFrame, np.ndarray, or other sktime compatible format
            Input data to check and convert.

        Returns
        -------
        X_inner : pd.DataFrame
            Data converted to the inner mtype specified by the tag.
        """
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return pd.DataFrame(X)
        else:
            return pd.DataFrame(X)

    def _fit(self, X):
        """Core fit logic. Must be implemented by subclasses.

        Should store the adjacency matrix in ``self.adjacency_matrix_``
        and variable names in ``self.variable_names_``.
        Optionally store ``self.p_values_`` and ``self.link_coefficients_``.

        Parameters
        ----------
        X : pd.DataFrame
            Data guaranteed to be of a type in ``self.get_tag("X_inner_mtype")``.

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError("abstract method")

    def get_graph(self):
        """Return the discovered causal graph.

        Returns
        -------
        graph : dict
            Dictionary containing:

            - ``"adjacency_matrix"`` : np.ndarray
                Adjacency matrix of the causal graph.
                For i.i.d. algorithms: shape ``(n_vars, n_vars)``.
                For time series algorithms: shape ``(n_vars, n_vars, max_lag+1)``.
            - ``"variable_names"`` : list of str
                Names of variables corresponding to rows/columns.
            - ``"graph_type"`` : str
                Type of graph, one of ``"DAG"``, ``"CPDAG"``, ``"lagged_DAG"``.
            - ``"edge_encoding"`` : dict
                Maps integer codes in adjacency matrix to edge semantics.
            - ``"p_values"`` : np.ndarray or None
                Significance values if available.
            - ``"link_coefficients"`` : np.ndarray or None
                Effect sizes / link strengths if available.
        """
        self.check_is_fitted()
        return {
            "adjacency_matrix": self.get_adjacency_matrix(),
            "variable_names": self.variable_names_,
            "graph_type": self.get_tag("graph_type"),
            "edge_encoding": self._get_edge_encoding(),
            "p_values": getattr(self, "p_values_", None),
            "link_coefficients": getattr(self, "link_coefficients_", None),
        }

    def get_adjacency_matrix(self):
        """Return the adjacency matrix of the discovered causal graph.

        Returns
        -------
        adjacency_matrix : np.ndarray
            For i.i.d. algorithms: shape ``(n_vars, n_vars)``.
            For time series algorithms: shape ``(n_vars, n_vars, max_lag+1)``.
        """
        self.check_is_fitted()
        return self._get_adjacency_matrix()

    def get_edge_list(self):
        """Return edge list of the discovered causal graph.

        Returns
        -------
        edges : list of tuple
            For i.i.d. algorithms: list of ``(source, target, edge_type)``
            where ``edge_type`` is an integer from the edge encoding.
            For time series algorithms: list of ``(source, target, lag, edge_type)``.
        """
        self.check_is_fitted()
        return self._get_edge_list()

    def _get_adjacency_matrix(self):
        """Extract adjacency matrix. Default reads from ``self.adjacency_matrix_``."""
        return self.adjacency_matrix_

    def _get_edge_list(self):
        """Convert adjacency matrix to edge list. Can be overridden by subclasses."""
        adj = self.adjacency_matrix_
        variable_names = self.variable_names_
        is_ts = self.get_tag("capability:time_series")

        edges = []
        if is_ts and adj.ndim == 3:
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    for tau in range(adj.shape[2]):
                        if adj[i, j, tau] != 0:
                            edges.append(
                                (variable_names[i], variable_names[j], tau, adj[i, j, tau])
                            )
        else:
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i, j] != 0:
                        edges.append(
                            (variable_names[i], variable_names[j], adj[i, j])
                        )
        return edges

    def _get_edge_encoding(self):
        """Return the edge encoding for this estimator based on graph_type tag."""
        graph_type = self.get_tag("graph_type")
        if graph_type in ("DAG", "lagged_DAG"):
            return {0: "none", 1: "directed"}
        elif graph_type == "CPDAG":
            return {0: "none", 1: "directed", -1: "undirected"}
        elif graph_type == "PAG":
            return {0: "none", 1: "directed", -1: "undirected", 2: "bidirected"}
        return {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance. ``create_test_instance`` uses the first (or only) dict in
            ``params``.
        """
        return {}
