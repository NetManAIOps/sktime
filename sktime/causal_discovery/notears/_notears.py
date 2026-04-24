# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Linear NOTEARS algorithm for sktime causal discovery API."""

__author__ = ["sktime developers"]
__all__ = ["NOTEARS"]

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from sktime.causal_discovery.base import BaseCausalDiscoverer


def _notears_linear(X, lambda1, max_iter, h_tol, rho_max):
    """Estimate weighted adjacency matrix with linear NOTEARS."""
    n_samples, n_vars = X.shape

    def _adj(w):
        w_pos = w[: n_vars * n_vars].reshape(n_vars, n_vars)
        w_neg = w[n_vars * n_vars :].reshape(n_vars, n_vars)
        W = w_pos - w_neg
        np.fill_diagonal(W, 0.0)
        return W

    def _loss_and_grad(W):
        residual = X - X @ W
        loss = 0.5 / n_samples * np.sum(residual * residual)
        grad = -X.T @ residual / n_samples
        return loss, grad

    def _h_and_grad(W):
        E = expm(W * W)
        h = np.trace(E) - n_vars
        grad_h = (E.T * (2.0 * W))
        return h, grad_h

    def _objective(w, rho, alpha):
        W = _adj(w)
        loss, grad_loss = _loss_and_grad(W)
        h, grad_h = _h_and_grad(W)

        obj = loss + lambda1 * np.sum(np.abs(W)) + 0.5 * rho * h * h + alpha * h
        smooth_grad = grad_loss + (rho * h + alpha) * grad_h

        grad_pos = smooth_grad + lambda1
        grad_neg = -smooth_grad + lambda1
        grad = np.concatenate([grad_pos.ravel(), grad_neg.ravel()])
        return obj, grad

    bounds = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((0.0, None))
    bounds = bounds + bounds

    w_est = np.zeros(2 * n_vars * n_vars)
    rho, alpha, h = 1.0, 0.0, np.inf

    for _ in range(max_iter):
        while rho < rho_max:
            sol = minimize(
                fun=lambda w: _objective(w, rho, alpha),
                x0=w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            w_new = sol.x
            h_new, _ = _h_and_grad(_adj(w_new))

            if h_new <= 0.25 * h:
                w_est = w_new
                h = h_new
                break

            rho *= 10.0

        if h <= h_tol or rho >= rho_max:
            break

        alpha += rho * h

    return _adj(w_est)


class NOTEARS(BaseCausalDiscoverer):
    """Linear NOTEARS causal discovery algorithm.

    This implementation uses the augmented Lagrangian optimization approach from
    NOTEARS (Zheng et al., 2018) for linear structural equation models.

    Parameters
    ----------
    lambda1 : float, default=0.01
        L1 regularization parameter controlling graph sparsity.
    max_iter : int, default=100
        Maximum number of augmented Lagrangian outer iterations.
    h_tol : float, default=1e-8
        Stopping tolerance for DAG constraint value.
    rho_max : float, default=1e16
        Maximum penalty parameter in augmented Lagrangian.
    w_threshold : float, default=0.3
        Threshold for converting weighted edges to binary adjacency entries.
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "capability:time_series": False,
        "capability:iid": True,
        "graph_type": "DAG",
        "python_dependencies": None,
    }

    def __init__(
        self,
        lambda1=0.01,
        max_iter=100,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.3,
    ):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        super().__init__()

    def _fit(self, X):
        X_np = X.to_numpy(dtype=float)

        W = _notears_linear(
            X=X_np,
            lambda1=self.lambda1,
            max_iter=self.max_iter,
            h_tol=self.h_tol,
            rho_max=self.rho_max,
        )

        adjacency = (np.abs(W) > self.w_threshold).astype(int)
        np.fill_diagonal(adjacency, 0)

        self.link_coefficients_ = W
        self.adjacency_matrix_ = adjacency
        self.variable_names_ = X.columns.tolist()
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "lambda1": 0.05,
            "max_iter": 20,
            "h_tol": 1e-6,
            "w_threshold": 0.2,
        }
