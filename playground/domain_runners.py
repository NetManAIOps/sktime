"""Domain runners for the newer LabTS tasks: causal discovery and distances.

- ``run_causal`` backs ``labts run --task causal``: fits a sktime causal
  discoverer (NOTEARS / PC / GES / PCMCI) on a bnlearn benchmark dataset with
  a ground-truth DAG and scores the discovered graph (SHD + edge
  precision/recall/F1).
- ``compute_distance_matrix`` backs ``labts dist``: pairwise distance matrices
  over panel datasets via ``sktime.distances`` and
  ``sktime.dists_kernels.ScipyDist``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from catalog import import_estimator_class, split_params
from runners import PlaygroundError, _build_estimator, _clean_number, _load_panel_xy

# ---------------------------------------------------------------------------
# causal discovery
# ---------------------------------------------------------------------------

_SACHS_CONT_DIR = (
    Path(__file__).resolve().parent.parent
    / "sktime"
    / "datasets"
    / "data"
    / "sachs_continuous"
)


def load_sachs_continuous(return_true_graph: bool = False):
    """Continuous Sachs protein-signaling data (Sachs et al. 2005).

    7466 flow-cytometry samples over 11 variables, with the 20-edge consensus
    DAG used by Zheng et al. (NOTEARS, NeurIPS 2018, Sec. 5.4). Files vendored
    from cmu-phil/example-causal-datasets (real/sachs).
    """
    import pandas as pd

    X = pd.read_csv(_SACHS_CONT_DIR / "sachs.csv", sep="\t")
    if not return_true_graph:
        return X
    edges = []
    for line in (_SACHS_CONT_DIR / "truth.txt").read_text().splitlines():
        match = re.match(r"^\d+\.\s+(\w+)\s+-->\s+(\w+)", line.strip())
        if match:
            edges.append((match.group(1), match.group(2)))
    return X, edges


def _load_causal_dataset(dataset: dict, log: list[str]):
    loader = import_estimator_class(dataset["loader"])
    X, true_edges = loader(return_true_graph=True)
    log.append(
        f"Loaded {dataset['name']} samples={len(X)} variables={X.shape[1]} "
        f"true_edges={len(true_edges)}"
    )
    return X, list(true_edges)


def _subsample(X, max_samples: int, seed: int, log: list[str]):
    if max_samples and len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=seed).sort_index()
        log.append(f"Subsampled to {len(X)} rows (seed={seed})")
    return X.reset_index(drop=True)


def graph_metrics(pred_adj, variable_names, true_edges) -> dict:
    """Score a discovered graph against a ground-truth DAG.

    pred_adj encodes 0=none, 1=directed, -1=undirected (CPDAG), possibly with
    a trailing lag axis (lagged_DAG, aggregated over lags here). SHD counts
    per-pair mark differences (none / i->j / j->i / undirected); an undirected
    prediction against a directed truth counts as one difference. Edge P/R/F1
    credit an undirected prediction when the true graph has either direction.
    """
    adj = np.asarray(pred_adj)
    if adj.ndim == 3:  # lagged_DAG: (n, n, max_lag+1) -> aggregate over lags
        adj = (np.abs(adj).sum(axis=2) > 0).astype(int)
    n = adj.shape[0]
    index_of = {str(name): i for i, name in enumerate(variable_names)}
    true_adj = np.zeros((n, n), dtype=int)
    for source, target in true_edges:
        i, j = index_of.get(str(source)), index_of.get(str(target))
        if i is not None and j is not None:
            true_adj[i, j] = 1

    def mark(matrix, i, j):
        if matrix[i, j] == 1:
            return "->"
        if matrix[j, i] == 1:
            return "<-"
        if matrix[i, j] == -1 or matrix[j, i] == -1:
            return "--"
        return ".."

    shd = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mark(adj, i, j) != mark(true_adj, i, j):
                shd += 1

    pred_edges = []  # (i, j, kind) kind in {"directed", "undirected"}
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                pred_edges.append((i, j, "directed"))
    seen_undirected = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == -1 or adj[j, i] == -1:
                key = (i, j)
                if key not in seen_undirected:
                    seen_undirected.add(key)
                    pred_edges.append((i, j, "undirected"))

    tp = 0
    matched_true = set()
    for i, j, kind in pred_edges:
        hit = None
        if kind == "directed" and true_adj[i, j] == 1:
            hit = (i, j)
        elif kind == "undirected":
            if true_adj[i, j] == 1:
                hit = (i, j)
            elif true_adj[j, i] == 1:
                hit = (j, i)
        if hit is not None:
            tp += 1
            matched_true.add(hit)
    fp = len(pred_edges) - tp
    n_true = int(true_adj.sum())
    fn = n_true - len(matched_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "SHD": int(shd),
        "Edge Precision": precision,
        "Edge Recall": recall,
        "Edge F1": f1,
        "Edges": len(pred_edges),
        "True Edges": n_true,
    }


def _causal_payload(estimator, variable_names, true_edges) -> dict:
    adj = np.asarray(estimator.get_adjacency_matrix())
    graph_type = estimator.get_tag("graph_type")
    edges = []
    if adj.ndim == 2:
        n = adj.shape[0]
        true_set = {(str(s), str(t)) for s, t in true_edges}
        true_set_rev = {(str(t), str(s)) for s, t in true_edges}
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    s, t = str(variable_names[i]), str(variable_names[j])
                    edges.append(
                        {
                            "source": s,
                            "target": t,
                            "type": "directed",
                            "in_true_graph": (s, t) in true_set,
                        }
                    )
                elif adj[i, j] == -1 and i < j:
                    s, t = str(variable_names[i]), str(variable_names[j])
                    edges.append(
                        {
                            "source": s,
                            "target": t,
                            "type": "undirected",
                            "in_true_graph": (s, t) in true_set or (s, t) in true_set_rev,
                        }
                    )
    else:  # lagged_DAG
        n = adj.shape[0]
        for i in range(n):
            for j in range(n):
                lags = np.where(adj[i, j] != 0)[0]
                for lag in lags:
                    edges.append(
                        {
                            "source": str(variable_names[i]),
                            "target": str(variable_names[j]),
                            "lag": int(lag),
                            "type": "directed",
                        }
                    )
    return {
        "graph_type": graph_type,
        "variable_names": [str(v) for v in variable_names],
        "adjacency": adj.tolist(),
        "edges": edges,
    }


def run_causal(spec: dict, dataset: dict, algorithm: dict, log: list[str]) -> dict:
    """Run a causal-discovery experiment (fit + graph scoring)."""
    eval_params, est_params = split_params("causal", spec.get("params") or {})
    max_samples = int(eval_params.get("max_samples") or 2000)
    seed = int(eval_params.get("seed") or 7)
    estimator = _build_estimator(algorithm, est_params)
    log.append(f"Estimator: {algorithm['name']} params={est_params or 'defaults'}")

    X, true_edges = _load_causal_dataset(dataset, log)
    X = _subsample(X, max_samples, seed, log)
    estimator.fit(X)
    variable_names = [str(v) for v in getattr(estimator, "variable_names_", X.columns)]
    metrics = graph_metrics(estimator.get_adjacency_matrix(), variable_names, true_edges)
    graph = _causal_payload(estimator, variable_names, true_edges)

    matched = {
        (edge["source"], edge["target"])
        for edge in graph["edges"]
        if edge.get("in_true_graph")
    }
    true_rows = [
        {
            "source": str(s),
            "target": str(t),
            "found": (str(s), str(t)) in matched
            or any(
                e["source"] == str(t) and e["target"] == str(s) and e["type"] == "undirected"
                for e in graph["edges"]
            ),
        }
        for s, t in true_edges
    ]
    return {
        "status": "ok",
        "metrics": metrics,
        "graph": graph,
        "series": {
            "kind": "causal_graph",
            "nodes": graph["variable_names"],
            "edges": graph["edges"],
            "meta": {
                "graph_type": graph["graph_type"],
                "n_samples": int(len(X)),
                "n_vars": len(graph["variable_names"]),
            },
        },
        "tables": {
            "edges": graph["edges"][:200],
            "true_edges": true_rows,
        },
        "summary": (
            f"Discovered {metrics['Edges']} edges ({metrics['True Edges']} true) with "
            f"{algorithm['name']}: SHD={metrics['SHD']}, "
            f"edge F1={metrics['Edge F1']:.3f}."
        ),
    }


def causal_script(result: dict) -> str:
    """Self-contained reproduction script for a causal run."""
    import textwrap

    spec = result["spec"]
    algorithm = result["algorithm"]
    dataset = result["dataset"]
    _eval_params, est_params = split_params("causal", spec.get("params") or {})
    module_name, _, class_name = algorithm["module"].rpartition(".")
    loader_module, _, loader_name = dataset["loader"].rpartition(".")
    args = ", ".join(f"{k}={repr(v)}" for k, v in est_params.items())
    max_samples = int(_eval_params.get("max_samples") or 2000)
    seed = int(_eval_params.get("seed") or 7)
    return textwrap.dedent(
        f"""\
        import numpy as np
        from {module_name} import {class_name}
        from {loader_module} import {loader_name}

        X, true_edges = {loader_name}(return_true_graph=True)
        if len(X) > {max_samples}:
            X = X.sample(n={max_samples}, random_state={seed}).sort_index()
        X = X.reset_index(drop=True)
        est = {class_name}({args})
        est.fit(X)
        adj = est.get_adjacency_matrix()
        names = [str(v) for v in est.variable_names_]
        print("variables:", names)
        print("discovered edges:", int((np.asarray(adj) != 0).sum()))
        print("true edges:", len(true_edges))
        """
    )


# ---------------------------------------------------------------------------
# pairwise distances
# ---------------------------------------------------------------------------

_SKTIME_DISTANCE_IDS = [
    "euclidean", "squared", "dtw", "ddtw", "wdtw", "wddtw", "erp", "edr",
    "lcss", "msm", "twe", "sbd", "smets", "dot", "granger",
]

_SCIPY_DISTANCE_IDS = [
    "euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
    "braycurtis", "cosine", "correlation", "minkowski", "hamming", "jaccard",
]

DISTANCES: list[dict] = [
    {
        "id": name,
        "name": name.upper(),
        "kind": "sktime",
        "module": "sktime.distances.pairwise_distance",
        "enabled": True,
    }
    for name in _SKTIME_DISTANCE_IDS
] + [
    {
        "id": f"scipy:{name}",
        "name": f"scipy {name}",
        "kind": "scipy",
        "module": "sktime.dists_kernels.ScipyDist",
        "enabled": True,
    }
    for name in _SCIPY_DISTANCE_IDS
]


def get_distance(metric_id: str) -> dict | None:
    return next((d for d in DISTANCES if d["id"] == metric_id), None)


def _load_panel_instances(dataset: dict, max_instances: int, log: list[str]):
    from sktime.datatypes import convert_to

    X_train, _y_train, X_test, _y_test = _load_panel_xy(dataset, log)
    X = X_train
    if len(X) > max_instances:
        X = X.iloc[:max_instances]
        log.append(f"Subsampled to {max_instances} instances for the distance matrix")
    arr = convert_to(X, to_type="numpy3D")
    return arr


def compute_distance_matrix(
    dataset: dict,
    metric_id: str,
    params: dict | None = None,
    max_instances: int = 50,
    log: list[str] | None = None,
) -> dict:
    """Pairwise distance matrix over a panel dataset's train instances."""
    log = log if log is not None else []
    entry = get_distance(metric_id)
    if entry is None:
        valid = ", ".join(d["id"] for d in DISTANCES)
        raise PlaygroundError(f"Unknown distance `{metric_id}`. Valid ids: {valid}")
    params = params or {}
    max_instances = max(2, int(max_instances or 50))
    arr = _load_panel_instances(dataset, max_instances, log)

    if entry["kind"] == "sktime":
        from sktime.distances import pairwise_distance

        matrix = pairwise_distance(arr, metric=entry["id"], **params)
    else:
        from sktime.dists_kernels import ScipyDist

        flat = arr.reshape(arr.shape[0], -1)
        matrix = np.asarray(ScipyDist(metric=entry["id"].split(":", 1)[1], **params).transform(flat))

    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    off_diag = matrix[~np.eye(n, dtype=bool)]
    return {
        "metric": entry["id"],
        "kind": entry["kind"],
        "params": params,
        "shape": [int(n), int(n)],
        "symmetric": bool(np.allclose(matrix, matrix.T, equal_nan=True)),
        "min": _clean_number(off_diag.min()) if off_diag.size else 0.0,
        "max": _clean_number(off_diag.max()) if off_diag.size else 0.0,
        "mean": _clean_number(off_diag.mean()) if off_diag.size else 0.0,
        "matrix": [[_clean_number(v) for v in row] for row in matrix],
        "instance_ids": list(range(int(n))),
    }
