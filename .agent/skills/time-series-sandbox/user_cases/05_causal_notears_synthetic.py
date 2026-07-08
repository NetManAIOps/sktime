"""Native NOTEARS causal discovery on a small synthetic tabular SEM.

Run from the repository root:
python3 .agent/skills/time-series-sandbox/user_cases/05_causal_notears_synthetic.py
"""

import numpy as np
import pandas as pd

from sktime.causal_discovery import NOTEARS


def main():
    rng = np.random.default_rng(7)
    n = 300

    x0 = rng.normal(size=n)
    x1 = 0.8 * x0 + rng.normal(scale=0.2, size=n)
    x2 = -0.6 * x1 + rng.normal(scale=0.2, size=n)
    X = pd.DataFrame({"source": x0, "relay": x1, "sink": x2})

    discoverer = NOTEARS(lambda1=0.01, max_iter=30, w_threshold=0.2)
    discoverer.fit(X)

    print("algorithm=NOTEARS(lambda1=0.01, max_iter=30, w_threshold=0.2)")
    print("variables=source, relay, sink")
    print("adjacency_matrix=")
    print(discoverer.get_adjacency_matrix())
    print("edge_list=")
    print(discoverer.get_edge_list())


if __name__ == "__main__":
    main()
