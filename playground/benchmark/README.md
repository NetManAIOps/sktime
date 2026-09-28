# labts benchmark — reproducibility leaderboard

A verification platform for the labts catalog: paper-reported results are
curated as **groundtruth**, re-run through the **labts CLI**, and compared on a
minimal web leaderboard.

```
groundtruth/*.json   curated claims: paper value + citation + exact labts command + tolerance
results/*.json       reproduction records, written by reproduce.py (one per claim)
state.json           merged view served at /api/benchmark (generated)
```

## View it

The leaderboard is a **self-contained static page** — state is inlined at build
time, so no server is needed:

```bash
python playground/benchmark/build_state.py     # merge catalog + groundtruth + results
# then simply open playground/benchmark/leaderboard.html in a browser
```

Optionally serve it instead (live state via `/api/benchmark`):

```bash
python playground/server.py                    # http://127.0.0.1:8765/benchmark
```

The page shows every catalog algorithm per task (green = runnable in the
current environment), and for each curated claim: the reported value, the
reproduced value, the delta vs. tolerance, the citation, and the exact command.

## Reproduce

```bash
python playground/benchmark/reproduce.py               # run pending claims
python playground/benchmark/reproduce.py --all         # re-run everything
python playground/benchmark/reproduce.py --entry ID    # one claim
python playground/benchmark/build_state.py             # refresh state.json
```

`reproduce.py` executes each claim's `command` with the current interpreter
(`python` in the recorded command is a placeholder — use an env with the soft
dependencies installed, e.g. `.venv-full`), parses the labts JSON envelope,
extracts `metric_key`, and marks the claim `reproduced` / `deviation` / `error`.

## Add a new claim

Append an entry to the matching `groundtruth/<task>.json`:

```json
{
  "id": "unique-slug",
  "algorithm": "registered-forecasting-DLinearForecaster",   // labts catalog id
  "algorithm_name": "DLinear",                               // display name
  "dataset": "hf-etth1",                                     // labts dataset id
  "dataset_name": "ETTh1",                                   // display name
  "metric": "MSE",                                           // display metric
  "metric_key": "MSE",                                       // key in the run envelope's data.metrics
  "value": 0.375,                                            // paper-reported value
  "settings": "protocol as reported in the paper",
  "source": {"title": "...", "venue": "...", "url": "...", "locator": "Table 2, row X"},
  "command": "python playground/labts.py run --task ... --compact",
  "tolerance": {"type": "relative", "value": 0.15},
  "notes": "protocol gaps, granularity caveats, why deviation is expected"
}
```

Rules of thumb:

- Only record values you actually saw in the source (paper table, official
  results CSV/page, official README). Put the exact locator in `source.locator`.
- Record the **exact** labts command that reproduces the claim; keep it runnable
  end-to-end with `--compact`.
- If the paper's protocol differs structurally from labts' evaluation (rolling
  vs. single-origin, dataset-average vs. single series, different
  implementation), say so in `notes` and choose a tolerance that reflects it —
  a `deviation` badge with an honest note is a valid, useful outcome.

## Current smoke coverage (15 claims)

| task | claim | source |
| --- | --- | --- |
| forecasting | DLinear ETTh1-96 / Weather-96 MSE, Autoformer ETTh1-96 MSE | Zeng et al., AAAI 2023, Table 2 |
| classification | ROCKET ItalyPowerDemand / ArrowHead accuracy | Dempster et al., DMKD 2020, official results CSV |
| regression | Rocket / RandomForest Covid3Month RMSE | Tan et al., DMKD 2021 (TSER archive results) |
| clustering | k-Shape ArrowHead / GunPoint ARI | Paparrizos & Gravano, SIGMOD 2015, official Matlab repo |
| anomaly_detection | LOF / IForest Yahoo AUC-ROC, IForest MITDB AUC-ROC | Paparrizos et al., PVLDB 2022 (TSB-UAD) |
| causal | NOTEARS / PC / GES Sachs continuous SHD | Zheng et al., NeurIPS 2018; Petersen, arXiv:2412.10039 |

The Sachs continuous dataset (`causal-sachs-cont`, vendored under
`sktime/datasets/data/sachs_continuous/` from
[cmu-phil/example-causal-datasets](https://github.com/cmu-phil/example-causal-datasets))
exists to make the causal claims exactly reproducible: same n=7466 data and
same 20-edge consensus DAG as the NOTEARS paper.
