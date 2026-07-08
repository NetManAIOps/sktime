---
name: time-series-sandbox
description: OpenClaw/Codex skill for the NetManAIOps/sktime Time Series Sandbox repository. Use when working in this repo or answering questions about setup, algorithms, datasets, runnable examples, notebooks, docs, Feishu KB routing, the TSBox Playground web app, forecasting, classification, regression, clustering, detection, transformations, distances/kernels, causal discovery, foundation-model forecasters, or choosing a Time Series Sandbox solution.
---

# Time Series Sandbox

Use this skill to answer repository-specific questions and to generate runnable
code for the Time Series Sandbox fork at:

- `https://github.com/NetManAIOps/sktime.git`

Prefer repository-native APIs under `sktime/...`. Do not replace them with
generic external alternatives unless the user explicitly asks.

## Setup

For a fresh clone with broad optional dependencies:

```bash
git clone https://github.com/NetManAIOps/sktime.git
cd sktime
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[all_extras]"
python3 -c "import sktime; print('Time Series Sandbox ready:', sktime.__version__)"
```

For an existing clone, run the same commands from the repository root without
`git clone`. If `python` is missing, use `python3`.

Use the bundled helper only when the user wants a one-command local setup:

```bash
bash .agent/skills/time-series-sandbox/setup.sh /path/to/sktime
```

If code execution fails because dependencies are missing, report the exact
missing package and suggest either `python3 -m pip install -e .` for core
dependencies or `python3 -m pip install -e ".[all_extras]"` for soft
dependencies.

## Repository Map

Use these current top-level capability areas:

- Forecasting: `sktime/forecasting`
- Classification: `sktime/classification`
- Regression: `sktime/regression`
- Clustering: `sktime/clustering`
- Detection and changepoints: `sktime/detection`
- Transformations and feature engineering: `sktime/transformations`
- Distances and kernels: `sktime/distances`, `sktime/dists_kernels`
- Alignment: `sktime/alignment`
- Parameter estimation: `sktime/param_est`
- Performance metrics: `sktime/performance_metrics`
- Splitters and evaluation helpers: `sktime/split`, `sktime/benchmarking`
- Dataset loaders and dataset classes: `sktime/datasets`
- Causal discovery: `sktime/causal_discovery`
- Deep/foundation model integrations: `sktime/forecasting`, `sktime/libs`,
  `sktime/networks`

## Catalog-First Workflow

For any request about available methods, datasets, which algorithm to use, or
how to map a user problem to a solution:

1. Read `REPO_METHODS_AND_DATASETS.md` first.
2. Identify the task type: forecasting, classification, regression,
   clustering, detection, transformation, distance/kernel, causal discovery, or
   dataset lookup.
3. Prefer catalog entries with concrete method names, module paths, and dataset
   loaders.
4. For classification datasets, inspect the expanded UCR/UEA section.
5. For forecasting datasets, inspect built-in forecasting loaders and the
   expanded Monash/TSF entries when present.
6. If the catalog is not enough, inspect the relevant source path under
   `sktime/` and label findings as verified from code.

Return solution recommendations with:

- exact method and dataset names
- category/subcategory
- module path
- minimal import/fit/predict or load snippet
- dependency caveats for soft-dependency estimators

## Code Generation Workflow

When writing code for a user:

1. Match the task, input shape, expected output, and constraints.
2. Choose the simplest repository-native API that satisfies the request.
3. Prefer built-in datasets/loaders for examples unless the user supplied data.
4. Run the code from the repository root when practical.
5. Include raw execution output in the final answer. If execution is blocked,
   include the blocker and the exact command needed to unblock it.

Final answers for code tasks must include:

- algorithm names
- exact API paths, for example `sktime.forecasting.naive.NaiveForecaster`
- core runnable snippet
- raw output or the execution blocker
- plain-language interpretation

## User Case Examples

Use the example scripts in `user_cases/` as starting points. Read or run only
the example that matches the user request.

- `user_cases/01_forecasting_naive_airline.py`: seasonal naive forecasting on
  `load_airline`
- `user_cases/02_classification_knn_unit_test.py`: distance-based time-series
  classification on `load_unit_test`
- `user_cases/03_clustering_kmeans_arrow_head.py`: time-series k-means on
  `load_arrow_head`
- `user_cases/04_detection_threshold_synthetic.py`: threshold-based anomaly or
  segment detection on a synthetic signal
- `user_cases/05_causal_notears_synthetic.py`: native NOTEARS causal discovery
  on a small synthetic tabular SEM

Run examples from the repository root:

```bash
python3 .agent/skills/time-series-sandbox/user_cases/01_forecasting_naive_airline.py
```

## TSBox Playground

The repo ships a runnable web Playground as a top-level module at
`playground/` (it was moved out of this skill; it is NOT under `.agent/`).

Use it when the user wants an executable TSBox/Sandbox demo, a browser UI,
task/dataset/algorithm filtering, evaluation results, generated code, reports,
or a quick way to review or demo sktime estimators end-to-end.

### Run and test

Start it from the repository root with Python 3.10+:

```bash
python3 playground/server.py          # serves http://127.0.0.1:8765
python3 playground/test_playground.py # unittest suite
```

Open `http://127.0.0.1:8765`.

### Layout

- `playground/server.py`: stdlib `ThreadingHTTPServer`. Endpoints: `/`,
  `/api/catalog`, `/api/run`, `/api/export/script`, `/api/export/report`.
  Static assets live in `playground/static/`.
- `playground/catalog.py`: builds the task/algorithm/dataset catalog consumed
  by the front end, and discovers sktime estimators.
- `playground/runners.py`: validates specs, runs experiments, and generates the
  reproduction script and report for each run.
- `playground/hf_data.py`: Hugging Face loader for online forecasting series.
- `playground/test_playground.py`: unittest suite.

### Dynamic algorithm registration

The Playground is fully dynamic. `catalog.discover_registered_algorithms()`
walks `sktime.registry.all_estimators()` for `forecaster`, `classifier`, and
`detector`, and exposes EVERY discovered estimator as enabled, plus its numeric
scalar hyperparameters from `get_params()`. There is no environment gating and
no disabled list: whatever sktime has registered shows up and is runnable.

- Three curated defaults (`NaiveForecaster`, `SummaryClassifier`,
  `ThresholdDetector`) keep their hand-tuned runners and are listed first.
- Every other discovered estimator goes through a generic per-task runner
  (`_run_<task>_generic`) and a generic export-script generator.
- Evaluation parameters (`horizon`; `threshold`/`window` for the curated
  anomaly detector) are split from estimator constructor parameters via
  `catalog.split_params()`.

Soft dependencies are not required to start the Playground, but they unlock most
discovered estimators. For broad coverage install them, for example:

```bash
python3 -m pip install statsmodels pmdarima numba pyod skchange stumpy tsfresh arch tbats
```

(or `python3 -m pip install -e ".[all_extras]"` for everything). Estimators that
still cannot run (missing dependency, mandatory constructor argument, or very
slow) fail gracefully as a `blocked` result with a clear reason, never a server
crash.

### Gotchas (verified; keep these intact)

- `server.py` JSON responses sanitize non-finite floats (`inf`/`nan` -> `null`).
  Some estimators expose `float('inf')` parameter defaults, which break the
  browser's `JSON.parse` if emitted raw. Do not remove the sanitize step.
- `static/app.js` `refreshOptions()` preserves the current algorithm selection
  across rebuilds; without that, changing the algorithm snaps back to the first
  (curated) option.
- `/api/catalog` makes a live Hugging Face metadata request on every call (no
  cache); the first page load depends on that round-trip.
- The curated anomaly `ThresholdDetector` detrends with a rolling median and
  thresholds the z-score of the residual, not the raw series; a raw-value
  threshold is meaningless on strongly trending data.

### Review and dev loop

A fast way to review or iterate on the Playground is the edit -> run ->
screenshot loop with a headless browser (for example Playwright):

```bash
python3 playground/server.py &        # serve the UI
python3 playground/test_playground.py # run unit tests
# drive http://127.0.0.1:8765 with a headless browser and capture screenshots
```

Capture the initial state plus one run of each task (forecasting,
classification, anomaly_detection) and read the screenshots back to spot visual
or functional regressions. The front end renders the dynamic algorithm dropdown
and per-estimator parameters automatically, so newly registered estimators
appear without any front-end changes.

## Causal Discovery

Use `sktime.causal_discovery` for causal graph tasks:

- `PC`: constraint-based tabular/i.i.d. CPDAG, soft dependency `causal-learn`
- `GES`: score-based tabular/i.i.d. CPDAG, soft dependency `causal-learn`
- `PCMCI`: lagged multivariate time-series graph, soft dependency `tigramite`
- `NOTEARS`: native linear tabular DAG discovery

Use bundled causal benchmark loaders when suitable:

- `sktime.datasets.load_sachs(return_true_graph=True)`
- `sktime.datasets.load_alarm(return_true_graph=True)`
- `sktime.datasets.load_asia(return_true_graph=True)`
- `sktime.datasets.load_causal_bnlearn_dataset(...)`

For teaching notebooks, prefer compact synthetic systems with known ground
truth. For lagged causality, visualize edges as `source[t-k] -> target[t]`.

## Notebooks and Reference Cases

When the user asks for examples or lectures:

1. Search notebooks under `examples/` and `lectures/`.
2. Return relevant `.ipynb` paths.
3. Build Colab links by appending the notebook path to:
   `https://colab.research.google.com/github/NetManAIOps/sktime/blob/main/`

Current causal discovery lecture notebooks include:

- `lectures/lec9/pc.ipynb`
- `lectures/lec9/pcmci.ipynb`
- `lectures/lec9/ges.ipynb`
- `lectures/lec9/causal_discovery_benchmark_demo.ipynb`
- `lectures/lec9/difference_in_differences.ipynb`

## Documentation Routing

For Time Series Sandbox feature details:

1. If Feishu/Lark tooling is available, search Feishu KB for
   `Time Series Sandbox` first and summarize any sandbox-specific additions.
2. Then check repository docs under `docs/`.
3. Then inspect source paths under `sktime/...`.

Label provenance in answers as one of:

- Feishu KB
- Repository docs (`docs/`)
- Catalog (`REPO_METHODS_AND_DATASETS.md`)
- Code paths (`sktime/...`)

## Response Style

Be concise and action-first. Prefer commands, exact paths, exact API names, and
small runnable snippets. Clearly separate verified facts from inferences.
