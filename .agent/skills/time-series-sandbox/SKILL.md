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

### Catalog snapshot (for coding agents)

A read-only JSON mirror of `/api/catalog` is kept at:

- `.agent/skills/time-series-sandbox/catalog_snapshot.json`

Prefer reading this file over the static lists in this document when you need the
current, dynamic set of tasks / algorithms / preprocessors / datasets — every
estimator discovered by `discover_registered_algorithms()` and
`discover_registered_preprocessors()` appears under `algorithms` /
`preprocessors`, including ones not named here. It also carries `metrics`,
`compatibility`, `dependencies`, and `hf` metadata, plus a `_meta.generated_at`
timestamp.

Refresh it whenever the registry, dependencies, or datasets may have changed:

```bash
python3 playground/catalog.py          # writes the snapshot, no server needed
```

`playground/server.py` also refreshes it in a background thread on every
startup, so launching the Playground keeps it current. It is still a static
snapshot: if it looks stale (check `_meta.generated_at`) or a Hugging Face /
registry entry looks wrong, rerun the command above rather than trusting it.

### Layout

- `playground/server.py`: stdlib `ThreadingHTTPServer`. Endpoints: `/`,
  `/api/catalog`, `/api/run`, `/api/export/script`, `/api/export/report`.
  Static assets live in `playground/static/`.
- `playground/catalog.py`: builds the task/algorithm/dataset catalog consumed
  by the front end, and discovers sktime estimators. Also registers the
  `causal` task (NOTEARS/PC/GES/PCMCI + bnlearn datasets with ground-truth
  DAGs) and marks compositors that accept sub-estimators
  (`required_params`/`accepts_estimators`).
- `playground/runners.py`: validates specs, runs experiments, and generates the
  reproduction script and report for each run. Supports preprocessing chains
  (ordered, length-preserving steps), nested estimator specs for compositors,
  registry-driven extra metrics, continuous anomaly scores, and re-scoring of
  saved runs (`evaluate_saved_run`).
- `playground/metrics.py`: unified metric registry (sktime forecasting /
  detection metrics, sklearn task metrics, DevAD score-based anomaly
  metrics) with `id/name/task/requires/direction/default` metadata; drives
  `labts ls metrics`, `run --metric`, and `labts evaluate`.
- `playground/domain_runners.py`: causal-discovery runner (graph metrics:
  SHD + edge precision/recall/F1 against the true DAG) and the pairwise
  distance-matrix backend for `labts dist`.
- `playground/analyzer.py`: `sktime.param_est` backend for `labts analyze`
  (seasonality, stationarity, AR lag order).
- `playground/hf_data.py`: Hugging Face loader for online forecasting series.
- `playground/test_playground.py`: unittest suite.
- `playground/labts.py`: LabTS API — headless CLI version of this Playground
  for autoresearch harnesses and automated pipelines. Full parity with the web
  endpoints: `catalog` ↔ `/api/catalog`, `run` ↔ `/api/run`,
  `script` ↔ `/api/export/script`, `report` ↔ `/api/export/report`, with a
  unified JSON envelope and pipeline flow (`run --out` → `script/report
  --from`). Adds stateful model lifecycle commands for DevAD detectors:
  `train` (persist under `playground/models/`), `detect`, `ls models`; plus
  `evaluate` (re-score saved runs with registry metrics), `analyze`
  (param_est), `dist` (pairwise distances), and `predict` (generic
  persistence backend, wired for mission M2). See the `labts-api` skill.
- `playground/trainer.py`: train/detect backends for the DevAD zoo
  (`sktime/libs/devad`) used by `labts.py train`/`detect`.

### Dynamic algorithm registration

The Playground is fully dynamic. `catalog.discover_registered_algorithms()`
walks `sktime.registry.all_estimators()` for `forecaster`, `classifier`,
`regressor`, `clusterer`, and `detector`, and exposes EVERY discovered
estimator as enabled, plus its numeric scalar hyperparameters from
`get_params()`. There is no environment gating and no disabled list: whatever
sktime has registered shows up and is runnable. Estimators whose constructor
requires arguments (compositors/meta-estimators such as ForecastingPipeline,
EnsembleForecaster, reduction wrappers) are enabled too and marked with
`required_params` / `accepts_estimators` — they run through **nested
estimator specs** (`spec.params` values of the form
`{"name": "x", "estimator": {"algorithm_id": ..., "params": {...}}}`, built
recursively by `runners._build_estimator` with task-type validation).

- Five curated defaults (`NaiveForecaster`, `SummaryClassifier`,
  `SklearnRegressorPipeline`, `TimeSeriesKMeans`, `ThresholdDetector`) keep
  their hand-tuned runners and are listed first; the causal task adds curated
  `NOTEARS` (native), `PC`/`GES` (soft dep `causal-learn`), and `PCMCI`
  (soft dep `tigramite`) plus the bnlearn `causal-sachs/alarm/asia` datasets
  with ground-truth DAGs.
- Every other discovered estimator goes through a generic per-task runner
  (`_run_<task>_generic`) and a generic export-script generator.
- Evaluation parameters (`horizon`; `threshold`/`window` for the curated
  anomaly detector; `max_samples`/`seed` for causal) are split from estimator
  constructor parameters via `catalog.split_params()`.
- Preprocessors chain: `spec.preprocessors` is an ordered list of
  length/instance-preserving steps (`--preprocessor` repeatable on the CLI,
  per-step params via `--pre-param STEP:key=value`); the legacy single
  `preprocessor_id` form is a one-step chain.
- Metrics come from the unified registry in `playground/metrics.py`:
  defaults are unchanged per task; `run --metric <id>` adds registry entries
  (e.g. DevAD `pa_f1`/`vus_roc` on top of the point P/R/F1), and
  `labts evaluate --from run.json` re-scores a saved run without re-fitting
  (anomaly via the saved continuous `scores`).

Soft dependencies are not required to start the Playground, but they unlock most
discovered estimators. For broad coverage install them, for example:

```bash
python3 -m pip install statsmodels pmdarima numba pyod skchange stumpy tsfresh arch tbats
```

(or `python3 -m pip install -e ".[all_extras]"` for everything). Estimators that
still cannot run (missing dependency, mandatory constructor argument, or very
slow) fail gracefully as a `blocked` result with a clear reason, never a server
crash.

### TSLib deep-learning forecasters

Six THUML Time-Series-Library models are vendored under `sktime/libs/tslib/`
(MIT, commit `4e938a1`; see `sktime/libs/tslib/README.md` for provenance and
local import fixes) and adapted as sktime forecasters in
`sktime/forecasting/tslib.py`:

- `TimesNetForecaster`, `ITransformerForecaster`, `DLinearForecaster`,
  `AutoformerForecaster`, `FEDformerForecaster`, `FreTSForecaster`

They subclass `BaseDeepNetworkPyTorch` (soft dependency `torch`, CPU build in
`playground/requirements.txt`), are default-constructible, and therefore show
up in the Playground/labts catalog automatically as
`registered-forecasting-<ClassName>` with their numeric hyperparameters
(`seq_len`, `d_model`, `e_layers`, `num_epochs`, ...) exposed. Without torch
they appear as disabled with a "Missing dependency" reason — nothing breaks.

Adapter notes:

- `fit` works with or without `fh`; the network's `pred_len` defaults to the
  constructor `pred_len` (12) and is enlarged to the max `fh` seen in fit.
  `predict` raises for fh beyond `pred_len`.
- Input series are z-scored in fit and un-scored in predict; datetime indexes
  get TSLib time features, non-datetime indexes pass `x_mark=None`.
- Vendored layer files not needed by these six models were dropped so the
  package imports cleanly without einops/mamba_ssm/pywt/reformer_pytorch.

### PyOD point-anomaly detectors

Nine pyod models are wrapped in `sktime/detection/adapters/pyod.py` as
default-constructible detectors, so registry discovery (and hence the
Playground/labts catalog) picks them up as
`registered-anomaly_detection-<ClassName>`:

- `PyODECODDetector`, `PyODCOPODDetector` (parameter-free), `PyODLOFDetector`,
  `PyODIForestDetector`, `PyODKNNDetector`, `PyODHBOSDetector`,
  `PyODCBLOFDetector`, `PyODMCDDetector`, `PyODOCSVMDetector`

All expose `contamination` (default 0.1) plus their model-specific numeric
hyperparameters, and are classified as `point_anomaly` in the catalog.
`pyod` is pinned in `playground/requirements.txt`.

Important: the wrappers override `_predict` via `_PyODPointsMixin` to return
anomaly **positions** (ilocs). Upstream `PyODDetector._predict` returns the
anomalous points' label *values* instead, which collapses every detection
onto iloc 1 in the playground's sparse-ilocs pipeline — do not remove the
mixin.

### DevAD anomaly-detection zoo (`sktime/libs/devad`)

Seventeen anomaly-detection families are vendored under `sktime/libs/devad/`
(imported from the `tinyfire27_devad_cli` branch; see
`sktime/libs/devad/README.md` for provenance) behind a uniform
`fit(x_train)` / `detect(x_test) -> scores + start_pos` API
(`models/Base.py`, `models/registry.py`):

- torch families: BeatGAN, COUTA, Donut, FCVAE, FITS, KAN-AD, LSTM-AD,
  ModernTCN, OmniAnomaly, TimesNet, TranAD, USAD
- scikit-learn subsequence families: IForest, KMeansAD, SubLOF, SubOCSVM,
  SubPCA

Each family has an `HP` table of hyperparameters; entries marked `REQUIRED`
get defaults from the sktime adapters (`_default_overrides`). Model families
AnomalyTransformer/CAE/DAGMM/EncDecAD exist as source files but are **not**
in `ModelRegistry` (upstream WIP) and are therefore not exposed.

Two entry points consume the zoo:

- `sktime/detection/adapters/devad.py` — one default-constructible detector
  per family (`DevADFITSDetector`, ...), discoverable as
  `registered-anomaly_detection-<ClassName>`. `_predict` thresholds the
  DevAD anomaly scores at `threshold_quantile` (default 0.99) and returns
  point ilocs; `win_len`/`epochs`/`batch_size` are first-class constructor
  params, every other HP goes through the `params` dict (a JSON string is
  accepted, e.g. `--param params='{"h_dim": 64}'`).
- `playground/trainer.py` — backs `labts.py train` / `labts.py detect` /
  `labts.py ls models` on top of DevAD's `services.model_service`: training
  persists `model.pt` + `manifest.json` + `training.log` under
  `playground/models/<model_id>/` (gitignored), detection reloads the model
  and returns the standard run envelope. This is the train-once/detect-many
  counterpart of the stateless `labts run`.

The zoo's native typer CLI also works:
`python -m sktime.libs.devad.cli.app model list|info|train|detect|evaluate`
(data in/out as `.npy` files; requires `typer`, pinned in
`playground/requirements.txt`; `rich` renders the training reporter).

### Experiment plugins (`playground/experiments/`) + `labts fork`

For autoresearch coding agents: algorithms can live as **single-file plugins**
in `playground/experiments/*.py` instead of inside the `sktime` package. See
`playground/experiments/__init__.py` for the contract (top-level `TASK`,
`Algorithm` class, optional `NAME`/`PARAMS`/`FORKED_FROM`; minimal
fit/predict signatures per task; sktime estimators are also valid plugins).

- Discovery: `playground/user_algos.discover_user_algorithms()` merges plugins
  into the catalog as `user-<filename>` (re-scanned on every catalog call, so
  edits show up immediately). A broken plugin only disables itself with a
  `disabled_reason` — it never crashes catalog discovery.
- `labts.py fork <algorithm_id> [--name X]` writes a subclass scaffold of any
  enabled catalog algorithm into `experiments/`; `labts.py check <file>`
  validates the contract and runs a tiny smoke experiment.
- Plugin `PARAMS` are the declared defaults: the runners apply them under the
  run's params (excluding per-task eval params), and the UI renders them as
  knobs like any other algorithm.
- forecasting plugins may use the simple contract `fit(y)` + `predict(steps)`;
  classes with `get_params` are driven through the sktime contract instead.

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
classification, regression, clustering, anomaly_detection) and read the
screenshots back to spot visual or functional regressions. The front end
renders the dynamic algorithm dropdown and per-estimator parameters
automatically, so newly registered estimators
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
