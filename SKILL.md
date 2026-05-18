---
name: time-series-sandbox
description: OpenClaw skill for Time Series Sandbox (based on NetManAIOps/sktime). Provides one-line install/run, algorithm family hints, dataset navigation, notebook case lookup, and doc routing (Feishu KB + docs folder).
---

# Time Series Sandbox Skill

Use this skill when the user asks about Time Series Sandbox (this repository), including setup, algorithms, datasets, notebooks, and docs.

Time Series Sandbox is based on:
- https://github.com/NetManAIOps/sktime.git

## Quick One-Liner: Install and Run

Make sure that you have install the `NetManAIOps/sktime` repository and set up the environment. **Do not** use the default sktime with `pip install sktime` as it may not include the latest sandbox features.
Use this single command when the user asks for a one-sentence setup command:

```bash
git clone https://github.com/NetManAIOps/sktime.git && cd sktime && python -m venv .venv && source .venv/bin/activate && python -m pip install -U pip && python -m pip install -e ".[all_extras]" && python -c "import sktime; print('Time Series Sandbox ready:', sktime.__version__)"
```

If the user already has the repo locally, use:

```bash
cd sktime && python -m venv .venv && source .venv/bin/activate && python -m pip install -U pip && python -m pip install -e ".[all_extras]" && python -c "import sktime; print('Time Series Sandbox ready:', sktime.__version__)"
```

## What This Repository Implements (High-Level)

When users ask "what algorithms are available", summarize by module category:

- Forecasting: `sktime/forecasting`
- Classification: `sktime/classification`
- Regression: `sktime/regression`
- Clustering: `sktime/clustering`
- Detection (anomaly/changepoint): `sktime/detection`
- Transformations and feature engineering: `sktime/transformations`
- Distances and kernels: `sktime/distances`, `sktime/dists_kernels`
- Alignment: `sktime/alignment`
- Parameter estimation: `sktime/param_est`
- Performance metrics: `sktime/performance_metrics`
- Pipeline/composition utilities: `sktime/pipeline`
- Data types and adapters: `sktime/datatypes`, `sktime/datasets`
- Splitters and evaluation helpers: `sktime/split`, `sktime/benchmarking`
- Causal discovery: `sktime/causal_discovery`

## Methods and Datasets Catalog (Use This First)
!!! IMPORTANT !!!
For any user request about "what methods/datasets are available" or "which solution to use",
you must consult this file first:

- `REPO_METHODS_AND_DATASETS.md`

This catalog is generated from repository source and contains:

- methods by category/subcategory/module
- public datasets/loaders
- local built-in dataset folders with paths
- expanded UCR/UEA entries (not merged into one row)
- expanded Monash TSF entries

## Detailed Prompt Workflow (Mandatory)

When users ask for algorithms, datasets, or solution mapping, follow this prompt flow:

1. Read `REPO_METHODS_AND_DATASETS.md` first.
2. Identify candidate methods/datasets by matching user task type:
   - forecasting / classification / regression / clustering / detection / transformation
3. Prefer entries that have clear module paths and concrete dataset names.
4. For classification datasets, explicitly check the expanded UCR/UEA table section.
5. For forecasting datasets, explicitly check the expanded Monash TSF table section.
6. Return recommendations in this order:
   - exact method/dataset names
   - category and subcategory
   - module/path from the catalog
   - minimal runnable import/loader snippet
7. If no exact match is found in the catalog, then inspect code paths under `sktime/` and report
   what is verified vs inferred.

Use the following response template internally when building the answer:

```text
Task type: <forecasting/classification/...>
Catalog source: REPO_METHODS_AND_DATASETS.md
Candidates:
- Method: <name> | Category/Subcategory: <...> | Module: <...>
- Dataset: <name> | Category/Subcategory: <...> | Path/Loader: <...>
Why selected:
- <reason 1>
- <reason 2>
Minimal usage:
- <import/fit/predict or load snippet>
```

## Code Generation and Execution Prompt (Mandatory)

When a user asks to "write code using algorithms in Time Series Sandbox", you must follow this prompt policy:

1. Use repository-native implementations first.
   - Prefer algorithms implemented in this repository (`sktime/...`) and avoid external alternatives unless the user explicitly asks otherwise.
2. Match the request before coding.
   - Identify task type, input/output expectation, and constraints (for example: forecasting horizon, classification labels, runtime simplicity).
3. Choose the simplest valid implementation.
   - Select the minimal algorithm and minimal dataset/loader that can satisfy the user request.
   - Keep code short and runnable; avoid unnecessary engineering complexity.
4. Execute and capture real outputs whenever possible.
   - Run the code and keep the raw output values/metrics/predictions.
   - If execution is not possible, clearly state the blocker and provide expected output format.
5. Return a structured final answer with all required parts.

Your final user-facing response must include all of the following:

- Algorithm name(s) used.
- Exact API location(s) in repository/module form (for example: `sktime.forecasting.naive.NaiveForecaster`).
- Core call snippet (minimal import + fit + predict/evaluate path).
- Raw output from execution (original values/metrics, not only summary wording).
- Plain-language interpretation of the output and why it answers the user's request.

Use this internal response scaffold when preparing the final output:

```text
Matched need:
- Task: <...>
- Constraints: <...>

Selected algorithm/API:
- Algorithm: <name>
- API path: <sktime....>
- Why this is the simplest fit: <...>

Core snippet:
<minimal runnable code>

Raw output:
<verbatim execution output>

Interpretation:
<clear explanation for user>
```

## Causal Discovery Guidance

Use this section when the user asks for causal discovery algorithms, causal
graphs, time-series causality demos, or lecture notebooks.

Repository-native APIs:

- PC: `sktime.causal_discovery.PC`
  - Tabular / i.i.d. constraint-based causal discovery.
  - Wraps `causal-learn` and returns a CPDAG.
  - Adjacency encoding: `1` directed edge, `-1` undirected edge, `0` no edge.
- GES: `sktime.causal_discovery.GES`
  - Tabular / i.i.d. score-based causal discovery.
  - Wraps `causal-learn` and returns a CPDAG.
  - Adjacency encoding is the same as PC.
- PCMCI: `sktime.causal_discovery.PCMCI`
  - Multivariate time-series causal discovery.
  - Wraps `tigramite` and returns a lagged DAG adjacency matrix with shape
    `(source_variable, target_variable, lag)`.
  - For lagged edges, `adjacency[i, j, tau] == 1` means variable `i` at
    `t - tau` points to variable `j` at `t`.
- NOTEARS: `sktime.causal_discovery.NOTEARS`
  - Native linear NOTEARS implementation for tabular DAG discovery.

Bundled causal benchmark loaders:

- `sktime.datasets.load_sachs(return_true_graph=True)`
- `sktime.datasets.load_alarm(return_true_graph=True)`
- `sktime.datasets.load_asia(return_true_graph=True)`
- generic loader: `sktime.datasets.load_causal_bnlearn_dataset(...)`

Dependency note:

- PC and GES require the soft dependency `causal-learn`.
- PCMCI requires the soft dependency `tigramite`.
- Teaching notebooks should install those dependencies explicitly in the first
  notebook cell, while importing the algorithms through `sktime.causal_discovery`.

Lecture notebook convention for causal discovery:

1. Prefer a small fixed-seed synthetic system with a known ground-truth graph
   when the goal is to teach graph recovery rather than benchmark accuracy.
2. For time-series causal discovery, keep the number of variables below 10 and
   plot the raw time series first with `matplotlib`.
3. Prefer algorithm-specific case studies when one shared dataset hides the
   method differences. Good teaching defaults are:
   - PC: a small collider / v-structure case where conditional independence
     produces identifiable arrowheads.
   - GES: a contemporaneous tabular SEM where a global BIC-style score recovers
     a sparse DAG over same-window measurements.
   - PCMCI: a lagged time-series propagation chain where direct lagged causes
     are separated from indirect lagged correlations.
   In GES and PCMCI teaching notebooks, include a PC baseline on the same data
   or on same-time slices so students can see how score-based and time-lagged
   methods differ from constraint-based PC.
4. Use a compact physical story with named variables rather than generic
   `X1`, `X2`, `X3` when the notebook is for teaching.
   For time-series lectures, prefer clean periodic shapes and visible delayed
   pulses over purely noisy autoregressive traces.
5. Use colors consistently:
   - green: correctly recovered edge
   - red: false positive edge
   - gray dashed: missed true edge
   - orange: reversed or orientation-mismatched edge
6. For PC and GES on time-series demos, use a lag-expanded tabular view such as
   `X[t-1] -> X[t]`, and explain that their output is a CPDAG over these lagged
   variables. If the CPDAG is mostly unoriented, redesign the case instead of
   presenting a graph full of bidirectional edges.
7. For PCMCI, fit directly on the multivariate time series and visualize
   lagged edges as `source[t-k] -> target[t]`.

## Reference Cases (Notebook Examples)

If users ask "do we have reference cases/examples" or "I want to see some examples" or something like that, do the following:

1. Search notebook files in:
   - `lectures/`
   - `examples/`
   - especially `lectures/lec9/` for causal discovery lecture demos
2. Return relevant `.ipynb` paths.
3. Return **clickable** Colab links by appending notebook path to:
   - `https://colab.research.google.com/github/NetManAIOps/sktime/blob/main`

Link rule:
- If notebook is `lectures/lec5/clasp.ipynb`, return:
  - `https://colab.research.google.com/github/NetManAIOps/sktime/blob/main/lectures/lec5/clasp.ipynb` as a Colab link.

4. (Important) Open the link in the user's browser (with `openclaw browser`) if possible.

Current causal discovery lecture notebooks:

- `lectures/lec9/pc_causal_discovery_demo.ipynb`
- `lectures/lec9/pcmci_causal_discovery_demo.ipynb`
- `lectures/lec9/ges_causal_discovery_demo.ipynb`

## Documentation Routing Rules

If user asks about Time Series Sandbox details or the usage of specific features, do the following:

1. If Feishu plugin is installed:
   - Prompt OpenClaw to search Feishu knowledge base first for "Time Series Sandbox" documents.
   - Extract details that may describe sandbox-specific additions beyond sktime.
   - Return the links and summaries from Feishu KB if found. Do not put the links in the code snippet! The links can only be included in the plain text.
2. Then check repository docs:
   - `docs/` folder for local documentation and implementation notes.

When answering, clearly label source provenance:
- Feishu KB
- Repository docs (`docs/`)
- Code paths (`sktime/...`)

## Response Style for OpenClaw

- Provide concise action-first answers.
- Prefer runnable commands over abstract guidance.
- When possible, include exact file paths and links.
- If uncertainty exists, explicitly say what was verified vs inferred.
