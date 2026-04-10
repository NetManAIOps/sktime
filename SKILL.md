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

## Reference Cases (Notebook Examples)

If users ask "do we have reference cases/examples" or "I want to see some examples" or something like that, do the following:

1. Search notebook files in:
   - `lectures/`
   - `examples/`
2. Return relevant `.ipynb` paths.
3. Return **clickable** Colab links by appending notebook path to:
   - `https://colab.research.google.com/github/NetManAIOps/sktime/blob/main`

Link rule:
- If notebook is `lectures/lec5/clasp.ipynb`, return:
  - `https://colab.research.google.com/github/NetManAIOps/sktime/blob/main/lectures/lec5/clasp.ipynb` as a Colab link.

4. (Important) Open the link in the user's browser (with `openclaw browser`) if possible.

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
