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

## Datasets: Where to Look

If users ask "what datasets are available", point to:

1. Local built-in datasets under `sktime/datasets/data`, including folders such as:
   - `ACSF1`, `Airline`, `ArrowHead`, `BasicMotions`, `ChickenEgg`, `Covid3Month`,
     `DailyDelhiClimate`, `GunPoint`, `ItalyPowerDemand`, `JapaneseVowels`, `Longley`,
     `Lynx`, `OSULeaf`, `PBS_dataset`, `PLAID`, `ShampooSales`, `Tecator`, `UnitTest`,
     `Uschange`, `etth_display_W112`, `mitdb`, `oil`, `seatbelts`, `segmentation`,
     `solar`, `yahoo`
2. External dataset source:
   - https://huggingface.co/datasets/Skyoung13/THU-ANM-DATASET

Advise users to inspect both sources for coverage and updated versions.

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
