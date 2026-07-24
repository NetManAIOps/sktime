---
name: labts-api
description: LabTS API — CLI version of the TSBox Sandbox Playground (NetManAIOps/sktime Time Series Sandbox) for autoresearch harnesses and automated pipelines. Use when an external harness/agent needs to discover available tasks/algorithms/datasets, run time-series experiments (forecasting, classification, anomaly detection), or export reproduction scripts and reports programmatically — one CLI command per call, stable JSON envelope, no HTTP server.
---

# LabTS API — Playground CLI

`playground/labts.py` is the **headless CLI equivalent of the Playground web
UI**. Every web capability has a CLI counterpart, backed by the same
catalog/runner code with identical results:

| Playground web endpoint     | CLI command                                                   |
|-----------------------------|---------------------------------------------------------------|
| `GET /api/catalog`          | `labts.py catalog [--compact]`                                |
| `POST /api/run`             | `labts.py run --spec '<json>' [--compact] [--out run.json]`   |
| `GET /api/export/script`    | `labts.py script (--spec '<json>' \| --from run.json)`        |
| `GET /api/export/report`    | `labts.py report (--spec '<json>' \| --from run.json)`        |

One process per call, no HTTP server, `run_id` session state not needed.
Run from the repository root with the repo venv:

```bash
.venv/bin/python playground/labts.py <command>
```

(any Python 3.10+ with sktime + soft deps works; `.venv` is the reference
environment, see `playground/requirements.txt`)

## Output contract

`catalog` and `run` print a **JSON envelope on stdout** (always, incl. errors):

```json
{
  "api": "labts",
  "api_version": "1.0",
  "kind": "catalog" | "result",
  "status": "ok" | "blocked" | "error",
  "data": { ... } | null,
  "error": null | "message"
}
```

`script` and `report` print **raw text** (Python / Markdown) on stdout for
direct redirection (`> experiment.py`); on failure stdout stays empty and the
JSON envelope goes to **stderr**.

| Exit code | status    | Meaning                                                        |
|-----------|-----------|----------------------------------------------------------------|
| 0         | `ok`      | Success.                                                       |
| 3         | `blocked` | Expected domain error: disabled/unknown algorithm or dataset, incompatible task combination, missing soft dependency, estimator fit failure. Record and move on; never retry the identical spec. |
| 1         | `error`   | Unexpected internal error.                                     |
| 2         | `error`   | Usage error: malformed JSON spec, unreadable/invalid `--from` file, empty export payload. |

Harness rule: parse stdout (stderr for `script`/`report`) as JSON, branch on
`status`; use the exit code as the process-level signal. Library warnings go to
stderr and never pollute stdout JSON. Non-finite floats are sanitized to `null`.

## `catalog`

Full form (default) keys: `tasks`, `algorithms`, `preprocessors`, `datasets`,
`metrics`, `compatibility`, `dependencies`, `hf`, `meta` (~1.2 MB).

`--compact` (~60 KB, preferred for LLM harnesses): enabled entries only, drops
`compatibility` (derivable as `algorithm.task == dataset.task`), `dependencies`,
`hf`; trims algorithms to `id/name/task/subtype/params`.

- `algorithms[].id`: curated ids (`naive-seasonal-last`, `summary-random-forest`,
  `threshold-detector`) or `registered-<task>-<Name>` for auto-discovered
  sktime estimators. `enabled: false` entries carry `disabled_reason`.
- `algorithms[].params`: numeric scalar defaults only. **Mixed namespace** —
  per-task eval params and estimator constructor params. Split via
  `meta.eval_params`; the rest are forwarded to the estimator constructor.
  Non-numeric constructor params (str/bool/enum) are not advertised but may
  still be passed in `spec.params`.
- `meta`: `generated_at`, `sktime_version`, `eval_params`,
  `defaults` (per-task default `dataset_id`/`algorithm_id`), `notes`.
- `datasets[].source`: `local` (built-in), `huggingface` (THU-ANM configs),
  `ucr_uea` (online archive). **No user-data upload channel** — experiments run
  on catalog datasets only.

Static mirror for browsing without paying discovery cost:
`.agent/skills/time-series-sandbox/catalog_snapshot.json`
(refresh: `python playground/catalog.py`).

## `run --spec`

```json
{
  "task": "forecasting | classification | anomaly_detection",
  "dataset_id": "airline",
  "algorithm_id": "naive-seasonal-last",
  "preprocessor_id": "none",
  "params": {"horizon": 6, "seasonal_period": 12},
  "preprocessor_params": {}
}
```

All fields optional — omitting everything runs the per-task default combination
from `meta.defaults`. `task`, `algorithm_id`, and `dataset_id` must agree on
the same task, else `blocked`. `--spec` also accepts `@path/to/spec.json` or
`-` (stdin).

Result `data` (full): `status`, `run_id`, `spec` (normalized), `task`,
`dataset`, `algorithm`, `preprocessor`, `duration_ms`, `log`, `metrics`,
`series` (plot points), `tables`, `summary`, `code` (self-contained
reproduction script), `report` (Markdown).

- `--compact`: drops `series`/`tables`/`code`/`report` on stdout — keeps
  `metrics`, `summary`, `log`, `spec`, `run_id`, `duration_ms`.
- `--out FILE`: also saves the **full** result envelope (never compacted) for
  later `script --from` / `report --from`.

Metrics by task: forecasting → `MAE`/`MSE`/`MAPE`; classification →
`Accuracy`/`Macro F1`; anomaly_detection → `Precision`/`Recall`/`F1`/
`Detected`/`Ground Truth`.

## `script` / `report`

Export the generated reproduction script (`code`) or Markdown report
(`report`) as raw text — the CLI counterpart of the web UI's export links.

- `--from run.json`: export from a result saved with `run --out`. **No
  re-run** — this is the pipeline-friendly path (run once, export many).
- `--spec ...`: runs the experiment fresh, then exports. Convenient for
  one-shot use; costs a full run.

A `--from` file produced by `run --compact` **stdout** (not `--out`) lacks the
export payloads and fails with exit 2 — always use `--out` for export chains.

## Examples

```bash
# 1. What can I do? (small)
python playground/labts.py catalog --compact | python -m json.tool

# 2. Default forecasting baseline (fast: curated algorithms skip discovery)
python playground/labts.py run --compact --spec '{}'

# 3. Registered estimator, eval + constructor params mixed
python playground/labts.py run --compact --spec '{
  "task": "forecasting",
  "dataset_id": "airline",
  "algorithm_id": "registered-forecasting-PolynomialTrendForecaster",
  "params": {"horizon": 4, "degree": 2}
}'

# 4. Anomaly detection with tuned eval threshold
python playground/labts.py run --compact --spec '{
  "task": "anomaly_detection",
  "dataset_id": "yahoo",
  "algorithm_id": "threshold-detector",
  "params": {"threshold": 2.5, "window": 24}
}'

# 5. Pipeline: run once, export script + report (web "export" buttons)
python playground/labts.py run --spec @spec.json --out run.json --compact
python playground/labts.py script --from run.json > experiment.py
python playground/labts.py report --from run.json > experiment.md
```

## Performance and cost notes (verified 2026-07)

- First registry discovery in a fresh process costs ~18–25 s (walks
  `sktime.registry.all_estimators`, test-constructs every estimator). Paid by
  **every** `catalog` call and every run/export whose `algorithm_id` is a
  `registered-*` id.
- `run` with a **curated** algorithm id skips discovery (~2–5 s total). Prefer
  curated ids in tight harness loops when the baseline suffices.
- Each CLI call is a fresh process: no cross-call cache. Call `catalog
  --compact` once, keep it, then issue `run` calls.
- Budget ~30 s per registered-algorithm run; online datasets (HF/UCR) add
  download time on first use.
- Full catalog includes a live Hugging Face metadata request (8 s timeout,
  falls back to a default config list offline).

## Gotchas

- `params` values are coerced to the type of the registered default
  (int/float/bool); pass numbers, not strings.
- Eval params (`horizon`, `context_window`; `threshold`, `window` for the
  curated anomaly detector) never reach the estimator constructor — see
  `meta.eval_params`.
- The curated `threshold-detector` detrends with a rolling median and
  thresholds the residual z-score, not the raw series.
- Export commands re-run the experiment when given `--spec`; deterministic
  specs give deterministic scripts, but `duration_ms`/`run_id` will differ.
  Use `--out` + `--from` when the export must match the run exactly.

## Related

- Playground HTTP UI (interactive, browser): `python playground/server.py` —
  same backends, see skill `time-series-sandbox`.
- Repo-native usage without the playground layer: `sktime/...` APIs directly,
  skill `time-series-sandbox`.
