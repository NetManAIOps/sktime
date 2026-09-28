---
name: labts-api
description: LabTS API — CLI version of the TSBox Sandbox Playground (NetManAIOps/sktime Time Series Sandbox) for autoresearch harnesses and automated pipelines. Use when an external harness/agent needs to discover available tasks/algorithms/datasets, run time-series experiments (forecasting, classification, regression, clustering, anomaly detection, causal discovery), re-score runs with registry metrics, estimate series parameters, compute distance matrices, or export reproduction scripts and reports programmatically — one CLI command per call, stable JSON envelope, no HTTP server.
---

# LabTS API — Playground CLI

`playground/labts.py` is the **headless CLI equivalent of the Playground web
UI**. Every web capability has a CLI counterpart, backed by the same
catalog/runner code with identical results:

| Playground web endpoint     | CLI command                                                   |
|-----------------------------|---------------------------------------------------------------|
| `GET /api/catalog`          | `labts.py catalog [--compact]`                                |
| (discovery shortcut)        | `labts.py ls tasks\|algorithms\|datasets\|preprocessors\|metrics\|models\|analyzers\|distances [--task X] [--all]` |
| `POST /api/run`             | `labts.py run --spec '<json>' [--compact] [--out run.json]`   |
|                             | `labts.py run --task forecasting --dataset airline --algorithm naive-seasonal-last --param horizon=6` |
| `GET /api/export/script`    | `labts.py script (--spec '<json>' \| --from run.json)`        |
| `GET /api/export/report`    | `labts.py report (--spec '<json>' \| --from run.json)`        |
| (fork algorithm)            | `labts.py fork <algorithm_id> [--name X]`                     |
| (validate plugin)           | `labts.py check <plugin.py>`                                  |
| (train + persist model)     | `labts.py train --algorithm <devad-detector-id> [--dataset yahoo] [--model-id X] [--param epochs=3] [--val-fraction 0.2]` |
| (detect with saved model)   | `labts.py detect --model-id X [--dataset yahoo] [--param threshold_quantile=0.99] [--out run.json]` |
| (re-score a run)            | `labts.py evaluate (--from run.json \| --spec '<json>') --metric M [--metric M2]` |
| (parameter estimation)      | `labts.py analyze --algorithm seasonality-acf --dataset airline [--param k=v]` |
| (pairwise distances)        | `labts.py dist --dataset unit-test --metric dtw [--metric scipy:cosine] [--max-instances 50]` |
| (predict, persisted model)  | `labts.py predict --model-id X [--dataset D] [--param k=v]` |

`fork` materializes any enabled catalog algorithm as an editable single-file
plugin in `playground/experiments/` (subclass scaffold with provenance
metadata); the plugin is immediately discoverable as `user-<name>`. `check`
validates the plugin contract and runs a tiny smoke experiment. See
`playground/experiments/__init__.py` for the plugin contract.

`run` is **stateless**: fit and predict happen in one call and the fitted
model is discarded. `train`/`detect` (DevAD detectors only, ids
`registered-anomaly_detection-DevAD*`) split the lifecycle: `train` persists
`model.pt` + `manifest.json` + `training.log` under
`playground/models/<model_id>/` (overwritten on re-train), `detect` reloads
the model and returns the **same result envelope as `run`** (so
`report --from` works on it). `ls models` lists persisted models.

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
  "kind": "catalog" | "result" | "train" | "fork" | "check" | "evaluate" | "analyze" | "dist" | "predict",
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
`metrics`, `analyzers`, `distances`, `compatibility`, `dependencies`, `hf`,
`meta` (~2 MB).

`--compact` (preferred for LLM harnesses): enabled entries only, drops

`compatibility` (derivable as `algorithm.task == dataset.task`), `dependencies`,
`hf`; trims algorithms to `id/name/task/subtype/params/required_params/accepts_estimators`.

- `algorithms[].id`: curated ids (`naive-seasonal-last`, `summary-random-forest`,
  `threshold-detector`, `causal-notears`) or `registered-<task>-<Name>` for
  auto-discovered sktime estimators. `enabled: false` entries carry
  `disabled_reason`.
- `algorithms[].params`: numeric scalar defaults only. **Mixed namespace** —
  per-task eval params and estimator constructor params. Split via
  `meta.eval_params`; the rest are forwarded to the estimator constructor.
  Non-numeric constructor params (str/bool/enum) are not advertised but may
  still be passed in `spec.params`.
- `algorithms[].required_params` / `accepts_estimators`: present on
  **compositors** (pipelines/ensembles) — constructors that need a
  sub-estimator. They are enabled and runnable through **nested specs**
  (see `run` below); ~80 of them (ForecastingPipeline, EnsembleForecaster,
  ColumnEnsembleForecaster, reduction/forecasting-compose wrappers, ...).
- `metrics`: the unified metric registry (`playground/metrics.py`) — every
  entry has `id/name/task/requires/direction/default/source`. `requires` ⊂
  `scores | labels | values | predictions` tells you which run payload the
  metric needs (score-based anomaly metrics need the continuous `scores`
  saved by every anomaly run).
- `analyzers`: param_est algorithms for `labts analyze`
  (SeasonalityACF, SeasonalityPeriodogram, StationarityADF, StationarityKPSS,
  ARLagOrderSelector).
- `distances`: distance ids for `labts dist` — 15 `sktime.distances`
  canonical names plus `scipy:<name>` variants via `ScipyDist`.
- `meta`: `generated_at`, `sktime_version`, `eval_params`,
  `defaults` (per-task default `dataset_id`/`algorithm_id`), `notes`.
- `datasets[].source`: `local` (built-in), `huggingface` (THU-ANM configs),
  `ucr_uea` (online archive). **No user-data upload channel** — experiments run
  on catalog datasets only.

Static mirror for browsing without paying discovery cost:
`.agent/skills/time-series-sandbox/catalog_snapshot.json`
(refresh: `python playground/catalog.py`).

## `ls` — quick discovery

`labts.py ls <section>` prints one catalog section as
`{"section", "count", "rows"}` inside the usual catalog envelope — cheaper to
eyeball than the full catalog:

- sections: `tasks`, `algorithms`, `datasets`, `preprocessors`, `metrics`,
  `models` (persisted `labts train` outputs), `analyzers`, `distances`
- `--task X`: keep entries usable for task X (matches `task` or
  `compatible_tasks`)
- `--all`: include disabled algorithms/preprocessors (default: enabled only)

## `run --spec` / `run --flags`

```json
{
  "task": "forecasting | classification | regression | clustering | anomaly_detection | causal",
  "dataset_id": "airline",
  "algorithm_id": "naive-seasonal-last",
  "preprocessors": [{"id": "registered-preprocessor-LogTransformer", "params": {}}],
  "params": {"horizon": 6, "seasonal_period": 12},
  "metrics": ["mase", "rmse"]
}
```

`--spec` accepts a JSON string, `@path/to/spec.json`, or `-` (stdin).
Equivalent flag form (no JSON needed):

```bash
labts.py run --task forecasting --dataset airline --algorithm naive-seasonal-last \
    --param horizon=6 --param seasonal_period=12 [--metric mase] \
    [--preprocessor registered-preprocessor-LogTransformer [--preprocessor ...]] \
    [--pre-param 1:key=value]
```

Both forms accept every field — all optional; omitting everything runs the
per-task default combination from `meta.defaults`. `task`, `algorithm_id`,
and `dataset_id` must agree on the same task, else `blocked`. `--param` /
`--pre-param` are repeatable `key=value` flags; numeric values are coerced.

**Metrics.** Without `--metric`, the result carries the historic per-task
default metric set (unchanged). Each `--metric <id>` (repeatable) adds a
registry metric to the result `metrics` dict — e.g. `--metric pa_f1
--metric vus_roc` for anomaly, `--metric mase` for forecasting. Every anomaly
run also saves continuous `scores` plus an `evaluation` payload
(labels/predictions) so `evaluate --from` can re-score without re-fitting.

**Preprocessing chains.** `--preprocessor` is repeatable; steps run in the
given order and each must preserve the series length / panel instance count
(a step that does not is `blocked` naming the step). Address per-step params
with the 1-based step prefix: `--pre-param 2:degree=2`. A bare
`--pre-param key=value` is accepted only with a single step (historic
behaviour). Legacy spec form `preprocessor_id` + `preprocessor_params` keeps
working and is equivalent to a one-step chain. In JSON specs prefer the
`preprocessors` list shown above.

**Nested estimator specs (compositors).** Algorithms marked
`required_params`/`accepts_estimators` take sub-estimators as nested param
values (JSON spec form only — there is no flag syntax):

```json
{
  "task": "forecasting",
  "dataset_id": "airline",
  "algorithm_id": "registered-forecasting-EnsembleForecaster",
  "params": {
    "horizon": 6,
    "forecasters": [
      {"name": "naive", "estimator": {"algorithm_id": "registered-forecasting-NaiveForecaster", "params": {"strategy": "last"}}},
      {"name": "theta", "estimator": {"algorithm_id": "registered-forecasting-ThetaForecaster"}}
    ]
  }
}
```

A dict with an `estimator` key is built recursively (`algorithm_id` or raw
`module` path + `params`); an optional `name` sibling produces the
`(name, estimator)` tuples sktime pipelines expect (`steps`, `forecasters`,
...). The sub-estimator's task must match the parent (transformers are allowed
everywhere) — mismatches are `blocked`. `registered-<task>-<Class>` ids of
curated classes (e.g. `registered-forecasting-NaiveForecaster`) resolve to the
curated entry's class. Exported `script` renders nested specs as real Python.

Result `data` (full): `status`, `run_id`, `spec` (normalized), `task`,
`dataset`, `algorithm`, `preprocessor`, `preprocessors`, `duration_ms`, `log`,
`metrics`, `series` (plot points), `tables`, `summary`, `code` (self-contained
reproduction script), `report` (Markdown); anomaly runs add `scores` +
`evaluation`, causal runs add `graph`.

- `--compact`: drops `series`/`tables`/`code`/`report`/`scores`/`evaluation`
  on stdout — keeps `metrics`, `summary`, `log`, `spec`, `run_id`,
  `duration_ms`. The `--out` file always gets the full result.
- `--out FILE`: also saves the **full** result envelope (never compacted) for
  later `script --from` / `report --from` / `evaluate --from`.

Metrics by task (defaults): forecasting → `MAE`/`MSE`/`MAPE`; classification →
`Accuracy`/`Macro F1`; regression → `MAE`/`RMSE`/`R²`; clustering →
`ARI`/`NMI`/`Clusters`/`Largest Cluster`; anomaly_detection →
`Precision`/`Recall`/`F1`/`Detected`/`Ground Truth`; causal → `SHD`/
`Edge Precision`/`Edge Recall`/`Edge F1`/`Edges`/`True Edges`.

## `script` / `report`

Export the generated reproduction script (`code`) or Markdown report
(`report`) as raw text — the CLI counterpart of the web UI's export links.

- `--from run.json`: export from a result saved with `run --out`. **No
  re-run** — this is the pipeline-friendly path (run once, export many).
- `--spec ...`: runs the experiment fresh, then exports. Convenient for
  one-shot use; costs a full run.

A `--from` file produced by `run --compact` **stdout** (not `--out`) lacks the
export payloads and fails with exit 2 — always use `--out` for export chains.

## `train` / `detect` — persistent DevAD models

Only for the 17 DevAD detectors (`labts.py ls algorithms --task
anomaly_detection`, names `DevAD*`); other detectors are not trainable — use
`run` directly (`train` exits `blocked` with a clear message).

```bash
# train once (persists playground/models/fits-v1/: model.pt, manifest.json, training.log)
labts.py train --algorithm registered-anomaly_detection-DevADFITSDetector \
    --dataset yahoo --model-id fits-v1 --param epochs=3 --val-fraction 0.2

# detect many (same envelope as run; --compact/--out supported)
labts.py detect --model-id fits-v1 --dataset yahoo --param threshold_quantile=0.99 --out det.json
labts.py report --from det.json
```

- `--model-id` defaults to `<family>-<dataset>`; re-training the same id
  overwrites it.
- `--param` keys: adapter params `win_len`, `epochs`, `batch_size`, `seed`,
  `device`, `threshold_quantile` (detect only) go to the adapter; **any other
  key is forwarded as a DevAD hyperparameter** (validated against the family's
  `HP` table — unknown keys fail `blocked` with the list of valid names).
  Non-scalar HPs can be passed as `--param params='{"h_dim": 64}'`.
- `--val-fraction F` holds out the series tail for validation, enabling early
  stopping for torch families.
- `train` result: `model_id`, `family`, resolved `params`, `model_dir`,
  `duration_ms`, `next_steps`. `detect` result: identical in shape to `run`
  (`metrics`/`series`/`tables`/`code`/`report`), with
  `spec.algorithm_id = "devad-trained:<model-id>"`.

## `evaluate` — re-score a run

```bash
# from a saved result — no re-fit; anomaly runs are re-scored from the saved
# continuous `scores` field (plus the `evaluation` labels/predictions payload)
labts.py evaluate --from run.json --metric pa_f1 --metric vus_roc

# from a spec — re-runs the experiment with the requested metrics
labts.py evaluate --spec '{"task": "forecasting", "dataset_id": "airline"}' --metric mase
```

- `--from` works for anomaly_detection (via `scores`), forecasting (forecast
  table + train series), and regression/classification/clustering (full
  series points) — whenever the `run --out` payload carries the required
  inputs. A file captured from `--compact` **stdout** lacks those payloads and
  is `blocked` with a pointer to `--spec`; always use `--out` for evaluate
  chains.
- Score-based anomaly metrics (AUC-ROC/AP/Point-F1/PA-F1/Affiliation-F1/
  Delay-F1/VUS-PR/VUS-ROC) need `scores`; label/point metrics (Precision,
  F1, Windowed F1, Rand Index, ...) work from labels+predictions.
- Output `data`: `task`, `run_id`, `metric_ids`, `metrics`, `source`
  (`saved run payload (no re-fit)` vs `fresh run (re-fit)`).

## `analyze` — parameter estimation (no evaluation stage)

```bash
labts.py analyze --algorithm seasonality-acf --dataset airline
labts.py analyze --algorithm stationarity-kpss --dataset yahoo
labts.py analyze --algorithm ar-lag-order --dataset airline --param maxlag=8
```

Fits a `sktime.param_est` estimator on a catalog series (forecasting or
anomaly_detection dataset) and prints the fitted estimates as JSON
(`data.estimates`, e.g. `sp`, `sp_significant`, `stationary`, `pvalue`,
`selected_model`). Analyzer ids: `ls analyzers`.

## `dist` — pairwise distance matrices

```bash
labts.py dist --dataset unit-test --metric dtw --metric scipy:cosine --max-instances 40
labts.py dist --dataset arrow-head --metric wdtw --param window=0.1
```

Computes the pairwise distance matrix over the dataset's **train** instances
(panel datasets: classification/regression/clustering) via
`sktime.distances.pairwise_distance` (15 canonical ids: `euclidean`,
`squared`, `dtw`, `ddtw`, `wdtw`, `wddtw`, `erp`, `edr`, `lcss`, `msm`,
`twe`, `sbd`, `smets`, `dot`, `granger`) or `scipy:<name>` via
`sktime.dists_kernels.ScipyDist` (`euclidean`, `sqeuclidean`, `cityblock`,
`chebyshev`, `canberra`, `braycurtis`, `cosine`, `correlation`, `minkowski`,
`hamming`, `jaccard`). `--param` applies to every requested metric (e.g.
`window=0.1` for dtw, `p=3` for scipy:minkowski). Output `data.results[]`:
`metric`, `shape`, `symmetric`, `min/max/mean` (off-diagonal), full `matrix`.

## `predict` — persisted-model prediction

```bash
labts.py predict --model-id <id> [--dataset D] [--param k=v] [--out run.json]
```

Calls `trainer.predict_estimator(model_id, dataset_id, params)` — the generic
persistence backend (merged on main): it reloads a persisted model and
evaluates it on the dataset's holdout split, returning the **same result
envelope as `run`** (so `report --from` / `evaluate --from` work on it).
DevAD models delegate to `detect_devad` (the `detect` alias); sktime models
reload from `model.zip` and are scored against the manifest's eval params
(`--param` overrides them, e.g. `--param horizon=24`). Unknown model ids are
`blocked` (exit 3) — see `ls models` for what is persisted.

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

# 6. Causal discovery on a bnlearn benchmark (graph metrics vs true DAG)
python playground/labts.py run --compact --task causal --dataset causal-sachs \
    --algorithm causal-notears --param max_samples=1500

# 7. Anomaly run + re-score with DevAD metrics (no re-fit)
python playground/labts.py run --task anomaly_detection --dataset yahoo \
    --algorithm registered-anomaly_detection-DevADSubPCADetector \
    --param win_len=8 --metric pa_f1 --metric vus_roc --out det.json --compact
python playground/labts.py evaluate --from det.json --metric affiliation_f1

# 8. Chained preprocessors with per-step params
python playground/labts.py run --task forecasting --dataset airline --param horizon=6 \
    --preprocessor registered-preprocessor-LogTransformer \
    --preprocessor registered-preprocessor-Detrender --pre-param 2:degree=1 --compact

# 9. Parameter estimation + distances
python playground/labts.py analyze --algorithm seasonality-periodogram --dataset airline
python playground/labts.py dist --dataset gunpoint --metric dtw --max-instances 30
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
  curated anomaly detector; `max_samples`, `seed` for causal) never reach the
  estimator constructor — see `meta.eval_params`.
- The curated `threshold-detector` detrends with a rolling median and
  thresholds the residual z-score, not the raw series; its continuous
  `scores` are the absolute z-scores.
- Nested estimator specs (compositors) exist only in JSON spec form —
  `--param` flags cannot express them. Curated-class registered ids
  (`registered-forecasting-NaiveForecaster`) resolve to the curated class.
- `--pre-param` needs the `STEP:` prefix as soon as more than one
  `--preprocessor` step is given.
- `evaluate --from` needs the full `--out` payload (same rule as
  `script --from`); anomaly re-scoring reads the saved `scores`, other tasks
  read saved series/tables. When in doubt, use `evaluate --spec` (re-runs).
- Requesting a metric whose `requires` inputs the run does not produce (e.g.
  the DevAD reconstruction metrics `predict_error`/`output_mae`, which no
  runner emits yet) is `blocked` with a clear reason, never a crash.
- Export commands re-run the experiment when given `--spec`; deterministic
  specs give deterministic scripts, but `duration_ms`/`run_id` will differ.
  Use `--out` + `--from` when the export must match the run exactly.

## Related

- Playground HTTP UI (interactive, browser): `python playground/server.py` —
  same backends, see skill `time-series-sandbox`.
- Repo-native usage without the playground layer: `sktime/...` APIs directly,
  skill `time-series-sandbox`.
