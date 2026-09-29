# labts benchmark — reproducibility leaderboard

A verification platform for the labts catalog: paper-reported results are
curated as **groundtruth**, re-run through the **labts CLI**, and compared on a
minimal web leaderboard.

```
groundtruth/*.json   curated claims: paper value + citation + exact labts train/run commands + tolerance
results/*.json       reproduction records, written by reproduce.py (one per claim)
state.json           merged view served at /api/benchmark (generated)
expand_commands.py   regenerates the train/run command pairs (all params explicit)
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
reproduced value, the delta vs. tolerance, the citation, and the exact
**train + run command pair** (every parameter, including catalog defaults,
is an explicit `--param` flag; `run --model-id` evaluates the persisted
model without refitting).

## Reproduce

```bash
python playground/benchmark/reproduce.py               # run pending claims
python playground/benchmark/reproduce.py --all         # re-run everything
python playground/benchmark/reproduce.py --entry ID    # one claim
python playground/benchmark/build_state.py             # refresh state.json
```

`reproduce.py` executes each claim's `train_command` then `run_command` with
the current interpreter (`python` in the recorded commands is a placeholder —
use an env with the soft dependencies installed, e.g. `.venv-full`), parses
the labts JSON envelope, compares every claimed metric against the groundtruth,
and marks the claim `reproduced` / `deviation` / `error`. Trained models are
persisted as `playground/models/bench-<claim-id>/` (gitignored, overwritten
on re-runs).

## Add a new claim

Append an entry to the matching `groundtruth/<task>.json`:

```json
{
  "id": "unique-slug",
  "algorithm": "registered-forecasting-DLinearForecaster",   // labts catalog id
  "algorithm_name": "DLinear",                               // display name (rows group by it)
  "dataset": "hf-etth1",                                     // labts dataset id
  "dataset_name": "ETTh1",                                   // display name
  "window": "horizon 96 · lookback 336",                     // config label (optional)
  "metrics": {"MSE": 0.375, "MAE": 0.399},                   // paper-reported values per metric
  "settings": "protocol as reported in the paper",
  "source": {"title": "...", "venue": "...", "url": "...", "locator": "Table 2, row X"},
  "command": "python playground/labts.py run --task ... --compact",
  "tolerance": {"type": "relative", "value": 0.15},
  "notes": "protocol gaps, granularity caveats, why deviation is expected"
}
```

Then generate the train/run pair (this replaces `command` with
`train_command`/`run_command` and keeps your explicit params in
`spec_params` for idempotent re-generation):

```bash
python playground/benchmark/expand_commands.py
```

You can also author `spec_params` (a dict of just the non-default params)
directly instead of `command` and skip the one-shot form entirely.

On the page, claims are grouped by `algorithm_name`: the parent row shows the
aggregate (`k/n configs reproduced`), each dataset/window config is a sub-row
with per-metric `paper→re-run ✓/✗` results, expandable to the exact commands,
citation, protocol, and notes.

Rules of thumb:

- Only record values you actually saw in the source (paper table, official
  results CSV/page, official README). Put the exact locator in `source.locator`.
- Keep the recorded commands runnable end-to-end; the judged metrics must be
  in the default result payload (no `--metric` flags needed on `run_command`).
- If the paper's protocol differs structurally from labts' evaluation (rolling
  vs. single-origin, dataset-average vs. single series, different
  implementation), say so in `notes` and choose a tolerance that reflects it —
  a `deviation` badge with an honest note is a valid, useful outcome.

## Current smoke coverage (24 claims, 20 reproduced)

Remaining deviations (each entry's `notes` carries the details):

- Autoformer ETTh1 96/336/720 (+28%/+27%/+34%): our CPU reimplementation with
  default hyperparameters converges above the paper's GPU-tuned number
  (the paper value is quoted from the FEDformer paper's own runs);
  horizon 192 lands within tolerance.
- IForest on YAHOO (+0.196): our window embedding/normalization differs from
  TSB-UAD's reference pipeline (FFT-based window length); LOF and MITDB
  still land within tolerance.

| task | claims | source |
| --- | --- | --- |
| forecasting | DLinear ETTh1/Weather × horizons {96,192,336,720} (MSE+MAE), Autoformer ETTh1 × {96,192,336,720} | Zeng et al., AAAI 2023, Table 2 |
| classification | ROCKET ItalyPowerDemand / ArrowHead accuracy | Dempster et al., DMKD 2020, official results CSV |
| regression | Rocket / RandomForest Covid3Month RMSE | Tan et al., DMKD 2021 (TSER archive results) |
| clustering | k-Shape ArrowHead / GunPoint ARI+NMI | Paparrizos & Gravano, SIGMOD 2015, official Matlab repo |
| anomaly_detection | LOF / IForest Yahoo AUC-ROC, IForest MITDB AUC-ROC | Paparrizos et al., PVLDB 2022 (TSB-UAD) |
| causal | NOTEARS / PC / GES Sachs continuous SHD | Zheng et al., NeurIPS 2018; Petersen, arXiv:2412.10039 |

The Sachs continuous dataset (`causal-sachs-cont`, vendored under
`sktime/datasets/data/sachs_continuous/` from
[cmu-phil/example-causal-datasets](https://github.com/cmu-phil/example-causal-datasets))
exists to make the causal claims exactly reproducible: same n=7466 data and
same 20-edge consensus DAG as the NOTEARS paper.

Protocol support added to labts for paper-grade reproduction:

- **Unified train/run lifecycle** (`labts train` + `labts run --model-id`):
  every claim's commands use the train-once/run-many pair — `train` fits and
  persists `playground/models/bench-<id>/` (algorithm params + eval split +
  fitted preprocessor in `manifest.json`), `run --model-id` reloads and scores
  the holdout with no refit. Protocol parity with the one-shot `run` is
  covered for rolling forecasting, clustering `fit_on=all`, causal discovery,
  and per-series multi-series anomaly bundles (see `playground/test_trainer.py`).
- **Rolling-origin forecasting eval**: `--param eval_mode=rolling` trains once
  and slides a (context → horizon) window over the test region without refitting
  (tslib adapters expose `predict_windows`). `--param train_fraction=`,
  `--param test_fraction=`, and `--param test_start_fraction=` /
  `--param test_end_fraction=` reproduce exact paper splits — e.g. the ETT
  protocol evaluates rows 11520–14400, dropping the volatile tail
  (`train_fraction=0.496 test_start_fraction=0.661 test_end_fraction=0.827`).
- **Multi-series anomaly detection**: the `tsb-yahoo` (367 series) and
  `tsb-mitdb` (32 series) datasets score every series in a TSB-UAD directory
  and average per-series metrics (dataset-level protocol). Data: download
  `TSB-UAD-Public.zip` from thedatum.org into
  `~/.cache/tsbox-sandbox-playground/tsb-uad` (or set `TSB_UAD_HOME`).

Fixed while debugging (see git history):

- `sktime/base/adapters/_tslearn.py` fed tslearn `(n, d, sz)` arrays without
  transposing to `(n, sz, d)` — every tslearn-backed estimator silently
  degenerated (KShape collapsed to one cluster). Fixed, and the two clustering
  tests' hard-coded expectations (generated with the bug) were regenerated.
- clustering evaluations now support `--param fit_on=all` (fused train+test,
  the no-held-out protocol used by clustering papers); the default fit-train /
  predict-test path is unchanged.
- `sktime/forecasting/tslib.py` crashed on non-datetime indexes (`None` time
  marks); zero marks are used instead.
- `sktime/forecasting/tslib.py` also sliced decoder windows with negative
  indices when `seq_len < label_len` (short series crashed with empty
  tensors); early windows are now left-zero-padded.
