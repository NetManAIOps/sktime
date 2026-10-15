# libraryts benchmark — reproducibility leaderboard

A verification platform for the libraryts catalog: paper-reported results are
curated as **groundtruth**, re-run through the **libraryts CLI**, and compared on a
minimal web leaderboard.

```
groundtruth/*.json   curated claims: paper value + citation + exact libraryts train/run commands + tolerance
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

The page shows one unified algorithm list per task: algorithms with curated
claims come first (sorted by aggregate status — reproduced, then
deviation/error/pending), followed by every other catalog algorithm as a
gray "not run yet" row (with its runnable/disabled state in this
environment). For each curated claim: the reported value, the reproduced
value, the delta vs. tolerance, the citation, and the exact
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
the libraryts JSON envelope, compares every claimed metric against the groundtruth,
and marks the claim `reproduced` / `deviation` / `error`. Trained models are
persisted as `playground/models/bench-<claim-id>/` (gitignored, overwritten
on re-runs).

## Add a new claim

Append an entry to the matching `groundtruth/<task>.json`:

```json
{
  "id": "unique-slug",
  "algorithm": "registered-forecasting-DLinearForecaster",   // libraryts catalog id
  "algorithm_name": "DLinear",                               // display name (rows group by it)
  "dataset": "hf-etth1",                                     // libraryts dataset id
  "dataset_name": "ETTh1",                                   // display name
  "window": "horizon 96 · lookback 336",                     // config label (optional)
  "metrics": {"MSE": 0.375, "MAE": 0.399},                   // paper-reported values per metric
  "settings": "protocol as reported in the paper",
  "source": {"title": "...", "venue": "...", "url": "...", "locator": "Table 2, row X"},
  "command": "libraryts run --task ... --compact",
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
- If the paper's protocol differs structurally from libraryts' evaluation (rolling
  vs. single-origin, dataset-average vs. single series, different
  implementation), say so in `notes` and choose a tolerance that reflects it —
  a `deviation` badge with an honest note is a valid, useful outcome.

## Current coverage (502 claims, 428 reproduced)

Per task (claims / reproduced / distinct algorithms): forecasting 97/70/4, classification 163/148/7,
clustering 166/143/4, anomaly_detection 57/51/9, regression 12/11/12 (one dataset), causal 7/5/3
(one dataset). Deviations are kept visible on purpose; each entry's `notes` carries the details.

- Forecasting: NLinear/DLinear/Linear on ETTh1/h2, ETTm1/m2, Exchange, Weather, Electricity and ILI.
  DLinear/Linear on ETTh2, ETTm2 and Exchange land above the paper (trend/shift-heavy datasets; the
  paper uses validation-based early stopping, libraryts a fixed 10 epochs). Autoformer ETTh1 deviates.
  Electricity at horizon 720 was dropped: the training process is killed (out of memory).
- Classification / clustering: ROCKET and MiniRocket on 46 UCR datasets, k-Shape on 43 (tiny datasets
  swing with the random init; classifiers' official per-dataset result files other than ROCKET/MiniRocket
  were not reachable, so Arsenal / HIVE-COTE etc. are not benchmarked).
- Anomaly detection: IForest/LOF/HBOS/OCSVM on TSB-UAD Yahoo, MITDB, NAB, Dodgers, Occupancy, MGAB,
  Genesis. OCSVM on MITDB/MGAB was left out (exceeds the 30 min budget on 100k-650k-point series).
- Regression / causal: only Covid3Month (TSER) and Sachs have locally available data with published numbers.

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

Protocol support added to libraryts for paper-grade reproduction:

- **Unified train/run lifecycle** (`libraryts train` + `libraryts run --model-id`):
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
