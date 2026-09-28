#!/usr/bin/env python3
"""LabTS API — CLI version of the TSBox Sandbox Playground, for automated harnesses.

One process per call, no HTTP server required. Every Playground web capability
has a CLI counterpart (same catalog/runner code, identical results):

    Web endpoint                  CLI command
    ----------------------------  ------------------------------------------
    GET  /api/catalog             python playground/labts.py catalog [--compact]
    POST /api/run                 python playground/labts.py run --spec '<json>' [--compact] [--out run.json]
                                  python playground/labts.py run --task forecasting --dataset airline \
                                      --algorithm naive-seasonal-last --param horizon=6
    GET  /api/export/script       python playground/labts.py script (--spec '<json>' | --from run.json)
    GET  /api/export/report       python playground/labts.py report (--spec '<json>' | --from run.json)

    Discovery shortcut            python playground/labts.py ls tasks|algorithms|datasets|preprocessors|metrics|models|analyzers|distances \
                                      [--task forecasting] [--all]

    Persistent models (DevAD)     python playground/labts.py train --algorithm <devad-detector-id> \
                                      --dataset yahoo --model-id my-model [--param epochs=3] [--val-fraction 0.2]
                                  python playground/labts.py detect --model-id my-model [--dataset yahoo] \
                                      [--param threshold_quantile=0.99] [--out run.json]

    Re-score a run                python playground/labts.py evaluate --from run.json --metric pa_f1 [--metric vus_roc]
                                  python playground/labts.py evaluate --spec '<json>' --metric mase   (re-runs)

    Parameter estimation          python playground/labts.py analyze --algorithm seasonality-acf --dataset airline

    Pairwise distances            python playground/labts.py dist --dataset unit-test --metric dtw [--metric scipy:cosine]

    Predict (persisted model)     python playground/labts.py predict --model-id M [--dataset D] [--param k=v]
                                  (generic backend from mission M2; blocked with a hint until it lands)

`--spec` accepts a JSON string, `@path/to/spec.json`, or `-` for stdin.
`run` also takes plain flags (`--task/--dataset/--algorithm`, repeatable
`--preprocessor`, `--metric`, plus repeatable `--param key=value` /
`--pre-param [STEP:]key=value`) so no JSON is needed for common calls;
omitting everything runs the per-task default.

`run` is stateless (fit+predict in one call, model discarded). For trainable
DevAD detectors, `train` persists the model under `playground/models/<id>/`
and `detect` reuses it, returning the same result envelope as `run`.

`catalog` and `run` print a JSON envelope on stdout (always, also for errors):

    {"api": "labts", "api_version": "1.0", "kind": "catalog" | "result",
     "status": "ok" | "blocked" | "error", "data": {...} | null,
     "error": null | "message"}

`script` and `report` print raw text (Python / Markdown) on stdout for direct
redirection (`> experiment.py`); on failure stdout stays empty and the JSON
envelope goes to stderr.

Exit codes: 0 = ok, 3 = blocked (domain error, e.g. disabled algorithm or
missing soft dependency), 1 = unexpected error, 2 = usage error.

Typical pipeline flow (run once, export many):

    python playground/labts.py run --spec @spec.json --out run.json
    python playground/labts.py script --from run.json > experiment.py
    python playground/labts.py report --from run.json > experiment.md
    python playground/labts.py evaluate --from run.json --metric pa_f1
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from catalog import EVAL_PARAMS, REPO_ROOT, _sanitize_for_json, build_catalog, get_enabled_algorithm  # noqa: E402
from runners import PlaygroundError, _normalize_spec, run_experiment  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

API = "labts"
API_VERSION = "1.0"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3

# Large, presentation-oriented result fields dropped by `run --compact`.
_RUN_HEAVY_KEYS = ("series", "tables", "code", "report", "scores", "evaluation")
# Result key backing each export command.
_EXPORT_KEYS = {"script": "code", "report": "report"}


class UsageError(Exception):
    """Invalid CLI usage or malformed spec/payload."""


def _meta() -> dict:
    """Contract metadata: versions, eval-param whitelist, per-task defaults."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        sktime_version = version("sktime")
    except PackageNotFoundError:
        sktime_version = None
    defaults = {}
    for task in EVAL_PARAMS:
        normalized = _normalize_spec({"task": task})
        defaults[task] = {
            "dataset_id": normalized["dataset_id"],
            "algorithm_id": normalized["algorithm_id"],
        }
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sktime_version": sktime_version,
        "eval_params": {task: sorted(keys) for task, keys in EVAL_PARAMS.items()},
        "defaults": defaults,
        "notes": [
            "Algorithm `params` mix eval params (per task, see meta.eval_params) "
            "and estimator constructor params; eval params configure the "
            "experiment, the rest are forwarded to the estimator constructor.",
            "Only numeric scalar constructor params are advertised; other "
            "constructor params (str/bool/enum) may still be passed in "
            "spec.params and are forwarded as-is.",
            "Compatibility is derivable client-side: algorithm.task == dataset.task.",
        ],
    }


def _compact_catalog(data: dict) -> dict:
    """Enabled entries only; drop the compatibility cross-product and env info."""
    keep_algo = ("id", "name", "task", "subtype", "params", "required_params", "accepts_estimators")
    keep_prep = ("id", "name", "compatible_tasks", "params")
    keep_ds = ("id", "name", "task", "source", "online", "default")
    return {
        "meta": data["meta"],
        "tasks": data["tasks"],
        "algorithms": [
            {k: a[k] for k in keep_algo if k in a}
            for a in data["algorithms"]
            if a.get("enabled")
        ],
        "preprocessors": [
            {k: p[k] for k in keep_prep if k in p}
            for p in data["preprocessors"]
            if p.get("enabled")
        ],
        "datasets": [
            {k: d[k] for k in keep_ds if k in d} for d in data["datasets"]
        ],
        "metrics": data["metrics"],
        "analyzers": data.get("analyzers", []),
        "distances": data.get("distances", []),
    }


def _compact_result(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in _RUN_HEAVY_KEYS}


def _load_spec(arg: str) -> dict:
    if arg == "-":
        text = sys.stdin.read()
    elif arg.startswith("@"):
        try:
            text = Path(arg[1:]).read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"Cannot read spec file: {exc}") from exc
    else:
        text = arg
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageError(f"Invalid JSON spec: {exc}") from exc
    if not isinstance(spec, dict):
        raise UsageError("Spec must be a JSON object.")
    return spec


def _load_result_file(path: str) -> dict:
    """Load `data` from a run envelope written by `run --out`."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(f"Cannot read result file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(f"Result file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise UsageError(f"File does not look like a LabTS run result: {path}")
    if payload.get("status") != "ok":
        raise UsageError(
            f"Run did not succeed (status={payload.get('status')}); nothing to export."
        )
    return payload["data"]


def _envelope(kind: str) -> dict:
    return {
        "api": API,
        "api_version": API_VERSION,
        "kind": kind,
        "status": "ok",
        "data": None,
        "error": None,
    }


def _write_json(path: str, payload: dict) -> None:
    try:
        Path(path).write_text(
            json.dumps(_sanitize_for_json(payload), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise UsageError(f"Cannot write --out file: {exc}") from exc


def _emit_json(stream, envelope: dict) -> None:
    stream.write(json.dumps(_sanitize_for_json(envelope), ensure_ascii=False) + "\n")


def _fail(envelope: dict, stream, exc: Exception, status: str, exit_code: int) -> int:
    envelope.update(status=status, error=str(exc))
    _emit_json(stream, envelope)
    return exit_code


def _main_catalog(args) -> int:
    envelope = _envelope("catalog")
    try:
        data = build_catalog(include_registered=True)
        data["meta"] = _meta()
        envelope["data"] = _compact_catalog(data) if args.compact else data
        exit_code = EXIT_OK
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_run(args) -> int:
    envelope = _envelope("result")
    try:
        spec = _load_spec(args.spec) if args.spec else _spec_from_flags(args)
        data = run_experiment(spec)
        if args.out:
            _write_json(args.out, {**envelope, "data": data})
        envelope["data"] = _compact_result(data) if args.compact else data
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _parse_kv_pairs(pairs: list[str] | None, what: str) -> dict:
    """Parse repeatable `key=value` flags into a dict.

    Values are coerced to int/float when they look numeric, matching the
    estimator-param coercion in the runners.
    """
    params: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise UsageError(f"Invalid {what} `{pair}`; expected key=value.")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise UsageError(f"Invalid {what} `{pair}`; empty key.")
        params[key] = _coerce_cli_value(value.strip())
    return params


def _coerce_cli_value(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_step_kv_pairs(pairs: list[str] | None, n_steps: int, what: str) -> list[dict]:
    """Parse repeatable `--pre-param` flags into per-step param dicts.

    Per-step addressing: `STEP:key=value` where STEP is the 1-based index of
    the `--preprocessor` occurrence it belongs to. A bare `key=value` (no
    prefix) is accepted only with a single step — the historic behaviour.
    """
    per_step: list[dict] = [dict() for _ in range(max(1, n_steps))]
    for pair in pairs or []:
        if "=" not in pair:
            raise UsageError(f"Invalid {what} `{pair}`; expected [STEP:]key=value.")
        head, _, value = pair.partition("=")
        head = head.strip()
        step_idx = 0
        key = head
        if ":" in head:
            step_s, _, key = head.partition(":")
            try:
                step_idx = int(step_s) - 1
            except ValueError:
                raise UsageError(
                    f"Invalid {what} `{pair}`; STEP must be a 1-based step index."
                ) from None
            if not 0 <= step_idx < n_steps:
                raise UsageError(
                    f"Invalid {what} `{pair}`; step {step_s} out of range "
                    f"(1..{n_steps} for the given --preprocessor steps)."
                )
        elif n_steps > 1:
            raise UsageError(
                f"Invalid {what} `{pair}`; with multiple --preprocessor steps, "
                "address params per step as `STEP:key=value` (e.g. `2:degree=2`)."
            )
        key = key.strip()
        if not key:
            raise UsageError(f"Invalid {what} `{pair}`; empty key.")
        per_step[step_idx][key] = _coerce_cli_value(value.strip())
    return per_step


def _spec_from_flags(args) -> dict:
    """Build a run spec from --task/--dataset/... flags instead of --spec."""
    preprocessors = list(args.preprocessor or [])
    spec = {
        "task": args.task,
        "dataset_id": args.dataset,
        "algorithm_id": args.algorithm,
        "params": _parse_kv_pairs(args.param, "--param"),
    }
    if getattr(args, "metric", None):
        spec["metrics"] = list(args.metric)
    if preprocessors:
        per_step = _parse_step_kv_pairs(args.pre_param, len(preprocessors), "--pre-param")
        spec["preprocessors"] = [
            {"id": pid, "params": per_step[i]} for i, pid in enumerate(preprocessors)
        ]
    elif args.pre_param:
        # legacy tolerant path: params without a preprocessor are ignored
        spec["preprocessor_params"] = _parse_kv_pairs(args.pre_param, "--pre-param")
    return {k: v for k, v in spec.items() if v not in (None, {})}


_LS_SECTIONS = (
    "tasks",
    "algorithms",
    "datasets",
    "preprocessors",
    "metrics",
    "models",
    "analyzers",
    "distances",
)


def _main_ls(args) -> int:
    """Quick resource listing, a filtered view of the catalog."""
    envelope = _envelope("catalog")
    try:
        if args.section == "models":
            from trainer import list_trained_models

            rows = list_trained_models()
        else:
            data = build_catalog(include_registered=True)
            rows = data[args.section]
        if args.section in ("algorithms", "preprocessors") and not args.all:
            rows = [row for row in rows if row.get("enabled")]
        if args.task:
            rows = [
                row
                for row in rows
                if row.get("task") in (args.task, "all")
                or args.task in (row.get("compatible_tasks") or [])
            ]
        envelope["data"] = {"section": args.section, "count": len(rows), "rows": rows}
        exit_code = EXIT_OK
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_train(args) -> int:
    """Train a DevAD detector on a dataset and persist the model."""
    from trainer import train_devad

    envelope = _envelope("train")
    try:
        spec = {
            "algorithm_id": args.algorithm,
            "dataset_id": args.dataset,
            "model_id": args.model_id,
            "preprocessor_id": args.preprocessor,
            "params": _parse_kv_pairs(args.param, "--param"),
            "preprocessor_params": _parse_kv_pairs(args.pre_param, "--pre-param"),
            "val_fraction": args.val_fraction,
        }
        spec = {k: v for k, v in spec.items() if v not in (None, {})}
        envelope["data"] = train_devad(spec)
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_detect(args) -> int:
    """Run a persisted DevAD model; same result envelope as `run`."""
    from trainer import detect_devad

    envelope = _envelope("result")
    try:
        spec = {
            "model_id": args.model_id,
            "dataset_id": args.dataset,
            "preprocessor_id": args.preprocessor,
            "params": _parse_kv_pairs(args.param, "--param"),
            "preprocessor_params": _parse_kv_pairs(args.pre_param, "--pre-param"),
        }
        spec = {k: v for k, v in spec.items() if v not in (None, {})}
        data = detect_devad(spec)
        if args.out:
            _write_json(args.out, {**envelope, "data": data})
        envelope["data"] = _compact_result(data) if args.compact else data
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_evaluate(args) -> int:
    """Re-score a run with registry metrics — from a saved result or a re-run."""
    from runners import evaluate_saved_run

    envelope = _envelope("evaluate")
    try:
        metric_ids = list(args.metric or [])
        if not metric_ids:
            raise UsageError("`labts evaluate` needs at least one --metric <id>.")
        if args.from_file:
            data = evaluate_saved_run(_load_result_file(args.from_file), metric_ids)
        else:
            from metrics import resolve_metrics

            spec = _load_spec(args.spec)
            spec["metrics"] = metric_ids
            result = run_experiment(spec)
            requested_names = [
                entry["name"] for entry in resolve_metrics(metric_ids, result["task"])
            ]
            data = {
                "status": "ok",
                "task": result["task"],
                "run_id": result["run_id"],
                "dataset_id": result["dataset"]["id"],
                "algorithm_id": result["algorithm"]["id"],
                "metric_ids": metric_ids,
                "metrics": {
                    name: result["metrics"][name]
                    for name in requested_names
                    if name in result["metrics"]
                },
                "source": "fresh run (re-fit)",
            }
        envelope["data"] = data
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_analyze(args) -> int:
    """Fit a param_est analyzer on a catalog series; print its estimates."""
    from analyzer import run_analysis

    envelope = _envelope("analyze")
    try:
        spec = {
            "analyzer_id": args.algorithm,
            "dataset_id": args.dataset,
            "params": _parse_kv_pairs(args.param, "--param"),
        }
        spec = {k: v for k, v in spec.items() if v not in (None, {})}
        envelope["data"] = run_analysis(spec)
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_dist(args) -> int:
    """Pairwise distance matrix over a panel dataset."""
    from domain_runners import compute_distance_matrix

    envelope = _envelope("dist")
    try:
        from catalog import get_dataset as _get_dataset

        dataset = _get_dataset(args.dataset or "unit-test")
        if dataset is None or not dataset.get("enabled"):
            raise PlaygroundError(f"Dataset is not enabled: {args.dataset}")
        params = _parse_kv_pairs(args.param, "--param")
        log = []
        results = []
        for metric_id in args.metric:
            results.append(
                compute_distance_matrix(
                    dataset,
                    metric_id,
                    params=params,
                    max_instances=args.max_instances,
                    log=log,
                )
            )
        envelope["data"] = {
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "split": "train",
            "max_instances": args.max_instances,
            "results": results,
            "log": log,
        }
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_predict(args) -> int:
    """Predict with a persisted model via trainer.predict_estimator (M2 backend)."""
    envelope = _envelope("predict")
    try:
        try:
            from trainer import predict_estimator
        except ImportError:
            raise PlaygroundError(
                "The generic predict backend is not available in this build: "
                "`trainer.predict_estimator` is provided by the persistence "
                "mission (M2, branch feat/labts-sktime-save-load-train-predict). "
                "For DevAD anomaly models use `labts detect --model-id ...` instead."
            ) from None
        spec = {
            "model_id": args.model_id,
            "dataset_id": args.dataset,
            "params": _parse_kv_pairs(args.param, "--param"),
        }
        spec = {k: v for k, v in spec.items() if v not in (None, {})}
        data = predict_estimator(**spec)
        if args.out:
            _write_json(args.out, {**envelope, "data": data})
        envelope["data"] = _compact_result(data) if args.compact else data
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_fork(args) -> int:
    """Materialize a catalog algorithm as an editable experiments/ plugin."""
    from user_algos import PluginError, scaffold_fork, user_algorithm_id

    envelope = _envelope("fork")
    try:
        algorithm = get_enabled_algorithm(args.algorithm_id)
        if algorithm is None:
            raise UsageError(
                f"Algorithm is not enabled or unknown: {args.algorithm_id}. "
                "Fork sources must be enabled catalog entries (see `labts.py ls algorithms`)."
            )
        target = scaffold_fork(algorithm, name=args.name)
        envelope["data"] = {
            "path": str(target),
            "algorithm_id": user_algorithm_id(target.stem),
            "forked_from": algorithm["id"],
            "next_steps": [
                f"edit {target}",
                f"labts.py check {target}",
                f"labts.py run --algorithm {user_algorithm_id(target.stem)} --task {algorithm['task']} ...",
            ],
        }
        exit_code = EXIT_OK
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except PluginError as exc:
        return _fail(envelope, sys.stdout, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_check(args) -> int:
    """Validate a plugin file and run a tiny smoke experiment."""
    from user_algos import check_plugin

    envelope = _envelope("check")
    try:
        path = Path(args.file)
        if not path.is_file():
            raise UsageError(f"Plugin file not found: {args.file}")
        result = check_plugin(path)
        envelope["data"] = result
        if result["ok"]:
            exit_code = EXIT_OK
        else:
            envelope.update(status="blocked", error=result["error"])
            exit_code = EXIT_BLOCKED
    except UsageError as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_USAGE)
    except Exception as exc:
        return _fail(envelope, sys.stdout, exc, "error", EXIT_ERROR)
    _emit_json(sys.stdout, envelope)
    return exit_code


def _main_export(args) -> int:
    """Print raw code/report on stdout; JSON envelope on stderr for errors."""
    key = _EXPORT_KEYS[args.command]
    envelope = _envelope(args.command)
    try:
        if args.from_file:
            data = _load_result_file(args.from_file)
        else:
            data = run_experiment(_load_spec(args.spec))
        text = data.get(key)
        if not text:
            raise UsageError(
                f"Result has no `{key}` payload. If it came from `run --compact` "
                "output, re-run with --out (full result is always saved there) "
                "or pass --spec instead."
            )
    except UsageError as exc:
        return _fail(envelope, sys.stderr, exc, "error", EXIT_USAGE)
    except PlaygroundError as exc:
        return _fail(envelope, sys.stderr, exc, "blocked", EXIT_BLOCKED)
    except Exception as exc:
        return _fail(envelope, sys.stderr, exc, "error", EXIT_ERROR)
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cat = sub.add_parser("catalog", help="Print the unified catalog JSON.")
    p_cat.add_argument(
        "--compact",
        action="store_true",
        help="Enabled entries only; drop the compatibility matrix, "
        "dependency status, and HF metadata.",
    )

    p_ls = sub.add_parser(
        "ls",
        help="List one catalog section (quick discovery without the full catalog).",
    )
    p_ls.add_argument("section", choices=_LS_SECTIONS)
    p_ls.add_argument(
        "--task",
        help="Keep only entries usable for this task "
        "(matches entry.task or entry.compatible_tasks).",
    )
    p_ls.add_argument(
        "--all",
        action="store_true",
        help="Include disabled algorithms/preprocessors (default: enabled only).",
    )

    p_run = sub.add_parser("run", help="Run one experiment spec.")
    p_run.add_argument(
        "--spec",
        help="Spec as a JSON string, @path/to/spec.json, or - for stdin. "
        "Alternative to the --task/--dataset/--algorithm flags.",
    )
    p_run.add_argument("--task", help="Task id, e.g. forecasting. Omit for the default task.")
    p_run.add_argument("--dataset", help="Dataset id; omit for the per-task default.")
    p_run.add_argument("--algorithm", help="Algorithm id; omit for the per-task default.")
    p_run.add_argument(
        "--preprocessor",
        action="append",
        help="Preprocessor id; repeatable — steps run in the given order, "
        "each must preserve length/instance count.",
    )
    p_run.add_argument(
        "--metric",
        action="append",
        metavar="ID",
        help="Extra metric from the registry (see `ls metrics --task X`); "
        "repeatable. Defaults are unchanged when omitted.",
    )
    p_run.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Algorithm/eval parameter; repeatable. Numbers are coerced.",
    )
    p_run.add_argument(
        "--pre-param",
        action="append",
        metavar="[STEP:]KEY=VALUE",
        help="Preprocessor parameter; repeatable. With multiple --preprocessor "
        "steps, prefix the 1-based step index, e.g. `2:degree=2`.",
    )
    p_run.add_argument(
        "--compact",
        action="store_true",
        help="Drop series/tables/code/report on stdout; keep metrics, "
        "summary, and log. The --out file always gets the full result.",
    )
    p_run.add_argument(
        "--out",
        metavar="FILE",
        help="Also save the full result envelope to FILE (input for "
        "`script --from` / `report --from`).",
    )

    for command, what in (("script", "reproduction script"), ("report", "Markdown report")):
        p_exp = sub.add_parser(
            command,
            help=f"Print the run's generated {what} as raw text on stdout.",
        )
        source = p_exp.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--spec",
            help="Run this spec and export from the fresh result "
            "(JSON string, @path/to/spec.json, or - for stdin).",
        )
        source.add_argument(
            "--from",
            dest="from_file",
            metavar="FILE",
            help="Export from a result saved earlier with `run --out` "
            "(no re-run).",
        )

    p_fork = sub.add_parser(
        "fork",
        help="Fork a catalog algorithm into an editable playground/experiments/ plugin.",
    )
    p_fork.add_argument("algorithm_id", help="Enabled catalog algorithm id to fork.")
    p_fork.add_argument(
        "--name",
        help="Plugin file/display name (default: the algorithm's name, slugified).",
    )

    p_train = sub.add_parser(
        "train",
        help="Train a DevAD detector on a dataset and persist the model "
        "under playground/models/<model-id>/.",
    )
    p_train.add_argument(
        "--algorithm",
        required=True,
        help="DevAD adapter id, e.g. registered-anomaly_detection-DevADFITSDetector.",
    )
    p_train.add_argument("--dataset", help="Anomaly dataset id (default: yahoo).")
    p_train.add_argument(
        "--model-id",
        help="Persisted model directory name (default: <family>-<dataset>). "
        "An existing model with the same id is overwritten.",
    )
    p_train.add_argument("--preprocessor", help="Preprocessor id (default: none).")
    p_train.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Adapter param (win_len/epochs/batch_size/seed/device/...) or "
        "DevAD hyperparameter (h_dim/lr/...); repeatable.",
    )
    p_train.add_argument(
        "--pre-param",
        action="append",
        metavar="KEY=VALUE",
        help="Preprocessor parameter; repeatable.",
    )
    p_train.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction of the series tail for validation "
        "(enables early stopping for torch families). Default: 0.",
    )

    p_detect = sub.add_parser(
        "detect",
        help="Run a persisted model (labts train) on a dataset; "
        "same result envelope as `run`.",
    )
    p_detect.add_argument(
        "--model-id",
        required=True,
        help="Trained model id under playground/models/ (see `ls models`).",
    )
    p_detect.add_argument("--dataset", help="Anomaly dataset id (default: yahoo).")
    p_detect.add_argument("--preprocessor", help="Preprocessor id (default: none).")
    p_detect.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Detect param: threshold_quantile (default 0.99), device; repeatable.",
    )
    p_detect.add_argument(
        "--pre-param",
        action="append",
        metavar="KEY=VALUE",
        help="Preprocessor parameter; repeatable.",
    )
    p_detect.add_argument("--compact", action="store_true", help="Like `run --compact`.")
    p_detect.add_argument("--out", metavar="FILE", help="Like `run --out`.")

    p_check = sub.add_parser(
        "check",
        help="Validate a plugin file and run a tiny smoke experiment.",
    )
    p_check.add_argument("file", help="Path to the plugin .py file.")

    p_eval = sub.add_parser(
        "evaluate",
        help="Re-score a run with registry metrics: from a saved result "
        "(`--from run.json`, no re-fit — anomaly runs use the saved continuous "
        "scores) or from a spec (`--spec ...`, re-runs the experiment).",
    )
    eval_source = p_eval.add_mutually_exclusive_group(required=True)
    eval_source.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="Result saved earlier with `run --out` (or `detect --out`).",
    )
    eval_source.add_argument(
        "--spec",
        help="Re-run this spec, then score (JSON string, @path, or - for stdin).",
    )
    p_eval.add_argument(
        "--metric",
        action="append",
        required=True,
        metavar="ID",
        help="Metric id from the registry (see `ls metrics --task X`); repeatable.",
    )

    p_analyze = sub.add_parser(
        "analyze",
        help="Fit a param_est analyzer on a catalog series and print the "
        "estimated parameters (no evaluation stage).",
    )
    p_analyze.add_argument(
        "--algorithm",
        help="Analyzer id (see `ls analyzers`; default: seasonality-acf).",
    )
    p_analyze.add_argument("--dataset", help="Series dataset id (default: airline).")
    p_analyze.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Analyzer constructor parameter; repeatable.",
    )

    p_dist = sub.add_parser(
        "dist",
        help="Pairwise distance matrix over a panel dataset (train split).",
    )
    p_dist.add_argument("--dataset", help="Panel dataset id (default: unit-test).")
    p_dist.add_argument(
        "--metric",
        action="append",
        required=True,
        metavar="ID",
        help="Distance id: sktime name (dtw, euclidean, ...) or scipy:<name> "
        "(see `ls distances`); repeatable.",
    )
    p_dist.add_argument(
        "--max-instances",
        type=int,
        default=50,
        help="Cap the number of train instances (default: 50).",
    )
    p_dist.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Distance parameter (e.g. window=0.1 for dtw, p=3 for "
        "scipy:minkowski); repeatable, applies to every requested metric.",
    )

    p_predict = sub.add_parser(
        "predict",
        help="Predict with a persisted model via the generic persistence "
        "backend (trainer.predict_estimator, provided by mission M2). "
        "Returns blocked with a hint while the backend is unavailable.",
    )
    p_predict.add_argument(
        "--model-id",
        required=True,
        help="Persisted model id (see `ls models`).",
    )
    p_predict.add_argument("--dataset", help="Dataset id to predict on.")
    p_predict.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Predict parameter; repeatable.",
    )
    p_predict.add_argument("--compact", action="store_true", help="Like `run --compact`.")
    p_predict.add_argument("--out", metavar="FILE", help="Like `run --out`.")

    args = parser.parse_args(argv)
    if args.command == "catalog":
        return _main_catalog(args)
    if args.command == "ls":
        return _main_ls(args)
    if args.command == "run":
        return _main_run(args)
    if args.command == "train":
        return _main_train(args)
    if args.command == "detect":
        return _main_detect(args)
    if args.command == "evaluate":
        return _main_evaluate(args)
    if args.command == "analyze":
        return _main_analyze(args)
    if args.command == "dist":
        return _main_dist(args)
    if args.command == "predict":
        return _main_predict(args)
    if args.command == "fork":
        return _main_fork(args)
    if args.command == "check":
        return _main_check(args)
    return _main_export(args)


if __name__ == "__main__":
    sys.exit(main())
