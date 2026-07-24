#!/usr/bin/env python3
"""LabTS API — CLI version of the TSBox Sandbox Playground, for automated harnesses.

One process per call, no HTTP server required. Every Playground web capability
has a CLI counterpart (same catalog/runner code, identical results):

    Web endpoint                  CLI command
    ----------------------------  ------------------------------------------
    GET  /api/catalog             python playground/labts.py catalog [--compact]
    POST /api/run                 python playground/labts.py run --spec '<json>' [--compact] [--out run.json]
    GET  /api/export/script       python playground/labts.py script (--spec '<json>' | --from run.json)
    GET  /api/export/report       python playground/labts.py report (--spec '<json>' | --from run.json)

`--spec` accepts a JSON string, `@path/to/spec.json`, or `-` for stdin.

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

from catalog import EVAL_PARAMS, REPO_ROOT, _sanitize_for_json, build_catalog  # noqa: E402
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
_RUN_HEAVY_KEYS = ("series", "tables", "code", "report")
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
    keep_algo = ("id", "name", "task", "subtype", "params")
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
        spec = _load_spec(args.spec)
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

    p_run = sub.add_parser("run", help="Run one experiment spec.")
    p_run.add_argument(
        "--spec",
        required=True,
        help="Spec as a JSON string, @path/to/spec.json, or - for stdin.",
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

    args = parser.parse_args(argv)
    if args.command == "catalog":
        return _main_catalog(args)
    if args.command == "run":
        return _main_run(args)
    return _main_export(args)


if __name__ == "__main__":
    sys.exit(main())
