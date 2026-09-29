#!/usr/bin/env python3
"""Regenerate the groundtruth train/run command pairs with explicit params.

Every groundtruth entry records the exact commands that reproduce its claim.
The pair is the labts train-once/run-many lifecycle:

    train_command = labts train --algorithm A --dataset D --model-id bench-<id>
                    + EVERY effective parameter as an explicit --param flag
                    (the entry's own protocol params first, then the catalog
                    default constructor/eval params, so nothing is implicit)
    run_command   = labts run --model-id bench-<id> --compact
                    (evaluate the persisted model, no refit; the judged
                    metrics are all in the default result payload, so no
                    --metric flags are needed)

The one-shot `command` field is replaced by `train_command`/`run_command`.
Idempotent: entries that already carry a pair are regenerated from the
recorded `spec_params` (the entry's original explicit params), so catalog
default changes propagate on re-run.

Usage:
    python playground/benchmark/expand_commands.py            # rewrite all
    python playground/benchmark/expand_commands.py --check    # print only
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAYGROUND_DIR = HERE.parent
REPO_ROOT = PLAYGROUND_DIR.parent
for path in (str(PLAYGROUND_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

GROUNDTRUTH_DIR = HERE / "groundtruth"

# Protocol eval defaults that live in the runners (not in catalog params);
# injected only when relevant and absent, so the pair is fully explicit.
TASK_EVAL_DEFAULTS = {
    "causal": {"max_samples": 2000, "seed": 7},
}


def parse_one_shot(command: str) -> dict:
    """Extract task/algorithm/dataset/params from a recorded one-shot command."""
    tokens = shlex.split(command)
    out = {"params": {}}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--task", "--algorithm", "--dataset") and i + 1 < len(tokens):
            out[tok[2:]] = tokens[i + 1]
            i += 2
        elif tok == "--param" and i + 1 < len(tokens):
            key, _, value = tokens[i + 1].partition("=")
            out["params"][key] = value
            i += 2
        else:
            i += 1
    return out


def coerce(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def fmt_value(value) -> str:
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def expand_entry(entry: dict, task: str, catalog_defaults: dict) -> dict:
    """Attach train_command/run_command (and spec_params) to one entry."""
    if entry.get("spec_params") is not None:
        explicit = dict(entry["spec_params"])
    else:
        parsed = parse_one_shot(entry["command"])
        explicit = {k: coerce(v) for k, v in parsed["params"].items()}

    merged = dict(explicit)
    for key, value in (catalog_defaults or {}).items():
        # None/bool defaults are skipped: omitting a flag keeps the default
        # anyway, and bools do not survive CLI string coercion faithfully.
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (list, tuple, dict)):
            continue
        merged.setdefault(key, value)
    for key, value in TASK_EVAL_DEFAULTS.get(task, {}).items():
        merged.setdefault(key, value)
    if task == "forecasting":
        merged.setdefault("horizon", 12)
        if str(merged.get("eval_mode") or "single") == "rolling":
            if "test_start_fraction" not in merged:
                merged.setdefault("test_fraction", 0.2)

    ordered = dict(explicit)
    for key in sorted(merged):
        if key not in ordered:
            ordered[key] = merged[key]

    model_id = f"bench-{entry['id']}"
    params = " ".join(f"--param {k}={fmt_value(v)}" for k, v in ordered.items())
    entry["spec_params"] = explicit
    entry["train_command"] = (
        f"python playground/labts.py train --algorithm {entry['algorithm']} "
        f"--dataset {entry['dataset']} --model-id {model_id} {params}"
    )
    entry["run_command"] = (
        f"python playground/labts.py run --model-id {model_id} --compact"
    )
    entry.pop("command", None)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print, do not write.")
    args = parser.parse_args()

    from catalog import get_enabled_algorithm

    for path in sorted(GROUNDTRUTH_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = payload["task"]
        changed = False
        for entry in payload["entries"]:
            algorithm = get_enabled_algorithm(entry["algorithm"])
            if algorithm is None:
                raise SystemExit(f"Unknown algorithm in {entry['id']}: {entry['algorithm']}")
            expand_entry(entry, task, algorithm.get("params") or {})
            changed = True
            if args.check:
                print(f"# {entry['id']}\n{entry['train_command']}\n{entry['run_command']}\n")
        if changed and not args.check:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {path} ({len(payload['entries'])} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
