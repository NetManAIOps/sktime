#!/usr/bin/env python3
"""Re-run the benchmark's groundtruth claims via the labts CLI.

Reads ``groundtruth/*.json``, executes each entry's recorded command pair
(``train_command`` then ``run_command`` — the train-once/run-many lifecycle
with every parameter explicit), extracts the claimed metric from the labts
result envelope, and writes ``results/<entry-id>.json``:

    {
      "id": ..., "ran_at": ..., "train_command": ..., "run_command": ...,
      "python": ..., "returncode": ..., "duration_s": ..., "value": ...,
      "delta": ..., "delta_rel": ..., "status": "reproduced|deviation|error",
      "metrics": {...}, "error": ..., "stdout_tail": ...
    }

``build_state.py`` merges these back with the groundtruth for the web page.

Usage:
    python playground/benchmark/reproduce.py                # run everything pending
    python playground/benchmark/reproduce.py --all          # re-run everything
    python playground/benchmark/reproduce.py --entry ID     # one entry
    python playground/benchmark/reproduce.py --task forecasting
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
GROUNDTRUTH_DIR = HERE / "groundtruth"
RESULTS_DIR = HERE / "results"

DEFAULT_TIMEOUT_S = 1800


def load_entries() -> list[dict]:
    entries = []
    for path in sorted(GROUNDTRUTH_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload["entries"]:
            entry = dict(entry)
            entry["task"] = payload["task"]
            entries.append(entry)
    return entries


def _parse_envelope(stdout: str) -> dict | None:
    """The labts envelope is printed as one JSON line (warnings may precede)."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _extract_metric(metrics: dict, metric_key: str) -> float | None:
    for key, value in metrics.items():
        if key.lower().replace("_", "-") == metric_key.lower().replace("_", "-"):
            return float(value)
    return None


def compare(value: float, gt_value: float, tolerance: dict) -> tuple[float, float, str]:
    """Return (delta, relative delta, status) for one metric."""
    gt = float(gt_value)
    delta = value - gt
    delta_rel = abs(delta) / abs(gt) if gt else abs(delta)
    if tolerance["type"] == "absolute":
        ok = abs(delta) <= float(tolerance["value"])
    else:
        ok = delta_rel <= float(tolerance["value"])
    return delta, delta_rel, ("reproduced" if ok else "deviation")


def _execute(command: str, timeout_s: int) -> subprocess.CompletedProcess:
    argv = [sys.executable] + shlex.split(command)[1:]  # replace leading "python"
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def run_entry(entry: dict, timeout_s: int) -> dict:
    train_command = entry["train_command"]
    run_command = entry["run_command"]
    started = time.time()
    proc = _execute(train_command, timeout_s)
    if proc.returncode == 0:
        train_envelope = _parse_envelope(proc.stdout or "")
        if train_envelope is None or train_envelope.get("status") != "ok":
            proc = subprocess.CompletedProcess(
                proc.args,
                1,
                proc.stdout,
                f"train status="
                f"{(train_envelope or {}).get('status')}: "
                f"{(train_envelope or {}).get('error')}",
            )
        else:
            proc = _execute(run_command, timeout_s)
    duration = time.time() - started
    record = {
        "id": entry["id"],
        "task": entry["task"],
        "ran_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "train_command": train_command,
        "run_command": run_command,
        "python": sys.executable,
        "returncode": proc.returncode,
        "duration_s": round(duration, 2),
        "values": None,
        "per_metric": None,
        "status": "error",
        "metrics": None,
        "error": None,
        "stdout_tail": (proc.stdout or "")[-2000:],
    }
    if proc.returncode != 0:
        record["error"] = f"exit code {proc.returncode}: {(proc.stderr or '')[-500:]}"
        return record
    envelope = _parse_envelope(proc.stdout or "")
    if envelope is None:
        record["error"] = "no JSON envelope found on stdout"
        return record
    if envelope.get("status") != "ok":
        record["error"] = f"labts status={envelope.get('status')}: {envelope.get('error')}"
        return record
    metrics = (envelope.get("data") or {}).get("metrics") or {}
    record["metrics"] = metrics

    per_metric = {}
    missing = []
    for name, gt_value in entry["metrics"].items():
        value = _extract_metric(metrics, name)
        if value is None:
            missing.append(name)
            continue
        delta, delta_rel, status = compare(value, gt_value, entry["tolerance"])
        per_metric[name] = {
            "gt": gt_value,
            "value": value,
            "delta": round(delta, 6),
            "delta_rel": round(delta_rel, 6),
            "status": status,
        }
    record["values"] = {name: m["value"] for name, m in per_metric.items()}
    record["per_metric"] = per_metric
    if missing:
        record["error"] = f"metrics {missing} not in result keys {list(metrics)}"
        return record
    statuses = {m["status"] for m in per_metric.values()}
    record["status"] = "reproduced" if statuses == {"reproduced"} else "deviation"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", help="Run a single groundtruth entry id.")
    parser.add_argument("--task", help="Run entries of one task only.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-run entries that already have results (default: only pending).",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    entries = load_entries()
    if args.entry:
        entries = [e for e in entries if e["id"] == args.entry]
    if args.task:
        entries = [e for e in entries if e["task"] == args.task]
    if not args.all:
        entries = [e for e in entries if not (RESULTS_DIR / f"{e['id']}.json").exists()]
    if not entries:
        print("Nothing to run.")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for entry in entries:
        print(f"▶ {entry['id']}  ($ {entry['run_command']})", flush=True)
        try:
            record = run_entry(entry, args.timeout)
        except subprocess.TimeoutExpired:
            record = {
                "id": entry["id"],
                "task": entry["task"],
                "status": "error",
                "error": f"timeout after {args.timeout}s",
                "ran_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "train_command": entry["train_command"],
                "run_command": entry["run_command"],
                "python": sys.executable,
            }
        (RESULTS_DIR / f"{entry['id']}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        mark = {"reproduced": "✓", "deviation": "✗", "error": "!"}.get(record["status"], "?")
        shown = ""
        if record.get("per_metric"):
            shown = "  ".join(
                f"{name} {m['value']:.6g}({'+' if m['delta'] >= 0 else ''}{m['delta']:.3g})"
                for name, m in record["per_metric"].items()
            )
        print(f"  {mark} {record['status']}: {shown} ({record.get('duration_s', '?')}s)\n", flush=True)
        if record["status"] == "error":
            failed += 1
    print(f"Done. {len(entries)} run, {failed} error(s). Results in {RESULTS_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
