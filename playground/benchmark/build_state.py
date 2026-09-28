#!/usr/bin/env python3
"""Build the benchmark web state (``state.json``).

Merges three sources into one JSON document served at ``/api/benchmark``:

1. the live labts catalog (every algorithm per task, with availability),
2. the curated paper-reported groundtruth (``groundtruth/*.json``),
3. the reproduction records (``results/*.json``, written by ``reproduce.py``).

Usage:
    python playground/benchmark/build_state.py              # live catalog build
    python playground/benchmark/build_state.py --snapshot   # reuse catalog snapshot
"""

from __future__ import annotations

import argparse
import datetime
import html as html_lib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAYGROUND_DIR = HERE.parent
REPO_ROOT = PLAYGROUND_DIR.parent
for path in (str(PLAYGROUND_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

STATE_PATH = HERE / "state.json"

TASKS = [
    ("forecasting", "Forecasting", "Predict future values of a series."),
    ("classification", "Classification", "Assign series to discrete classes."),
    ("regression", "Regression", "Predict a continuous value from a series."),
    ("clustering", "Clustering", "Group series without labels."),
    ("anomaly_detection", "Anomaly Detection", "Flag anomalous points or segments."),
    ("causal", "Causal Discovery", "Recover the causal graph behind the data."),
]


def load_catalog(live: bool) -> dict:
    if live:
        from catalog import build_catalog

        return build_catalog(include_registered=True)
    snapshot = (
        REPO_ROOT / ".agent" / "skills" / "time-series-sandbox" / "catalog_snapshot.json"
    )
    return json.loads(snapshot.read_text(encoding="utf-8"))


def load_entries() -> list[dict]:
    entries = []
    for path in sorted((HERE / "groundtruth").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload["entries"]:
            entry = dict(entry)
            entry["task"] = payload["task"]
            entries.append(entry)
    return entries


def load_result(entry_id: str) -> dict | None:
    path = HERE / "results" / f"{entry_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def attach_status(entry: dict, result: dict | None) -> dict:
    """Recompute the display status from the stored value (source of truth)."""
    out = dict(entry)
    out.pop("command", None)  # keep state lean; command comes from result/entry
    out["command"] = entry["command"]
    if result is None:
        out["status"] = "pending"
        out["value_reproduced"] = None
        out["delta"] = None
        out["delta_rel"] = None
        out["ran_at"] = None
        out["duration_s"] = None
        return out
    out["status"] = result.get("status", "error")
    out["value_reproduced"] = result.get("value")
    out["delta"] = result.get("delta")
    out["delta_rel"] = result.get("delta_rel")
    out["ran_at"] = result.get("ran_at")
    out["duration_s"] = result.get("duration_s")
    out["error"] = result.get("error")
    out["metrics"] = result.get("metrics")
    return out


# ---------------------------------------------------------------------------
# static HTML rendering (the page must be fully readable with JS disabled,
# e.g. in IDE preview panes — everything below is baked into leaderboard.html)
# ---------------------------------------------------------------------------

STATUS_LABEL = {
    "reproduced": "Reproduced",
    "deviation": "Deviation",
    "error": "Error",
    "pending": "Pending",
}


def esc(value) -> str:
    return html_lib.escape(str(value if value is not None else ""), quote=True)


def fmt_num(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    if a < 0.001 or a >= 1e6:
        return f"{v:.2e}"
    return f"{v:.{digits}g}"


def _status_html(status: str) -> str:
    return (
        f'<span class="status {status}"><span class="dot"></span>'
        f"{STATUS_LABEL.get(status, status)}</span>"
    )


def _delta_html(entry: dict) -> str:
    delta = entry.get("delta")
    if delta is None:
        return '<div class="delta">—</div>'
    tol = entry["tolerance"]
    if tol["type"] == "absolute":
        tol_txt = f"±{fmt_num(tol['value'], 2)}"
    else:
        tol_txt = f"±{round(float(tol['value']) * 100)}%"
    cls = {"reproduced": "ok", "deviation": "bad"}.get(entry["status"], "")
    sign = "+" if delta > 0 else ""
    return (
        f'<div class="delta {cls}">{sign}{fmt_num(delta)} '
        f'<span class="tol">{tol_txt}</span></div>'
    )


def render_claim(entry: dict) -> str:
    source = entry["source"]
    ran = "not run yet"
    if entry.get("ran_at"):
        ran = (
            esc(entry["ran_at"].replace("T", " ")[:19])
            + " UTC · "
            + fmt_num(entry.get("duration_s"), 3)
            + "s"
        )
    tol = entry["tolerance"]
    tol_txt = (
        f"±{fmt_num(tol['value'], 3)} absolute"
        if tol["type"] == "absolute"
        else f"±{round(float(tol['value']) * 100)}% relative"
    )
    reproduced = "—" if entry["status"] == "pending" else fmt_num(entry.get("value_reproduced"))
    error_html = ""
    if entry.get("error"):
        error_html = (
            '<div class="full"><div class="kv-label">Error</div>'
            f'<div class="kv" style="color:var(--red)">{esc(entry["error"])}</div></div>'
        )
    metrics_html = ""
    if entry.get("metrics"):
        metrics_html = (
            '<div class="full"><div class="kv-label">All metrics returned</div>'
            f'<div class="raw-metrics">{esc(json.dumps(entry["metrics"], ensure_ascii=False))}</div></div>'
        )
    return f"""
<details class="claim">
  <summary>
    <div class="alg">{esc(entry['algorithm_name'])}<span class="lib">{esc(source['venue'].split(',')[0])}</span></div>
    <div class="ds">{esc(entry['dataset_name'])}</div>
    <div class="met">{esc(entry['metric'])}</div>
    <div class="num gt">{fmt_num(entry['value'])}</div>
    <div class="num rep">{reproduced}</div>
    {_delta_html(entry)}
    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
      {_status_html(entry['status'])}
      <span class="chev">›</span>
    </div>
  </summary>
  <div class="claim-detail">
    <div class="detail-grid">
      <div class="full">
        <div class="kv-label">Reproduce — exact labts CLI command</div>
        <div class="cmd">{esc(entry['command'])}<button class="copy" data-cmd="{esc(entry['command'])}">Copy</button></div>
      </div>
      <div>
        <div class="kv-label">Source (groundtruth)</div>
        <div class="kv">
          <a href="{esc(source['url'])}" target="_blank" rel="noopener">{esc(source['title'])}</a><br/>
          {esc(source['venue'])} · {esc(source['locator'])}
        </div>
      </div>
      <div>
        <div class="kv-label">Protocol (as reported)</div>
        <div class="kv">{esc(entry['settings'])}</div>
      </div>
      <div>
        <div class="kv-label">Tolerance</div>
        <div class="kv">{tol_txt}</div>
      </div>
      <div>
        <div class="kv-label">Last run</div>
        <div class="kv">{ran}</div>
      </div>
      <div class="full">
        <div class="kv-label">Notes</div>
        <div class="kv">{esc(entry['notes'])}</div>
      </div>
      {error_html}
      {metrics_html}
    </div>
  </div>
</details>"""


def render_section(task: dict, entries: list[dict], algs: list[dict]) -> str:
    if entries:
        claims = f'<div class="claims">{"".join(render_claim(e) for e in entries)}</div>'
    else:
        claims = (
            '<div class="empty">No groundtruth claims curated yet for this task — '
            "add one under <code>playground/benchmark/groundtruth/</code>.</div>"
        )
    chips = "".join(
        f'<span class="chip {"enabled" if a["enabled"] else "disabled"}" '
        f'title="{esc(a["id"])}" data-name="{esc((a["name"] + " " + a["id"]).lower())}">'
        f'<span class="dot"></span>{esc(a["name"])}<span class="tag">{esc(a["source"])}</span></span>'
        for a in algs
    )
    return f"""
<section class="task-section panel" id="task-{esc(task['id'])}">
  <div class="panel-head">
    <h2>{esc(task['name'])}</h2>
    <span class="desc">{esc(task['description'])}</span>
  </div>
  <div class="panel-sub">{task['enabled_count']} of {task['algorithm_count']} algorithms runnable in this environment</div>

  <div class="section-label">Tracked claims ({len(entries)})</div>
  {claims}

  <div class="section-label">Catalog — all {len(algs)} {esc(task['name'].lower())} algorithms</div>
  <input class="filter" type="search" placeholder="Filter algorithms…" />
  <div class="chips">{chips}</div>
</section>"""


def render_stats(summary: dict, total_algorithms: int) -> str:
    pct = "—" if summary["reproduced_pct"] is None else f"{summary['reproduced_pct']}%"
    cells = [
        (total_algorithms, "algorithms tracked"),
        (summary["entries"], "groundtruth claims"),
        (f"{summary['reproduced']}/{summary['run_total']}", "claims reproduced"),
        (pct, "reproduction rate"),
    ]
    return "".join(
        f'<div class="stat"><div class="num">{esc(n)}</div>'
        f'<div class="lbl">{esc(l)}</div></div>'
        for n, l in cells
    )


def render_tabs(tasks: list[dict]) -> str:
    return "".join(
        f'<a class="tab" href="#{esc(t["id"])}" data-task="{esc(t["id"])}">'
        f'{esc(t["name"])}<span class="cnt">{t["algorithm_count"]}</span></a>'
        for t in tasks
    )


def render_page(state: dict) -> str:
    template = (HERE / "template.html").read_text(encoding="utf-8")
    total_algorithms = sum(t["algorithm_count"] for t in state["tasks"])
    sections = "".join(
        render_section(
            task,
            [e for e in state["entries"] if e["task"] == task["id"]],
            state["algorithms"].get(task["id"], []),
        )
        for task in state["tasks"]
    )
    generated = esc(state["generated_at"].replace("T", " ")[:19] + " UTC")
    return (
        template.replace("<!--BENCH:STATS-->", render_stats(state["summary"], total_algorithms))
        .replace("<!--BENCH:TABS-->", render_tabs(state["tasks"]))
        .replace("<!--BENCH:SECTIONS-->", sections)
        .replace("<!--BENCH:GENERATED-->", generated)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Read the catalog snapshot instead of a live (slow) build.",
    )
    args = parser.parse_args()

    catalog = load_catalog(live=not args.snapshot)
    algorithms: dict[str, list[dict]] = {}
    for row in catalog.get("algorithms", []):
        algorithms.setdefault(row["task"], []).append(
            {
                "id": row["id"],
                "name": row.get("name") or row["id"],
                "enabled": bool(row.get("enabled")),
                "source": "curated" if row.get("curated") else "registry",
            }
        )
    for rows in algorithms.values():
        rows.sort(key=lambda r: (not r["enabled"], r["name"].lower()))

    entries = [attach_status(e, load_result(e["id"])) for e in load_entries()]

    def count(status: str) -> int:
        return sum(1 for e in entries if e["status"] == status)

    reproduced = count("reproduced")
    run_total = reproduced + count("deviation") + count("error")
    state = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "catalog_source": "live" if not args.snapshot else "snapshot",
        "tasks": [
            {
                "id": tid,
                "name": name,
                "description": desc,
                "algorithm_count": len(algorithms.get(tid, [])),
                "enabled_count": sum(1 for a in algorithms.get(tid, []) if a["enabled"]),
            }
            for tid, name, desc in TASKS
        ],
        "algorithms": {tid: algorithms.get(tid, []) for tid, _, _ in TASKS},
        "entries": entries,
        "summary": {
            "entries": len(entries),
            "reproduced": reproduced,
            "deviation": count("deviation"),
            "error": count("error"),
            "pending": count("pending"),
            "run_total": run_total,
            "reproduced_pct": round(100 * reproduced / run_total, 1) if run_total else None,
        },
    }
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {STATE_PATH}")

    # Self-contained static page: all content pre-rendered into the HTML, so
    # the leaderboard opens straight from disk (file://) and stays fully
    # readable even with JavaScript disabled (IDE preview panes).
    leaderboard_path = HERE / "leaderboard.html"
    leaderboard_path.write_text(render_page(state), encoding="utf-8")
    print(f"Wrote {leaderboard_path} (self-contained, open directly in a browser)")

    print(json.dumps(state["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
