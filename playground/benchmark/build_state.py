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
    if result is None:
        out["status"] = "pending"
        out["values"] = None
        out["per_metric"] = None
        out["ran_at"] = None
        out["duration_s"] = None
        return out
    out["status"] = result.get("status", "error")
    out["values"] = result.get("values")
    out["per_metric"] = result.get("per_metric")
    out["ran_at"] = result.get("ran_at")
    out["duration_s"] = result.get("duration_s")
    out["error"] = result.get("error")
    # raw envelope metrics; "metrics" stays the groundtruth dict from the entry
    out["raw_metrics"] = result.get("metrics")
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


def _badge(status: str, text: str) -> str:
    return f'<span class="status {status}"><span class="dot"></span>{esc(text)}</span>'


def _metrics_line(entry: dict) -> str:
    """Per-metric inline results: `MSE 0.375→0.319 ✓  MAE 0.399→0.411 ✓`."""
    parts = []
    for name, gt in entry["metrics"].items():
        pm = (entry.get("per_metric") or {}).get(name)
        if pm is None:
            parts.append(
                f'<span class="m-pen"><span class="m-name">{esc(name)}</span> '
                f"{fmt_num(gt)}→—</span>"
            )
            continue
        cls = {"reproduced": "m-ok", "deviation": "m-bad"}.get(pm["status"], "m-pen")
        mark = "✓" if pm["status"] == "reproduced" else "✗"
        parts.append(
            f'<span class="{cls}"><span class="m-name">{esc(name)}</span> '
            f'{fmt_num(gt)}→{fmt_num(pm["value"])} {mark}</span>'
        )
    return f'<div class="mline">{"".join(parts)}</div>'


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
    error_html = ""
    if entry.get("error"):
        error_html = (
            '<div class="full"><div class="kv-label">Error</div>'
            f'<div class="kv" style="color:var(--red)">{esc(entry["error"])}</div></div>'
        )
    metrics_html = ""
    if entry.get("raw_metrics"):
        metrics_html = (
            '<div class="full"><div class="kv-label">All metrics returned</div>'
            f'<div class="raw-metrics">{esc(json.dumps(entry["raw_metrics"], ensure_ascii=False))}</div></div>'
        )
    window = f'<span class="win">{esc(entry["window"])}</span>' if entry.get("window") else ""
    return f"""
<details class="claim">
  <summary>
    <div class="ds">{esc(entry['dataset_name'])}{window}</div>
    {_metrics_line(entry)}
    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
      {_status_html(entry['status'])}
      <span class="chev">›</span>
    </div>
  </summary>
  <div class="claim-detail">
    <div class="detail-grid">
      <div class="full">
        <div class="kv-label">1 · Train — fit &amp; persist the model (every parameter explicit)</div>
        <div class="cmd">{esc(entry['train_command'])}<button class="copy" data-cmd="{esc(entry['train_command'])}">Copy</button></div>
        <div class="kv-label" style="margin-top:10px">2 · Run — evaluate the persisted model, no refit</div>
        <div class="cmd">{esc(entry['run_command'])}<button class="copy" data-cmd="{esc(entry['run_command'])}">Copy</button></div>
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
        <div class="kv">{esc(entry.get('settings', ''))}</div>
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
        <div class="kv">{esc(entry.get('notes', ''))}</div>
      </div>
      {error_html}
      {metrics_html}
    </div>
  </div>
</details>"""


def render_group(algorithm_name: str, entries: list[dict]) -> str:
    """Parent row for one algorithm: its dataset/config claims as sub-rows."""
    n = len(entries)
    k = sum(1 for e in entries if e["status"] == "reproduced")
    if all(e["status"] == "pending" for e in entries):
        agg_status, agg_text = "pending", f"{n} configs · not run"
    elif k == n:
        agg_status, agg_text = "reproduced", f"{k}/{n} configs reproduced"
    else:
        agg_status, agg_text = "deviation", f"{k}/{n} configs reproduced"
    venue = entries[0]["source"]["venue"].split(",")[0]
    return f"""
<details class="claim-group">
  <summary>
    <div class="alg">{esc(algorithm_name)}<span class="lib">{esc(venue)}</span></div>
    <div class="grp-note">{n} dataset/config claim{"s" if n > 1 else ""}</div>
    <div style="display:flex;align-items:center;justify-content:flex-end;gap:10px">
      {_badge(agg_status, agg_text)}
      <span class="chev">›</span>
    </div>
  </summary>
  <div class="group-body">{"".join(render_claim(e) for e in entries)}</div>
</details>"""


def render_section(task: dict, entries: list[dict], algs: list[dict]) -> str:
    if entries:
        groups: dict[str, list[dict]] = {}
        for e in entries:
            groups.setdefault(e["algorithm_name"], []).append(e)
        claims = (
            '<div class="claim-groups">'
            + "".join(render_group(name, es) for name, es in groups.items())
            + "</div>"
        )
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
