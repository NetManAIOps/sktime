#!/usr/bin/env python3
"""Local HTTP server for the LibraryTS Playground."""

from __future__ import annotations

import json
import math
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
STATIC_ROOT = HERE / "static"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from catalog import REPO_ROOT, build_catalog, write_snapshot
from runners import RUNS, PlaygroundError, get_run, run_experiment

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PlaygroundHandler(BaseHTTPRequestHandler):
    server_version = "LibraryTSPlayground/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/catalog":
            self._json(build_catalog(include_registered=True))
            return
        if parsed.path == "/api/export/script":
            self._export(parsed.query, "code", "text/x-python", "experiment.py")
            return
        if parsed.path == "/api/export/report":
            self._export(parsed.query, "report", "text/markdown", "experiment.md")
            return
        if parsed.path == "/api/models":
            from trainer import list_trained_models

            self._json({"status": "ok", "models": list_trained_models()})
            return
        if parsed.path in {"/", "/index.html"}:
            self._file(STATIC_ROOT / "index.html")
            return
        if parsed.path in {"/benchmark", "/benchmark.html"}:
            leaderboard = (HERE / "benchmark" / "leaderboard.html").resolve()
            if leaderboard.is_file():
                self._raw_file(leaderboard)
            else:
                self._json(
                    {
                        "status": "error",
                        "error": "leaderboard.html not built; run "
                        "`python playground/benchmark/build_state.py` first",
                    },
                    status=404,
                )
            return
        if parsed.path == "/api/benchmark":
            state = HERE / "benchmark" / "state.json"
            if not state.is_file():
                self._json(
                    {
                        "status": "error",
                        "error": "state.json not built; run "
                        "`python playground/benchmark/build_state.py` first",
                    },
                    status=404,
                )
                return
            self._json(json.loads(state.read_text(encoding="utf-8")))
            return
        self._file(STATIC_ROOT / parsed.path.lstrip("/"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in {
            "/api/run", "/api/train", "/api/analyze", "/api/dist", "/api/evaluate"
        }:
            self._json({"status": "error", "error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            spec = json.loads(body) if body else {}
            if parsed.path == "/api/analyze":
                from analyzer import run_analysis

                result = run_analysis(spec)
            elif parsed.path == "/api/dist":
                from domain_runners import distance_report

                max_instances = int(spec.get("max_instances") or 20)
                if max_instances < 2:
                    raise PlaygroundError("max_instances must be >= 2.")
                result = {
                    "status": "ok",
                    **distance_report(
                        spec.get("dataset_id"),
                        list(spec.get("metrics") or []),
                        spec.get("params") or {},
                        max_instances,
                    ),
                }
            elif parsed.path == "/api/evaluate":
                from runners import evaluate_saved_run

                saved = get_run(spec.get("run_id") or "")
                if saved is None:
                    raise PlaygroundError(
                        f"Unknown run_id: {spec.get('run_id')!r}; run an experiment first."
                    )
                metric_ids = list(spec.get("metrics") or [])
                if not metric_ids:
                    raise PlaygroundError("Pick at least one metric to evaluate.")
                result = evaluate_saved_run(saved, metric_ids)
            elif parsed.path == "/api/train":
                from trainer import train as train_model

                result = train_model(spec)
            elif spec.get("model_id"):
                # same code path as `libraryts run --model-id`
                from trainer import run_trained_model

                result = run_trained_model(
                    spec["model_id"],
                    spec.get("dataset_id"),
                    spec.get("params") or {},
                    spec.get("metrics") or [],
                )
                if result.get("run_id"):
                    RUNS[result["run_id"]] = result
            else:
                result = run_experiment(spec)
            self._json(result)
        except PlaygroundError as exc:
            self._json({"status": "blocked", "error": str(exc)}, status=200)
        except Exception as exc:
            self._json(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                status=500,
            )

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _export(self, query: str, key: str, content_type: str, filename: str):
        run_id = parse_qs(query).get("run_id", [""])[0]
        result = get_run(run_id)
        if result is None:
            self._json({"status": "error", "error": f"Unknown run_id: {run_id}"}, status=404)
            return
        payload = result[key].encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, payload, status: int = 200):
        def sanitize(obj):
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [sanitize(v) for v in obj]
            return obj

        body = json.dumps(sanitize(payload), indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(STATIC_ROOT.resolve())
        except (OSError, ValueError):
            self._json({"status": "error", "error": "Not found"}, status=404)
            return
        self._raw_file(resolved)

    def _raw_file(self, path: Path):
        """Serve a trusted file without the STATIC_ROOT containment check."""
        if not path.is_file():
            self._json({"status": "error", "error": "Not found"}, status=404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="also rewrite the tracked agent catalog snapshot "
        "(.agent/skills/time-series-sandbox/catalog_snapshot.json); off by default "
        "so starting the server never dirties the working tree",
    )
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), PlaygroundHandler)
    url = f"http://{args.host}:{args.port}"

    import threading

    def _warm_catalog():
        # Build the catalog (disk-cached) in the background so the first
        # page load does not pay for discovery.
        try:
            build_catalog(include_registered=True)
            if args.refresh_snapshot:
                sys.stderr.write(f"Catalog snapshot refreshed: {write_snapshot()}\n")
        except Exception as exc:
            sys.stderr.write(f"Catalog warm-up skipped: {type(exc).__name__}: {exc}\n")

    threading.Thread(target=_warm_catalog, daemon=True).start()

    print(f"LibraryTS Playground running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LibraryTS Playground GUI.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
