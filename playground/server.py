#!/usr/bin/env python3
"""Local HTTP server for the TSBox Sandbox Playground."""

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
from runners import PlaygroundError, get_run, run_experiment

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PlaygroundHandler(BaseHTTPRequestHandler):
    server_version = "TSBoxSandboxPlayground/1.0"

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
        if parsed.path in {"/", "/index.html"}:
            self._file(STATIC_ROOT / "index.html")
            return
        self._file(STATIC_ROOT / parsed.path.lstrip("/"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._json({"status": "error", "error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            spec = json.loads(body) if body else {}
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
        if not resolved.is_file():
            self._json({"status": "error", "error": "Not found"}, status=404)
            return
        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
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
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), PlaygroundHandler)
    url = f"http://{args.host}:{args.port}"

    import threading

    def _refresh_snapshot():
        try:
            snapshot = write_snapshot()
            sys.stderr.write(f"Catalog snapshot refreshed: {snapshot}\n")
        except Exception as exc:
            sys.stderr.write(
                f"Catalog snapshot skipped: {type(exc).__name__}: {exc}\n"
            )

    threading.Thread(target=_refresh_snapshot, daemon=True).start()

    print(f"TSBox Sandbox Playground running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Playground.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

