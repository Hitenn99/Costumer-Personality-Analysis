"""Local web server for the Customer Personality Analysis dashboard.

Serves dashboard/ and adds a small JSON API so a new dataset can be uploaded from the
"Dataset" page. Each upload is validated, saved to data/uploads/ and the ML pipeline is
re-run in a subprocess; the browser polls /api/status and reloads when it finishes.

    python server.py            # then open http://localhost:8000

API
    GET  /api/status            pipeline job state + progress
    POST /api/upload            raw file bytes in the body, file name in X-Filename
    POST /api/restore           re-run the pipeline on the original dataset
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "dashboard"
UPLOADS = ROOT / "data" / "uploads"
ORIGINAL = ROOT / "data" / "marketing_campaign_modified.csv"
PIPELINE = ROOT / "ml" / "run_pipeline.py"
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 8000))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXT = {".csv", ".tsv", ".txt"}

sys.path.insert(0, str(ROOT / "ml"))
from preprocess import MIN_ROWS, read_table, validate_columns  # noqa: E402

# Progress milestones parsed from pipeline output (prefix -> percent, label).
STAGES = [
    ("[load]", 5, "Loading data"),
    ("[clean]", 10, "Cleaning & engineering features"),
    ("[features]", 15, "Scaling features"),
    ("[autoencoder] epoch", None, "Training autoencoder"),
    ("[autoencoder] trained", 75, "Autoencoder trained"),
    ("[kmeans]", 80, "Clustering latent space (K-Means)"),
    ("[analysis]", 90, "Profiling segments & AI analysis"),
    ("[propensity]", 94, "Response model trained"),
    ("[output]", 98, "Writing dashboard data"),
]


class Job:
    """Single pipeline job shared between request threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | done | error
        self.dataset = None
        self.progress = 0
        self.stage = ""
        self.message = ""
        self.log: list[str] = []
        self.started = self.finished = None

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "dataset": self.dataset, "progress": self.progress,
                    "stage": self.stage, "message": self.message, "log": self.log[-12:],
                    "started": self.started, "finished": self.finished}

    def start(self, data_path: Path, display_name: str) -> bool:
        with self.lock:
            if self.state == "running":
                return False
            self.state, self.dataset, self.progress = "running", display_name, 1
            self.stage, self.message, self.log = "Starting pipeline", "", []
            self.started, self.finished = time.time(), None
        threading.Thread(target=self._run, args=(data_path, display_name), daemon=True).start()
        return True

    def _update_progress(self, line: str) -> None:
        for prefix, pct, label in STAGES:
            if line.startswith(prefix):
                if pct is None:  # "[autoencoder] epoch 40/150 ..."
                    m = re.search(r"epoch (\d+)/(\d+)", line)
                    pct = 15 + int(55 * int(m.group(1)) / int(m.group(2))) if m else self.progress
                    label = f"Training autoencoder ({m.group(1)}/{m.group(2)} epochs)" if m else label
                self.progress, self.stage = max(self.progress, pct), label

    def _run(self, data_path: Path, display_name: str) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONWARNINGS": "ignore"}
        cmd = [sys.executable, str(PIPELINE), "--data", str(data_path), "--name", display_name]
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            error_line = ""
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                print(f"  pipeline | {line}")
                with self.lock:
                    self.log.append(line)
                    self._update_progress(line)
                    if line.startswith("[error]"):
                        error_line = line.removeprefix("[error]").strip()
            code = proc.wait()
        except Exception as exc:  # e.g. interpreter not found
            code, error_line = 1, str(exc)
        with self.lock:
            self.finished = time.time()
            if code == 0:
                self.state, self.progress, self.stage = "done", 100, "Dashboard updated"
                self.message = f"Dashboard rebuilt from {display_name}."
            else:
                self.state, self.stage = "error", "Pipeline failed"
                tail = next((l for l in reversed(self.log) if "Error" in l or "error" in l), "")
                self.message = error_line or tail or f"Pipeline exited with code {code}."


JOB = Job()


def _safe_name(name: str) -> str:
    name = Path(name).name or "upload.csv"
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:100]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD), **kwargs)

    # Generated data must never be served stale after a re-run.
    def end_headers(self) -> None:
        if self.path.startswith("/data/") or self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:
        if "/api/status" not in self.path:
            super().log_message(fmt, *args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/api/status":
            return self._json(HTTPStatus.OK, JOB.snapshot())
        return super().do_GET()

    def do_POST(self) -> None:
        route = self.path.split("?")[0]
        if route == "/api/upload":
            return self._upload()
        if route == "/api/restore":
            if not JOB.start(ORIGINAL, ORIGINAL.name):
                return self._json(HTTPStatus.CONFLICT, {"error": "A pipeline run is already in progress."})
            return self._json(HTTPStatus.ACCEPTED, {"ok": True})
        return self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

    def _upload(self) -> None:
        if JOB.snapshot()["state"] == "running":
            return self._json(HTTPStatus.CONFLICT, {"error": "A pipeline run is already in progress."})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "The file is empty."})
        if length > MAX_UPLOAD_BYTES:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                              {"error": f"File is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB."})
        name = _safe_name(unquote(self.headers.get("X-Filename", "upload.csv")))
        if Path(name).suffix.lower() not in ALLOWED_EXT:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "Please upload a .csv, .tsv or .txt file."})

        UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
        dest.write_bytes(self.rfile.read(length))

        try:
            df = read_table(dest)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            return self._json(HTTPStatus.BAD_REQUEST, {"error": f"Could not read the file as CSV/TSV: {exc}"})
        missing = validate_columns(df.columns)
        if missing:
            dest.unlink(missing_ok=True)
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "Missing required columns: " + ", ".join(missing),
                                                       "missing": missing})
        if len(df) < MIN_ROWS:
            dest.unlink(missing_ok=True)
            return self._json(HTTPStatus.BAD_REQUEST,
                              {"error": f"The file has {len(df)} rows; at least {MIN_ROWS} are needed."})

        if not JOB.start(dest, name):
            return self._json(HTTPStatus.CONFLICT, {"error": "A pipeline run is already in progress."})
        return self._json(HTTPStatus.ACCEPTED, {"ok": True, "rows": len(df), "columns": len(df.columns)})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Customer dashboard running at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == "__main__":
    main()
