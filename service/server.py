"""
service/server.py

Local demo server (standard library only, no new dependencies). Serves the demo page in service/static and a small JSON API
around RoleDiscrepancyScorer (service/scorer.py):

  GET  /api/health      which detector is loaded, whether one is loading
  GET  /api/cases       service/demo_cases.json (32 cases from real data, seeded random selection, failures kept)
  GET  /api/thresholds  service/thresholds.json (pooled and per-channel thresholds with their measured behaviour)
  POST /api/score       {text, declared, condition, policy, carrier?} -> bare result (+ wrapped result if a carrier is chosen)

Memory: one DeBERTa-base ensemble (3 seeds) is about 3.3 GB peak on this machine, so only ONE condition is held in memory;
switching condition unloads the other and takes about 16 s (the page tells the user). Requests are serialized by a lock.
Binds to 127.0.0.1 only. Run:  python -m service.server [--port 8765] [--condition naive]
"""

from __future__ import annotations

import argparse
import gc
import json
import mimetypes
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from service.scorer import RoleDiscrepancyScorer
from src.ingestion.schema import Role
from src.supervision.carriers import embed_in_channel

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
CONDITIONS = ("naive", "transplant")
POLICIES = ("pooled", "per_channel")
MAX_CHARS = 4000


class Holder:
    """Holds at most one loaded ensemble. `factory(condition)` builds a scorer; injected so tests can use fakes."""

    def __init__(self, factory=RoleDiscrepancyScorer.from_checkpoints):
        self.factory, self.scorer, self.condition, self.loading = factory, None, None, False
        self.lock = threading.RLock()   # re-entrant: handle_score holds it for the whole request, get() takes it again

    def get(self, condition: str):
        with self.lock:
            if self.condition != condition:
                self.scorer, self.condition = None, None
                gc.collect()                      # release the other ensemble before loading this one
                self.loading = True
                try:
                    self.scorer, self.condition = self.factory(condition), condition
                finally:
                    self.loading = False
            return self.scorer


def parse_score_request(payload: dict) -> dict:
    """Validate and normalise a /api/score body. Raises ValueError with a user-readable message."""
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to score.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text is too long ({len(text)} characters); the limit is {MAX_CHARS}.")
    declared = payload.get("declared")
    if declared not in {r.value for r in Role}:
        raise ValueError("Pick a declared channel.")
    condition = payload.get("condition", "naive")
    if condition not in CONDITIONS:
        raise ValueError("Unknown detector.")
    policy = payload.get("policy", "per_channel")
    if policy not in POLICIES:
        raise ValueError("Unknown operating point.")
    carrier = payload.get("carrier") or None
    if carrier is not None and carrier not in {r.value for r in Role}:
        raise ValueError("Unknown carrier.")
    return {"text": text, "declared": declared, "condition": condition, "policy": policy, "carrier": carrier}


def wrap_in_carrier(text: str, channel: str) -> str:
    """Deterministic carrier wrap using the project's own (training) carrier templates: seed 0, so the same text always wraps the same way."""
    return embed_in_channel(text, Role(channel), random.Random(0))


def handle_score(payload: dict, holder: Holder) -> dict:
    req = parse_score_request(payload)
    with holder.lock:                                    # whole request serialized: no request can hold the old ensemble during a switch
        scorer = holder.get(req["condition"])
        out = {"bare": scorer.score(req["text"], req["declared"], req["policy"]), "wrapped": None, "wrapped_text": None,
               "truncation_note": len(req["text"]) > 300}
        if req["carrier"]:
            wrapped = wrap_in_carrier(req["text"], req["carrier"])
            out["wrapped_text"] = wrapped
            out["wrapped"] = scorer.score(wrapped, req["declared"], req["policy"])
        del scorer
    return out


def make_handler(holder: Holder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/health":
                return self._json(200, {"loaded": holder.condition, "loading": holder.loading})
            if path == "/api/cases":
                return self._json(200, json.loads((HERE / "demo_cases.json").read_text()))
            if path == "/api/thresholds":
                return self._json(200, json.loads((HERE / "thresholds.json").read_text()))
            rel = "index.html" if path in ("/", "") else path.lstrip("/").removeprefix("static/")
            target = (STATIC / rel).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                return self._json(404, {"error": "not found"})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text/") or "javascript" in ctype else ""))

        def do_POST(self):
            if self.path != "/api/score":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                return self._json(200, handle_score(payload, holder))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except FileNotFoundError as e:
                return self._json(500, {"error": f"Detector files are missing: {e}"})
            except Exception as e:  # keep the demo alive; the page shows the message
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--condition", default="naive", choices=CONDITIONS, help="detector to preload in the background")
    args = ap.parse_args()
    holder = Holder()
    threading.Thread(target=lambda: holder.get(args.condition), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(holder))
    print(f"demo on http://127.0.0.1:{args.port}  (preloading '{args.condition}' detector)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
