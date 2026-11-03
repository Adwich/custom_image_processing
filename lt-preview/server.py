"""HTTP front for the live cut-out preview.

  GET  /health                     -> {"ok": true}
  POST /preview  {"slot_id": uuid} -> {"slot_id", "cuts": {"Huvud": data-URL, ...}, "flags", "faces", "head_route", "ms"}

The photo is never sent by the browser: the slot must exist in custom_config_slots (lemon-tree-uploads service,
status uploaded/bound) and the file is read from the private bucket with the service's secret key. Storefront
origins only (same list as the upload service). Results are cached by the file's sha256, so repeat taps and
"same photo for all" never pay PhotoRoom twice.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import cutout

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SECRET = os.environ["LT_SUPABASE_SECRET_KEY"]
ORIGINS = [o.strip() for o in os.environ.get("LT_ALLOWED_ORIGINS", "").split(",") if o.strip()]
BUCKET = "lemontree-uploads"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
ADMIN = os.environ.get("LT_PREVIEW_ADMIN_KEY", "")  # x-lt-admin: skip the cache (timing, re-checks)
RATE = (20, 600)  # previews per client IP per 10 minutes
CACHE_MAX = 300

_cache: OrderedDict[str, dict] = OrderedDict()
_cache_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}
_hits: dict[str, deque] = defaultdict(deque)


def log(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def _sb_headers() -> dict:
    return {"apikey": SECRET, "Authorization": f"Bearer {SECRET}"}


def slot(slot_id: str) -> dict | None:
    r = requests.post(f"{SUPABASE_URL}/rest/v1/rpc/lt_slots_get", json={"p_ids": [slot_id]},
                      headers={**_sb_headers(), "content-type": "application/json"}, timeout=10)
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


def photo_bytes(path: str) -> bytes:
    r = requests.get(f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}", headers=_sb_headers(), timeout=30)
    r.raise_for_status()
    return r.content


def allowed(origin: str | None) -> bool:
    return bool(origin) and origin in ORIGINS


def rate_ok(ip: str) -> bool:
    n, window = RATE
    q, now = _hits[ip], time.time()
    while q and q[0] < now - window:
        q.popleft()
    if len(q) >= n:
        return False
    q.append(now)
    return True


def compute(key: str, path: str, fresh: bool = False) -> dict:
    """One computation per file at a time; later callers wait for it and read the cache."""
    with _cache_lock:
        if fresh:
            _cache.pop(key, None)
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        ev = _inflight.get(key)
        if ev is None:
            ev = _inflight[key] = threading.Event()
            mine = True
        else:
            mine = False
    if not mine:
        ev.wait(60)
        with _cache_lock:
            if key in _cache:
                return _cache[key]
        raise cutout.PreviewError("preview failed")
    try:
        res = cutout.previews(photo_bytes(path))
        body = {**{k: v for k, v in res.items() if k != "cuts"},
                "cuts": {n: "data:image/webp;base64," + base64.b64encode(b).decode() for n, b in res["cuts"].items()}}
        with _cache_lock:
            _cache[key] = body
            while len(_cache) > CACHE_MAX:
                _cache.popitem(last=False)
        return body
    finally:
        with _cache_lock:
            _inflight.pop(key, None)
        ev.set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # structured logs only
        pass

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        origin = self.headers.get("origin")
        if allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204 if allowed(self.headers.get("origin")) else 403, {})

    def do_GET(self):
        self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/preview":
            return self._send(404, {"error": "not_found"})
        if not allowed(self.headers.get("origin")):
            return self._send(403, {"error": "origin"})
        ip = self.headers.get("fly-client-ip") or self.client_address[0]
        if not rate_ok(ip):
            return self._send(429, {"error": "rate"})
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("content-length") or 0), 4096)) or b"{}")
        except ValueError:
            return self._send(400, {"error": "json"})
        sid = str(body.get("slot_id", ""))
        if not UUID.match(sid):
            return self._send(400, {"error": "slot_id"})
        t = time.perf_counter()
        try:
            s = slot(sid)
            if not s or s.get("status") not in ("uploaded", "bound") or not s.get("upload_path"):
                return self._send(404, {"error": "unknown_slot"})
            fresh = bool(ADMIN) and self.headers.get("x-lt-admin") == ADMIN
            res = compute(s.get("content_hash") or sid, s["upload_path"], fresh)
        except cutout.PreviewError as exc:
            log("preview_failed", slot=sid, reason=str(exc))
            return self._send(422, {"error": "preview_failed", "reason": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log("preview_error", slot=sid, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return self._send(500, {"error": "internal"})
        log("preview", slot=sid, flags=res["flags"], route=res["head_route"], ms=res["ms"],
            wall_ms=round((time.perf_counter() - t) * 1000))
        self._send(200, {"slot_id": sid, **res})


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    t = time.perf_counter()
    cutout.warm_up()  # load both models before the first shopper, not during their preview
    log("start", port=port, origins=ORIGINS, warm_ms=round((time.perf_counter() - t) * 1000))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
