from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .config import AppConfig
from .db import Database


def _normalize_asset_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        trimmed = item.strip()
        if trimmed:
            out.append(trimmed)
    return out


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


class WorkerControlApi:
    def __init__(self, config: AppConfig, db: Database):
        self._config = config
        self._db = db
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._config.control_api_enabled:
            return
        if self._server is not None:
            return

        handler = self._make_handler()
        server = ThreadingHTTPServer(
            (self._config.control_api_host, self._config.control_api_port),
            handler,
        )
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="worker-control-api",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        db = self._db
        token = self._config.control_api_token

        def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
            length = int(handler.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = handler.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}

        def _write(
            handler: BaseHTTPRequestHandler,
            status: int,
            payload: dict[str, Any],
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

        def _authorized(handler: BaseHTTPRequestHandler) -> bool:
            if not token:
                return True
            header = handler.headers.get("Authorization", "")
            return header == f"Bearer {token}"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if not _authorized(self):
                    _write(self, HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
                    return

                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/")
                query = parse_qs(parsed.query)

                if path == "" or path == "/":
                    _write(self, HTTPStatus.OK, {"ok": True})
                    return
                if path == "/health":
                    _write(self, HTTPStatus.OK, {"ok": True})
                    return

                if path == "/exports":
                    raw_limit = query.get("limit", ["20"])[0]
                    try:
                        limit = max(1, min(100, int(raw_limit)))
                    except Exception:
                        limit = 20
                    items = db.list_export_jobs(limit=limit)
                    _write(self, HTTPStatus.OK, {"items": _to_jsonable(items)})
                    return

                if path.startswith("/exports/"):
                    job_id = path.split("/", 2)[2]
                    item = db.fetch_export_job(job_id)
                    if not item:
                        _write(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
                        return
                    _write(self, HTTPStatus.OK, {"item": _to_jsonable(item)})
                    return

                _write(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})

            def do_POST(self) -> None:
                if not _authorized(self):
                    _write(self, HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
                    return

                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/")
                if path != "/exports":
                    _write(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
                    return

                body = _read_json(self)
                requested_by = body.get("requested_by")
                requested_by = requested_by if isinstance(requested_by, str) else None
                asset_ids = _normalize_asset_ids(body.get("asset_ids"))
                wait_seconds_raw = body.get("wait_seconds", 0)
                try:
                    wait_seconds = max(0, min(300, int(wait_seconds_raw)))
                except Exception:
                    wait_seconds = 0

                request_payload: dict[str, Any] = {}
                if asset_ids:
                    request_payload["asset_ids"] = asset_ids

                job_id = str(uuid4())
                item = db.create_export_job(
                    job_id=job_id,
                    request_payload=request_payload,
                    requested_by=requested_by,
                )

                if wait_seconds > 0:
                    deadline = time.monotonic() + float(wait_seconds)
                    current = item
                    while time.monotonic() < deadline:
                        status = str(current.get("status") or "")
                        if status in {"completed", "failed"}:
                            break
                        time.sleep(1.0)
                        latest = db.fetch_export_job(job_id)
                        if not latest:
                            break
                        current = latest
                    _write(self, HTTPStatus.OK, {"item": _to_jsonable(current)})
                    return

                _write(self, HTTPStatus.ACCEPTED, {"item": _to_jsonable(item)})

        return Handler
