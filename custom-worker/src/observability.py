from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

_SENTRY_ENABLED = False
_SENTRY_SDK = None
_SENTRY_CAPTURE_CHECKIN = None
_LAST_HEARTBEAT_TS = 0.0


def _normalize_log_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseException):
        return {"type": value.__class__.__name__, "error": str(value)}
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def log_json(level: str, msg: str, **fields: Any) -> None:
    priority_keys = [
        "service",
        "cycle",
        "asset_id",
        "run_id",
        "id",
        "file_kind",
        "storage_path",
        "status",
        "error",
    ]

    ordered: dict[str, Any] = {"level": level, "msg": msg}
    for key in priority_keys:
        if key in fields:
            ordered[key] = _normalize_log_value(fields[key])
    ordered["ts"] = datetime.now(timezone.utc).isoformat()
    for key, value in fields.items():
        if key not in ordered:
            ordered[key] = _normalize_log_value(value)

    line = json.dumps(ordered, ensure_ascii=True, default=str)
    if level in {"error", "warn"}:
        print(line, file=sys.stderr, flush=True)
    else:
        print(line, flush=True)


def init_sentry(service: str) -> bool:
    global _SENTRY_ENABLED, _SENTRY_SDK, _SENTRY_CAPTURE_CHECKIN

    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return False

    try:
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0"))
    except ValueError:
        traces_sample_rate = 0.0

    environment = (
        os.getenv("SENTRY_ENVIRONMENT")
        or os.getenv("FLY_APP_NAME")
        or os.getenv("ENV")
        or os.getenv("NODE_ENV")
        or "production"
    )
    release = (
        os.getenv("SENTRY_RELEASE")
        or os.getenv("FLY_IMAGE_REF")
        or os.getenv("FLY_DEPLOYMENT_ID")
        or os.getenv("FLY_MACHINE_ID")
        or os.getenv("GIT_SHA")
    )

    try:
        import sentry_sdk
        from sentry_sdk.crons import capture_checkin
    except Exception as exc:
        log_json(
            "warn",
            "sentry_import_failed",
            service=service,
            error=str(exc),
        )
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=traces_sample_rate,
    )
    sentry_sdk.set_tag("service", service)
    _SENTRY_ENABLED = True
    _SENTRY_SDK = sentry_sdk
    _SENTRY_CAPTURE_CHECKIN = capture_checkin

    log_json(
        "info",
        "sentry_initialized",
        service=service,
        environment=environment,
        release=release,
        traces_sample_rate=traces_sample_rate,
    )
    return True


def capture_exception(err: Exception, **fields: Any) -> None:
    if _SENTRY_ENABLED and _SENTRY_SDK is not None:
        _SENTRY_SDK.capture_exception(err)
    log_json("error", "worker_exception", error=str(err), **fields)


def maybe_send_heartbeat(service: str, jobs_claimed: int, poll_seconds: int) -> None:
    global _LAST_HEARTBEAT_TS

    if not _SENTRY_ENABLED or _SENTRY_CAPTURE_CHECKIN is None:
        return

    interval_seconds = int(os.getenv("SENTRY_HEARTBEAT_INTERVAL_SECONDS", "60"))
    if interval_seconds <= 0:
        return

    now = time.time()
    if (now - _LAST_HEARTBEAT_TS) < interval_seconds:
        return

    interval_minutes = max(1, int(round(interval_seconds / 60)))
    monitor_slug = os.getenv("SENTRY_MONITOR_SLUG", "custom-worker")

    try:
        _SENTRY_CAPTURE_CHECKIN(
            monitor_slug=monitor_slug,
            status="ok",
            monitor_config={
                "schedule": {
                    "type": "interval",
                    "value": interval_minutes,
                    "unit": "minute",
                },
                "checkin_margin": 2,
            },
        )
        _LAST_HEARTBEAT_TS = now
        log_json(
            "info",
            "worker_heartbeat",
            service=service,
            jobs_claimed=jobs_claimed,
            poll_seconds=poll_seconds,
            heartbeat_interval_seconds=interval_seconds,
            monitor_slug=monitor_slug,
        )
    except Exception as exc:
        capture_exception(
            exc,
            service=service,
            stage="sentry_heartbeat",
            monitor_slug=monitor_slug,
        )
