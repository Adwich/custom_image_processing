from __future__ import annotations

import signal
import threading

from .config import ConfigError, load_config
from .db import Database
from .drive import GoogleDriveClient
from .ingest import ingest_from_drive
from .logging_events import EventLogger
from .observability import capture_exception, init_sentry, log_json, maybe_send_heartbeat
from .process import AssetProcessor
from .resolve import resolve_asset_dx_order, resolve_pending_links
from .storage import SupabaseStorageClient


def run_worker() -> None:
    service = "custom-worker"
    init_sentry(service=service)

    try:
        config = load_config(strict=True)
    except ConfigError as exc:
        log_json("error", "startup_failed", service=service, error=str(exc))
        raise SystemExit(f"Startup failed: {exc}") from exc

    db = Database(config.supabase_db_url)
    db.ensure_runtime_tables()

    logger = EventLogger(db)
    drive = GoogleDriveClient(config.gdrive_service_account_json, config.gdrive_folder_id)
    storage = SupabaseStorageClient(
        config.supabase_url,
        config.supabase_service_role_key,
        config.supabase_storage_bucket,
    )
    processor = AssetProcessor(config, db, storage, logger)

    stop_event = threading.Event()

    def _shutdown_handler(signum, _frame):
        log_json("info", "worker_shutdown_signal", service=service, signal=signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    log_json(
        "info",
        "worker_started",
        service=service,
        poll_interval_seconds=config.poll_interval_seconds,
        max_ingest_per_run=config.max_ingest_per_run,
        max_process_per_run=config.max_process_per_run,
        max_resolve_per_run=config.max_resolve_per_run,
    )

    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        ingest_count = 0
        resolved = 0
        processed = 0

        try:
            ingest_count = ingest_from_drive(
                config=config,
                db=db,
                drive=drive,
                storage=storage,
                logger=logger,
                resolve_asset_fn=lambda asset_id: resolve_asset_dx_order(
                    db, logger, config, asset_id
                ),
            )
        except Exception as exc:
            capture_exception(
                exc,
                service=service,
                cycle=cycle,
                stage="ingest_from_drive",
            )

        try:
            resolved = resolve_pending_links(
                db=db,
                logger=logger,
                config=config,
                limit=config.max_resolve_per_run,
            )
        except Exception as exc:
            capture_exception(
                exc,
                service=service,
                cycle=cycle,
                stage="resolve_pending_links",
            )

        try:
            processed = processor.process_images()
        except Exception as exc:
            capture_exception(
                exc,
                service=service,
                cycle=cycle,
                stage="process_images",
            )

        log_json(
            "info",
            "worker_cycle_completed",
            service=service,
            cycle=cycle,
            ingested=ingest_count,
            resolved=resolved,
            processed=processed,
            sleep_seconds=config.poll_interval_seconds,
        )

        maybe_send_heartbeat(
            service=service,
            jobs_claimed=(ingest_count + processed),
            poll_seconds=config.poll_interval_seconds,
        )

        stop_event.wait(config.poll_interval_seconds)

    log_json("info", "worker_stopped", service=service)


if __name__ == "__main__":
    run_worker()
