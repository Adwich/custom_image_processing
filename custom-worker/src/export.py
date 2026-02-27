from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone
from typing import Any

from .config import AppConfig
from .db import Database
from .logging_events import EventLogger
from .storage import SupabaseStorageClient


_SAFE_PATH_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _safe_part(value: str) -> str:
    cleaned = _SAFE_PATH_RE.sub("_", value.strip())
    return cleaned or "unknown"


class ExportProcessor:
    def __init__(
        self,
        config: AppConfig,
        db: Database,
        storage: SupabaseStorageClient,
        logger: EventLogger,
    ):
        self.config = config
        self.db = db
        self.storage = storage
        self.logger = logger

    def process_exports(self) -> int:
        jobs = self.db.claim_export_jobs(self.config.max_export_per_run)
        if not jobs:
            return 0

        for job in jobs:
            job_id = str(job["id"])
            payload = job.get("request_payload")
            try:
                self.logger.log_event(
                    entity_type="export",
                    entity_id=job_id,
                    event_type="export_started",
                    event_data={
                        "job_id": job_id,
                        "requested_by": job.get("requested_by"),
                        "request_payload": payload or {},
                    },
                )
                self._process_one(job_id, payload)
            except Exception as exc:
                self.db.mark_export_job_failed(job_id=job_id, error=str(exc))
                self.logger.log_error(
                    error_type="export_failed",
                    message=str(exc),
                    severity="error",
                    status="open",
                    context={
                        "job_id": job_id,
                        "request_payload": payload or {},
                    },
                )
                self.logger.log_event(
                    entity_type="export",
                    entity_id=job_id,
                    event_type="export_failed",
                    event_data={"job_id": job_id, "error": str(exc)},
                )
        return len(jobs)

    def _extract_asset_ids(self, payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            return []
        raw = payload.get("asset_ids")
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for value in raw:
            if not isinstance(value, str):
                continue
            v = value.strip()
            if _UUID_RE.match(v):
                out.append(v)
        return out

    def _process_one(self, job_id: str, payload: Any) -> None:
        asset_ids = self._extract_asset_ids(payload)
        assets = self.db.fetch_assets_for_export(asset_ids=asset_ids or None)
        if not assets:
            raise RuntimeError("No exportable assets found (requires status='processed').")

        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        zip_storage_path = f"exports/{job_id}/custom_export_{now}.zip"

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for asset in assets:
                asset_id = str(asset["id"])
                client_order_id = str(asset.get("client_order_id") or "unknown")
                source_path = str(asset["processed_storage_path"])
                file_bytes = self.storage.download_bytes(source_path)
                ext = ".png"
                if "." in source_path.rsplit("/", 1)[-1]:
                    ext = "." + source_path.rsplit(".", 1)[-1]
                zip_name = (
                    f"{_safe_part(client_order_id)}/"
                    f"{_safe_part(asset_id)}_final{_safe_part(ext).replace('_', '.')}"
                )
                zf.writestr(zip_name, file_bytes)

        zip_bytes = archive.getvalue()
        self.storage.upload_bytes(
            path=zip_storage_path,
            data=zip_bytes,
            content_type="application/zip",
        )
        signed_url = self.storage.create_signed_url(
            path=zip_storage_path,
            expires_in_seconds=self.config.export_signed_url_ttl_seconds,
        )

        self.db.mark_export_job_completed(
            job_id=job_id,
            zip_storage_path=zip_storage_path,
            zip_signed_url=signed_url,
            asset_count=len(assets),
        )
        self.logger.log_event(
            entity_type="export",
            entity_id=job_id,
            event_type="export_completed",
            event_data={
                "job_id": job_id,
                "asset_count": len(assets),
                "zip_storage_path": zip_storage_path,
                "zip_signed_url": signed_url,
            },
        )
