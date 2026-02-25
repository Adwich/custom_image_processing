from __future__ import annotations

import mimetypes
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import AppConfig
from .db import Database
from .drive import DriveFile, GoogleDriveClient
from .logging_events import EventLogger
from .storage import SupabaseStorageClient


SAFE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MIME_EXTENSION_MAP = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


@dataclass(frozen=True)
class ParsedDriveMeta:
    client_order_id: Optional[str]
    quantity: Optional[int]
    cut_option: Optional[str]


def parse_client_order_id(drive_path: str, file_name: str) -> Optional[str]:
    path = drive_path.strip("/")
    if path:
        top = path.split("/", 1)[0].strip()
        if top:
            return top

    match = re.match(r"^([^-_/]+)-", file_name)
    if match:
        return match.group(1).strip()
    return None


def parse_quantity(drive_path: str, file_name: str) -> Optional[int]:
    path_parts = [p for p in drive_path.strip("/").split("/") if p]
    if len(path_parts) >= 2:
        match = re.match(r"^(\d+)x-", path_parts[1], re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"-(\d+)(?:\.[^.]+)?$", file_name)
    if match:
        return int(match.group(1))
    return None


def parse_cut_option(drive_path: str, file_name: str) -> Optional[str]:
    hay = f"{drive_path} {file_name}".lower()
    for option in ("head", "body", "car", "none"):
        if re.search(rf"\b{option}\b", hay):
            return option
    return None


def parse_drive_metadata(drive_path: str, file_name: str) -> ParsedDriveMeta:
    return ParsedDriveMeta(
        client_order_id=parse_client_order_id(drive_path, file_name),
        quantity=parse_quantity(drive_path, file_name),
        cut_option=parse_cut_option(drive_path, file_name),
    )


def safe_extension_from_drive(mime_type: str, file_name: str) -> str:
    mime_ext = MIME_EXTENSION_MAP.get(mime_type.lower())
    if mime_ext:
        return mime_ext

    guessed, _ = mimetypes.guess_type(file_name)
    if guessed and guessed in MIME_EXTENSION_MAP:
        return MIME_EXTENSION_MAP[guessed]

    if "." in file_name:
        ext = file_name.rsplit(".", 1)[1].lower()
        if ext in SAFE_EXTENSIONS:
            return "jpg" if ext == "jpeg" else ext

    return "jpg"


def _max_iso(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if not a:
        return b
    if not b:
        return a
    pa = datetime.fromisoformat(a.replace("Z", "+00:00")).astimezone(timezone.utc)
    pb = datetime.fromisoformat(b.replace("Z", "+00:00")).astimezone(timezone.utc)
    return a if pa >= pb else b


def ingest_from_drive(
    config: AppConfig,
    db: Database,
    drive: GoogleDriveClient,
    storage: SupabaseStorageClient,
    logger: EventLogger,
    resolve_asset_fn,
) -> int:
    cursor = db.get_kv(config.gdrive_cursor_kv_key)
    files = drive.list_files_since(cursor, config.max_ingest_per_run)
    if not files:
        return 0

    ingested_count = 0
    newest_cursor = cursor

    for file in files:
        newest_cursor = _max_iso(newest_cursor, file.modified_time)
        try:
            inserted = _ingest_single_file(config, db, drive, storage, logger, resolve_asset_fn, file)
            if inserted:
                ingested_count += 1
        except Exception as exc:
            logger.log_error(
                error_type="ingest_failed",
                message=str(exc),
                severity="error",
                client_order_id=None,
                context={
                    "drive_file_id": file.file_id,
                    "drive_path": file.drive_path,
                    "file_name": file.name,
                },
            )

    if newest_cursor:
        db.set_kv(config.gdrive_cursor_kv_key, newest_cursor)

    return ingested_count


def _ingest_single_file(
    config: AppConfig,
    db: Database,
    drive: GoogleDriveClient,
    storage: SupabaseStorageClient,
    logger: EventLogger,
    resolve_asset_fn,
    file: DriveFile,
) -> bool:
    if db.asset_exists_for_drive_file(file.file_id):
        return False

    parsed = parse_drive_metadata(file.drive_path, file.name)
    if not parsed.client_order_id:
        logger.log_error(
            error_type="ingest_parse_failed",
            message="Could not parse client_order_id from drive path or filename",
            severity="error",
            context={
                "drive_file_id": file.file_id,
                "drive_path": file.drive_path,
                "file_name": file.name,
            },
        )
        return False

    file_bytes = drive.download_file(file.file_id)
    asset_id = str(uuid.uuid4())
    extension = safe_extension_from_drive(file.mime_type, file.name)
    original_path = f"{parsed.client_order_id}/original/{asset_id}.{extension}"

    storage.upload_bytes(original_path, file_bytes, file.mime_type or "application/octet-stream")

    status = "ingested" if parsed.cut_option else "needs_metadata"
    inserted = db.insert_asset(
        {
            "id": asset_id,
            "client_order_id": parsed.client_order_id,
            "drive_file_id": file.file_id,
            "drive_path": file.drive_path,
            "original_storage_path": original_path,
            "cut_option": parsed.cut_option,
            "scent": None,
            "quantity": parsed.quantity,
            "status": status,
            "error": None,
        }
    )
    if not inserted:
        return False

    logger.log_event(
        entity_type="asset",
        entity_id=asset_id,
        event_type="asset_ingested",
        event_data={
            "asset_id": asset_id,
            "client_order_id": parsed.client_order_id,
            "drive_file_id": file.file_id,
            "drive_path": file.drive_path,
            "original_storage_path": original_path,
            "quantity": parsed.quantity,
            "cut_option": parsed.cut_option,
            "status": status,
        },
    )

    if status == "needs_metadata":
        logger.log_error(
            error_type="needs_metadata",
            message="Asset ingested without cut_option metadata",
            severity="warning",
            asset_id=asset_id,
            client_order_id=parsed.client_order_id,
            context={
                "drive_file_id": file.file_id,
                "drive_path": file.drive_path,
            },
        )

    resolve_asset_fn(asset_id)
    return True
