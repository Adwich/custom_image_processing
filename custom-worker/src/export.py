from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone
from typing import Any

from PIL import Image, ImageDraw, ImageFont

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


def _alpha_suffix(position: int) -> str:
    # 1 -> a, 2 -> b, ... 26 -> z, 27 -> aa
    if position <= 0:
        return "a"
    chars: list[str] = []
    value = position
    while value > 0:
        value, rem = divmod(value - 1, 26)
        chars.append(chr(ord("a") + rem))
    return "".join(reversed(chars))


def _as_positive_quantity(value: Any) -> int:
    try:
        qty = int(value)
    except Exception:
        return 1
    return qty if qty > 0 else 1


def _as_scent(value: Any) -> str:
    scent = str(value).strip() if value is not None else ""
    return scent or "no_scent"


def _barcode_png_with_labels(dx_order_id: str, scent: str) -> bytes:
    try:
        from barcode import Code128
        from barcode.writer import ImageWriter
    except Exception as exc:
        raise RuntimeError(
            "python-barcode is required for export barcode generation"
        ) from exc

    writer_options = {
        "module_width": 0.33,
        "module_height": 22.0,
        "quiet_zone": 2.0,
        "font_size": 0,
        "text_distance": 0,
        "write_text": False,
    }
    base = Code128(dx_order_id, writer=ImageWriter()).render(writer_options).convert("RGBA")
    width, height = base.size

    font = ImageFont.load_default()
    pad_x = 16
    pad_y = 10
    line_gap = 4
    label_1 = f"DX: {dx_order_id}"
    label_2 = f"Scent: {scent}"
    d = ImageDraw.Draw(base)
    box_1 = d.textbbox((0, 0), label_1, font=font)
    box_2 = d.textbbox((0, 0), label_2, font=font)
    text_h = (box_1[3] - box_1[1]) + line_gap + (box_2[3] - box_2[1])
    text_w = max(box_1[2] - box_1[0], box_2[2] - box_2[0])

    out_w = max(width + (pad_x * 2), text_w + (pad_x * 2))
    out_h = height + (pad_y * 2) + text_h
    out = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))

    barcode_x = (out_w - width) // 2
    out.paste(base, (barcode_x, pad_y), base)

    draw = ImageDraw.Draw(out)
    text_y = pad_y + height + 2
    draw.text((pad_x, text_y), label_1, fill=(0, 0, 0, 255), font=font)
    draw.text(
        (pad_x, text_y + (box_1[3] - box_1[1]) + line_gap),
        label_2,
        fill=(0, 0, 0, 255),
        font=font,
    )

    payload = io.BytesIO()
    out.save(payload, format="PNG")
    return payload.getvalue()


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
        folder_counts: dict[tuple[str, str], int] = {}
        with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for asset in assets:
                dx_order_id = str(asset.get("dx_order_id") or "").strip()
                if not dx_order_id:
                    raise RuntimeError("Asset missing dx_order_id; cannot build export tree")

                source_path = str(asset["processed_storage_path"])
                file_bytes = self.storage.download_bytes(source_path)
                ext = ".png"
                if "." in source_path.rsplit("/", 1)[-1]:
                    ext = "." + source_path.rsplit(".", 1)[-1]
                ext = _safe_part(ext).replace("_", ".")

                scent = _as_scent(asset.get("scent"))
                qty = _as_positive_quantity(asset.get("quantity"))
                order_folder = _safe_part(dx_order_id)
                item_base = _safe_part(f"{scent}_{qty}x")

                key = (order_folder, item_base)
                idx = folder_counts.get(key, 0) + 1
                folder_counts[key] = idx
                item_folder = item_base if idx == 1 else f"{item_base}_{_alpha_suffix(idx - 1)}"

                image_zip_path = f"{order_folder}/{item_folder}/image{ext}"
                barcode_zip_path = f"{order_folder}/{item_folder}/barcode.png"
                barcode_bytes = _barcode_png_with_labels(dx_order_id=dx_order_id, scent=scent)

                zf.writestr(image_zip_path, file_bytes)
                zf.writestr(barcode_zip_path, barcode_bytes)

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
