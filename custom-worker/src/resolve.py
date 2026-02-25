from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import AppConfig
from .db import Database
from .logging_events import EventLogger


def _serialize_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for row in rows:
        created_at = row.get("created_at")
        if isinstance(created_at, datetime):
            created_text = created_at.isoformat()
        else:
            created_text = str(created_at)
        serialized.append(
            {
                "dx_order_id": row.get("dx_order_id"),
                "order_status_en": row.get("order_status_en"),
                "created_at": created_text,
            }
        )
    return serialized


def resolve_asset_dx_order(
    db: Database,
    logger: EventLogger,
    config: AppConfig,
    asset_id: str,
    allow_re_resolve: bool = False,
) -> None:
    asset = db.fetch_asset(asset_id)
    if not asset:
        return

    client_order_id = (asset.get("client_order_id") or "").strip()
    current_dx = asset.get("dx_order_id")

    if current_dx and not allow_re_resolve:
        return

    if not client_order_id:
        logger.log_error(
            error_type="asset_missing_client_order_id",
            message="Asset missing client_order_id; cannot resolve dx_order_id",
            severity="error",
            asset_id=asset_id,
            context={"asset_id": asset_id},
        )
        return

    candidates = db.fetch_orders_by_client_order_id(client_order_id)
    eligible_set = set(config.eligible_order_status_en)
    eligible = [
        row for row in candidates if str(row.get("order_status_en")) in eligible_set
    ]

    summary = {
        "client_order_id": client_order_id,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "eligible_allowlist": list(config.eligible_order_status_en),
    }

    if len(eligible) == 1:
        chosen = eligible[0]
        dx_order_id = str(chosen["dx_order_id"])
        order_status_en = str(chosen["order_status_en"])

        db.update_asset_resolution_success(asset_id, dx_order_id, order_status_en)
        db.resolve_errors_for_asset(
            asset_id,
            [
                "no_eligible_order_for_client_order_id",
                "multiple_eligible_orders_for_client_order_id",
            ],
        )

        logger.log_event(
            entity_type="asset",
            entity_id=asset_id,
            event_type="asset_dx_order_resolved",
            event_data={
                **summary,
                "dx_order_id": dx_order_id,
                "order_status_en": order_status_en,
            },
        )
        return

    db.update_asset_resolution_needs_manual_link(asset_id)

    if len(eligible) == 0:
        candidates_payload = _serialize_candidates(candidates)
        logger.log_error(
            error_type="no_eligible_order_for_client_order_id",
            message="No eligible orders found for client_order_id using order_status_en allowlist",
            severity="error",
            asset_id=asset_id,
            client_order_id=client_order_id,
            context={**summary, "candidates": candidates_payload},
        )
        logger.log_event(
            entity_type="asset",
            entity_id=asset_id,
            event_type="asset_dx_order_resolution_failed",
            event_data={**summary, "failure_type": "no_eligible", "candidates": candidates_payload},
        )
        return

    eligible_payload = _serialize_candidates(eligible)
    logger.log_error(
        error_type="multiple_eligible_orders_for_client_order_id",
        message="Multiple eligible orders found for client_order_id; strict mode forbids auto-pick",
        severity="error",
        asset_id=asset_id,
        client_order_id=client_order_id,
        context={**summary, "eligible_candidates": eligible_payload},
    )
    logger.log_event(
        entity_type="asset",
        entity_id=asset_id,
        event_type="asset_dx_order_resolution_failed",
        event_data={**summary, "failure_type": "multiple_eligible", "eligible_candidates": eligible_payload},
    )


def resolve_pending_links(
    db: Database,
    logger: EventLogger,
    config: AppConfig,
    limit: int,
) -> int:
    asset_ids = db.fetch_assets_needing_resolution(limit)
    for asset_id in asset_ids:
        try:
            resolve_asset_dx_order(db, logger, config, asset_id)
        except Exception as exc:
            logger.log_error(
                error_type="resolve_pending_failed",
                message=str(exc),
                severity="error",
                asset_id=asset_id,
                context={"asset_id": asset_id},
            )
    return len(asset_ids)


def validate_order_still_eligible(
    db: Database,
    logger: EventLogger,
    config: AppConfig,
    dx_order_id: str,
    asset_id: str | None = None,
    client_order_id: str | None = None,
) -> bool:
    status = db.fetch_order_status(dx_order_id)
    eligible = status is not None and status in set(config.eligible_order_status_en)

    if not eligible:
        logger.log_error(
            error_type="order_became_ineligible_before_export",
            message="Order status is no longer eligible before export",
            severity="error",
            asset_id=asset_id,
            client_order_id=client_order_id,
            dx_order_id=dx_order_id,
            context={
                "dx_order_id": dx_order_id,
                "order_status_en": status,
                "eligible_allowlist": list(config.eligible_order_status_en),
            },
        )
    return eligible
