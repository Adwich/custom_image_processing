from __future__ import annotations

from typing import Any, Optional

from .db import Database


class EventLogger:
    def __init__(self, db: Database):
        self.db = db

    def log_event(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        event_data: dict[str, Any],
        created_by: Optional[str] = None,
    ) -> None:
        self.db.insert_event(entity_type, entity_id, event_type, event_data, created_by)

    def log_error(
        self,
        error_type: str,
        message: str,
        severity: str,
        status: str = "open",
        asset_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        dx_order_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self.db.insert_error(
            error_type=error_type,
            message=message,
            severity=severity,
            status=status,
            asset_id=asset_id,
            client_order_id=client_order_id,
            dx_order_id=dx_order_id,
            context=context,
        )
