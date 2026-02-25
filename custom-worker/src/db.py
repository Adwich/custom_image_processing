from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import psycopg
from psycopg.rows import dict_row


class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn

    @contextmanager
    def connection(self):
        conn = psycopg.connect(self._dsn, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_runtime_tables(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    create table if not exists public.custom_kv (
                        key text primary key,
                        value text,
                        updated_at timestamptz not null default now()
                    );
                    """
                )

    def get_kv(self, key: str) -> Optional[str]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("select value from public.custom_kv where key = %s", (key,))
                row = cur.fetchone()
                return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into public.custom_kv (key, value)
                    values (%s, %s)
                    on conflict (key)
                    do update set value = excluded.value, updated_at = now()
                    """,
                    (key, value),
                )

    def asset_exists_for_drive_file(self, drive_file_id: str) -> bool:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select 1 from public.custom_assets where drive_file_id = %s",
                    (drive_file_id,),
                )
                return cur.fetchone() is not None

    def insert_asset(self, asset: dict[str, Any]) -> bool:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into public.custom_assets (
                        id,
                        client_order_id,
                        drive_file_id,
                        drive_path,
                        original_storage_path,
                        cut_option,
                        scent,
                        quantity,
                        status,
                        error
                    )
                    values (
                        %(id)s,
                        %(client_order_id)s,
                        %(drive_file_id)s,
                        %(drive_path)s,
                        %(original_storage_path)s,
                        %(cut_option)s,
                        %(scent)s,
                        %(quantity)s,
                        %(status)s,
                        %(error)s
                    )
                    on conflict (drive_file_id) do nothing
                    returning id
                    """,
                    asset,
                )
                return cur.fetchone() is not None

    def fetch_asset(self, asset_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, client_order_id, dx_order_id, status, cut_option,
                           original_storage_path, created_at
                    from public.custom_assets
                    where id = %s
                    """,
                    (asset_id,),
                )
                return cur.fetchone()

    def fetch_orders_by_client_order_id(self, client_order_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select o.dx_order_id, o.order_status_en, o.created_at
                    from public.orders o
                    where o.client_order_id = %s
                    """,
                    (client_order_id,),
                )
                return cur.fetchall()

    def update_asset_resolution_success(
        self,
        asset_id: str,
        dx_order_id: str,
        order_status_en: str,
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update public.custom_assets
                    set
                        dx_order_id = %s,
                        order_status_at_resolution = %s,
                        status = case when status = 'needs_manual_link' then 'ingested' else status end,
                        updated_at = now()
                    where id = %s
                    """,
                    (dx_order_id, order_status_en, asset_id),
                )

    def update_asset_resolution_needs_manual_link(self, asset_id: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update public.custom_assets
                    set
                        dx_order_id = null,
                        order_status_at_resolution = null,
                        status = 'needs_manual_link',
                        updated_at = now()
                    where id = %s
                    """,
                    (asset_id,),
                )

    def resolve_errors_for_asset(self, asset_id: str, error_types: Iterable[str]) -> None:
        types = list(error_types)
        if not types:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update public.custom_errors
                    set status = 'resolved'
                    where asset_id = %s
                      and status = 'open'
                      and error_type = any(%s)
                    """,
                    (asset_id, types),
                )

    def fetch_assets_needing_resolution(self, limit: int) -> list[str]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id
                    from public.custom_assets
                    where dx_order_id is null
                      and status in ('ingested', 'needs_metadata', 'needs_manual_link')
                    order by created_at asc
                    limit %s
                    """,
                    (limit,),
                )
                return [str(row["id"]) for row in cur.fetchall()]

    def claim_assets_for_processing(self, limit: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id
                    from public.custom_assets
                    where status in ('ingested', 'needs_manual_link')
                      and cut_option is not null
                      and original_storage_path is not null
                    order by created_at asc
                    for update skip locked
                    limit %s
                    """,
                    (limit,),
                )
                ids = [str(row["id"]) for row in cur.fetchall()]
                if not ids:
                    return []
                cur.execute(
                    """
                    update public.custom_assets
                    set status = 'processing', updated_at = now()
                    where id = any(%s::uuid[])
                    returning id, client_order_id, cut_option, original_storage_path, status
                    """,
                    (ids,),
                )
                return cur.fetchall()

    def mark_asset_processed(
        self,
        asset_id: str,
        processed_storage_path: str,
        mask_storage_path: Optional[str],
        status: str,
        error: Optional[str],
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update public.custom_assets
                    set
                        processed_storage_path = %s,
                        mask_storage_path = %s,
                        status = %s,
                        error = %s,
                        updated_at = now()
                    where id = %s
                    """,
                    (processed_storage_path, mask_storage_path, status, error, asset_id),
                )

    def mark_asset_failed(self, asset_id: str, error: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update public.custom_assets
                    set status = 'failed', error = %s, updated_at = now()
                    where id = %s
                    """,
                    (error, asset_id),
                )

    def fetch_order_status(self, dx_order_id: str) -> Optional[str]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select order_status_en from public.orders where dx_order_id = %s",
                    (dx_order_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return str(row["order_status_en"])

    def insert_event(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        event_data: dict[str, Any],
        created_by: Optional[str] = None,
    ) -> None:
        payload = json.dumps(event_data)
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into public.custom_events (
                        entity_type,
                        entity_id,
                        event_type,
                        event_data,
                        created_by
                    )
                    values (%s, %s, %s, %s::jsonb, %s)
                    """,
                    (entity_type, entity_id, event_type, payload, created_by),
                )

    def insert_error(
        self,
        error_type: str,
        message: str,
        severity: str,
        status: str,
        asset_id: Optional[str],
        client_order_id: Optional[str],
        dx_order_id: Optional[str],
        context: Optional[dict[str, Any]],
    ) -> None:
        payload = json.dumps(context or {})
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into public.custom_errors (
                        error_type,
                        message,
                        severity,
                        status,
                        asset_id,
                        client_order_id,
                        dx_order_id,
                        context
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        error_type,
                        message,
                        severity,
                        status,
                        asset_id,
                        client_order_id,
                        dx_order_id,
                        payload,
                    ),
                )
