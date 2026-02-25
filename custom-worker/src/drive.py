from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


FOLDER_MIME = "application/vnd.google-apps.folder"


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    drive_path: str


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class GoogleDriveClient:
    def __init__(self, service_account_json: str, root_folder_id: str):
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        info = json.loads(service_account_json)
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._root_folder_id = root_folder_id

    def _list_children(self, folder_id: str) -> list[dict[str, Any]]:
        query = f"'{folder_id}' in parents and trashed = false"
        page_token: Optional[str] = None
        rows: list[dict[str, Any]] = []
        while True:
            response = (
                self._service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=page_token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    pageSize=1000,
                )
                .execute()
            )
            rows.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return rows

    def list_files_since(self, since_iso: Optional[str], limit: int) -> list[DriveFile]:
        since = _parse_ts(since_iso) if since_iso else None

        files: list[DriveFile] = []
        stack: list[tuple[str, str]] = [(self._root_folder_id, "")]

        while stack and len(files) < (limit * 10):
            folder_id, prefix = stack.pop()
            for row in self._list_children(folder_id):
                name = str(row.get("name", ""))
                path = f"{prefix}/{name}".lstrip("/")
                mime_type = str(row.get("mimeType", ""))

                if mime_type == FOLDER_MIME:
                    stack.append((str(row["id"]), path))
                    continue

                modified_time = str(row.get("modifiedTime", ""))
                if since and modified_time:
                    if _parse_ts(modified_time) <= since:
                        continue

                files.append(
                    DriveFile(
                        file_id=str(row["id"]),
                        name=name,
                        mime_type=mime_type,
                        modified_time=modified_time,
                        drive_path=path,
                    )
                )

        files.sort(key=lambda item: item.modified_time)
        return files[:limit]

    def download_file(self, file_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload

        request = self._service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()
