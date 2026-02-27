from __future__ import annotations

import json
from urllib.parse import quote

import requests


class SupabaseStorageClient:
    def __init__(self, supabase_url: str, service_role_key: str, bucket: str):
        self._base = supabase_url.rstrip("/")
        self._key = service_role_key
        self._bucket = bucket

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "apikey": self._key,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def upload_bytes(self, path: str, data: bytes, content_type: str) -> None:
        encoded_path = quote(path, safe="/")
        url = f"{self._base}/storage/v1/object/{self._bucket}/{encoded_path}"
        headers = self._headers(content_type)
        headers["x-upsert"] = "true"
        response = requests.post(url, headers=headers, data=data, timeout=60)
        if response.status_code >= 300:
            raise RuntimeError(
                f"Storage upload failed [{response.status_code}] for {path}: {response.text}"
            )

    def download_bytes(self, path: str) -> bytes:
        encoded_path = quote(path, safe="/")
        url = f"{self._base}/storage/v1/object/{self._bucket}/{encoded_path}"
        response = requests.get(url, headers=self._headers(), timeout=60)
        if response.status_code >= 300:
            raise RuntimeError(
                f"Storage download failed [{response.status_code}] for {path}: {response.text}"
            )
        return response.content

    def create_signed_url(self, path: str, expires_in_seconds: int) -> str:
        encoded_path = quote(path, safe="/")
        url = f"{self._base}/storage/v1/object/sign/{self._bucket}/{encoded_path}"
        response = requests.post(
            url,
            headers=self._headers("application/json"),
            data=json.dumps({"expiresIn": int(expires_in_seconds)}),
            timeout=60,
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"Storage sign failed [{response.status_code}] for {path}: {response.text}"
            )
        payload = response.json()
        signed = payload.get("signedURL") or payload.get("signedUrl")
        if not signed:
            raise RuntimeError(f"Storage sign response missing signedURL for {path}")
        if isinstance(signed, str) and signed.startswith("http"):
            return signed
        signed_path = str(signed)
        if not signed_path.startswith("/"):
            signed_path = f"/{signed_path}"
        return f"{self._base}/storage/v1{signed_path}"
