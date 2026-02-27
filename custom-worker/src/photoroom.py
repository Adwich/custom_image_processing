from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image

from .images import load_image_from_bytes


class PhotoRoomError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhotoRoomClient:
    api_key: str
    api_url: str
    timeout_seconds: int = 60

    def remove_background(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ) -> Image.Image:
        if not self.api_key.strip():
            raise PhotoRoomError("PHOTOROOM_API_KEY is required for segmentation_backend=photoroom")

        headers = {
            "x-api-key": self.api_key,
            "Accept": "image/png,application/json",
        }
        files = {
            "image_file": (filename, image_bytes, "application/octet-stream"),
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            files=files,
            timeout=self.timeout_seconds,
        )

        if response.status_code >= 400:
            details = response.text[:500].strip()
            raise PhotoRoomError(
                f"PhotoRoom API error {response.status_code}: {details or 'empty error response'}"
            )

        image_payload = self._extract_image_bytes(response)
        return load_image_from_bytes(image_payload).convert("RGBA")

    def _extract_image_bytes(self, response: requests.Response) -> bytes:
        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("image/"):
            return response.content

        payload = response.json()
        if not isinstance(payload, dict):
            raise PhotoRoomError("PhotoRoom response did not include an image payload")

        for key in ("result_b64", "image", "image_base64", "output", "data", "result"):
            value = payload.get(key)
            decoded = self._decode_candidate(value)
            if decoded:
                return decoded

        raise PhotoRoomError(f"Unable to decode PhotoRoom response payload: {json.dumps(payload)[:300]}")

    def _decode_candidate(self, value: Any) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, dict):
            for nested_key in ("base64", "b64", "image", "data"):
                nested = self._decode_candidate(value.get(nested_key))
                if nested:
                    return nested
            return None
        if not isinstance(value, str):
            return None

        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("data:image"):
            _, _, data_part = raw.partition(",")
            raw = data_part

        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            return None
