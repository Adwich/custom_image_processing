from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image

from .images import load_image_from_bytes


class AILabToolsError(RuntimeError):
    pass


@dataclass(frozen=True)
class AILabToolsHeadClient:
    api_key: str
    api_url: str
    timeout_seconds: int = 60
    return_form: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.api_url.strip())

    def extract_head(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ) -> Image.Image:
        if not self.enabled:
            raise AILabToolsError(
                "AILabTools head extraction is not configured (missing key or URL)"
            )

        headers = {
            "ailabapi-api-key": self.api_key,
            "Accept": "application/json,image/png",
        }
        files = {
            "image": (filename, image_bytes, "application/octet-stream"),
        }
        data: dict[str, str] = {}
        if self.return_form.strip():
            data["return_form"] = self.return_form.strip()

        response = requests.post(
            self.api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            snippet = response.text[:500].strip()
            raise AILabToolsError(
                f"AILabTools API error {response.status_code}: {snippet or 'empty response'}"
            )

        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("image/"):
            return load_image_from_bytes(response.content).convert("RGBA")

        payload = response.json()
        if not isinstance(payload, dict):
            raise AILabToolsError("Unexpected AILabTools response payload")

        error_code = int(payload.get("error_code", 0))
        if error_code != 0:
            error_msg = str(payload.get("error_msg", "unknown error"))
            raise AILabToolsError(f"AILabTools returned error_code={error_code}: {error_msg}")

        data_obj = payload.get("data")
        if not isinstance(data_obj, dict):
            raise AILabToolsError("AILabTools response missing data object")

        image_url = self._extract_image_url(data_obj)
        if not image_url:
            raise AILabToolsError("AILabTools response missing image_url")

        image_response = requests.get(image_url, timeout=self.timeout_seconds)
        if image_response.status_code >= 400:
            raise AILabToolsError(
                f"Failed to download AILabTools image URL ({image_response.status_code})"
            )
        return load_image_from_bytes(image_response.content).convert("RGBA")

    def _extract_image_url(self, data_obj: dict[str, Any]) -> str:
        direct = data_obj.get("image_url")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        elements = data_obj.get("elements")
        if isinstance(elements, list):
            for item in elements:
                if isinstance(item, dict):
                    url = item.get("image_url")
                    if isinstance(url, str) and url.strip():
                        return url.strip()

        return ""
