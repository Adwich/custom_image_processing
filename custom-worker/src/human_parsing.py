from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image


class HumanParsingError(RuntimeError):
    pass


def _normalize_label(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def _decode_image_mask_bytes(mask_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
    with Image.open(BytesIO(mask_bytes)) as img:
        mask_img = img.convert("L")
        if mask_img.size != target_size:
            mask_img = mask_img.resize(target_size, resample=Image.Resampling.NEAREST)
        arr = np.asarray(mask_img, dtype=np.uint8)
    return arr > 20


def _decode_base64_mask(value: str, target_size: tuple[int, int]) -> np.ndarray:
    raw = value.strip()
    if raw.startswith("data:image"):
        _, _, raw = raw.partition(",")
    payload = base64.b64decode(raw, validate=True)
    return _decode_image_mask_bytes(payload, target_size)


def _coerce_mask(value: Any, target_size: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return _decode_image_mask_bytes(value, target_size)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return _decode_base64_mask(s, target_size)
        except Exception:
            return None
    if isinstance(value, list):
        arr = np.asarray(value)
        if arr.ndim == 2:
            arr = arr.astype(np.uint8)
            if arr.shape[::-1] != target_size:
                arr = np.asarray(
                    Image.fromarray(arr, mode="L").resize(
                        target_size,
                        resample=Image.Resampling.NEAREST,
                    ),
                    dtype=np.uint8,
                )
            return arr > 0
    if isinstance(value, dict):
        for nested_key in ("mask", "mask_b64", "base64", "data", "image", "png_b64"):
            nested = _coerce_mask(value.get(nested_key), target_size)
            if nested is not None:
                return nested
    return None


def compose_head_part_mask(
    part_masks: dict[str, np.ndarray],
    include_labels: tuple[str, ...],
    exclude_labels: tuple[str, ...],
) -> np.ndarray | None:
    if not part_masks:
        return None

    norm_parts = {_normalize_label(k): v for k, v in part_masks.items()}
    include = [_normalize_label(v) for v in include_labels]
    exclude = [_normalize_label(v) for v in exclude_labels]

    include_mask: np.ndarray | None = None
    for key in include:
        part = norm_parts.get(key)
        if part is None:
            continue
        include_mask = part if include_mask is None else np.logical_or(include_mask, part)

    if include_mask is None:
        return None

    exclude_mask: np.ndarray | None = None
    for key in exclude:
        part = norm_parts.get(key)
        if part is None:
            continue
        exclude_mask = part if exclude_mask is None else np.logical_or(exclude_mask, part)

    if exclude_mask is not None:
        include_mask = np.logical_and(include_mask, np.logical_not(exclude_mask))

    return include_mask


@dataclass(frozen=True)
class HumanPartParserClient:
    provider: str
    api_url: str
    api_key: str
    timeout_seconds: int = 45

    @property
    def enabled(self) -> bool:
        return self.provider not in ("", "none") and bool(self.api_url.strip())

    def parse_part_masks(self, image_bytes: bytes, image_size: tuple[int, int]) -> dict[str, np.ndarray]:
        if not self.enabled:
            return {}

        headers = {"Accept": "application/json,image/png"}
        if self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["x-api-key"] = self.api_key

        files = {
            "image_file": ("image.png", image_bytes, "application/octet-stream"),
        }
        data = {"provider": self.provider}

        response = requests.post(
            self.api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=self.timeout_seconds,
        )

        if response.status_code >= 400:
            snippet = response.text[:500].strip()
            raise HumanParsingError(
                f"Human parsing API error {response.status_code}: {snippet or 'empty response'}"
            )

        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("image/"):
            mask = _decode_image_mask_bytes(response.content, image_size)
            return {"head": mask}

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise HumanParsingError("Human parsing response was not JSON/image") from exc

        if not isinstance(payload, dict):
            raise HumanParsingError("Human parsing JSON payload must be an object")

        parts: dict[str, np.ndarray] = {}

        direct = _coerce_mask(
            payload.get("head_mask")
            or payload.get("head_mask_b64")
            or payload.get("mask")
            or payload.get("mask_b64"),
            image_size,
        )
        if direct is not None:
            parts["head"] = direct

        payload_parts = payload.get("parts")
        if isinstance(payload_parts, dict):
            for key, value in payload_parts.items():
                parsed = _coerce_mask(value, image_size)
                if parsed is not None:
                    parts[str(key)] = parsed

        return parts
