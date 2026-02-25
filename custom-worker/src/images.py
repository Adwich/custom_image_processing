from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation


FRAMING_PRESETS: dict[str, dict[str, float | str]] = {
    "head": {
        "target_axis": "height",
        "target_ratio": 0.83,
        "up_shift_ratio": 0.07,
        "margin_x": 0.20,
        "margin_y": 0.25,
    },
    "body": {
        "target_axis": "height",
        "target_ratio": 0.78,
        "up_shift_ratio": 0.02,
        "margin_x": 0.15,
        "margin_y": 0.20,
    },
    "car": {
        "target_axis": "width",
        "target_ratio": 0.88,
        "up_shift_ratio": 0.0,
        "margin_x": 0.12,
        "margin_y": 0.12,
    },
    "none": {
        "target_axis": "height",
        "target_ratio": 0.90,
        "up_shift_ratio": 0.0,
        "margin_x": 0.10,
        "margin_y": 0.10,
    },
}


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as img:
        fixed = ImageOps.exif_transpose(img)
        return fixed.convert("RGBA")


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def alpha_from_rgba(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return arr[:, :, 3]


def _bbox_from_alpha(alpha: np.ndarray, threshold: int = 20) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > threshold)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def frame_cutout(
    cutout: Image.Image,
    cut_option: str,
    output_size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    preset = FRAMING_PRESETS.get(cut_option, FRAMING_PRESETS["body"])

    rgba = cutout.convert("RGBA")
    alpha = alpha_from_rgba(rgba)
    bbox = _bbox_from_alpha(alpha)
    if bbox is None:
        raise ValueError("Cannot frame image with empty alpha mask")

    x0, y0, x1, y1 = bbox
    bbox_w = x1 - x0
    bbox_h = y1 - y0

    margin_x = int(round(bbox_w * float(preset["margin_x"])))
    margin_y = int(round(bbox_h * float(preset["margin_y"])))

    cx0 = max(0, x0 - margin_x)
    cy0 = max(0, y0 - margin_y)
    cx1 = min(rgba.width, x1 + margin_x)
    cy1 = min(rgba.height, y1 + margin_y)

    cropped = rgba.crop((cx0, cy0, cx1, cy1))

    subject_box_crop = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
    sbw = subject_box_crop[2] - subject_box_crop[0]
    sbh = subject_box_crop[3] - subject_box_crop[1]

    axis = str(preset["target_axis"])
    target = float(preset["target_ratio"]) * float(output_size)
    if axis == "width":
        scale = target / max(1.0, float(sbw))
    else:
        scale = target / max(1.0, float(sbh))

    new_w = max(1, int(round(cropped.width * scale)))
    new_h = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    sbx0 = subject_box_crop[0] * scale
    sby0 = subject_box_crop[1] * scale
    sbx1 = subject_box_crop[2] * scale
    sby1 = subject_box_crop[3] * scale

    subject_center_x = (sbx0 + sbx1) / 2.0
    subject_center_y = (sby0 + sby1) / 2.0
    subject_height = sby1 - sby0

    target_center_x = output_size / 2.0
    target_center_y = (output_size / 2.0) - (float(preset["up_shift_ratio"]) * subject_height)

    paste_x = int(round(target_center_x - subject_center_x))
    paste_y = int(round(target_center_y - subject_center_y))

    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y), resized)

    info = {
        "crop_box": (cx0, cy0, cx1, cy1),
        "bbox": bbox,
        "scale": scale,
        "paste": (paste_x, paste_y),
    }
    return canvas, info


def _circular_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array([[True]], dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= (radius * radius)


def add_outer_white_stroke(image: Image.Image, stroke_px: int) -> Image.Image:
    if stroke_px <= 0:
        return image.copy()

    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    orig_mask = arr[:, :, 3] > 20

    dilated = binary_dilation(orig_mask, structure=_circular_kernel(stroke_px))
    stroke_mask = np.logical_and(dilated, np.logical_not(orig_mask))

    stroke_layer = np.zeros_like(arr, dtype=np.uint8)
    stroke_layer[stroke_mask] = np.array([255, 255, 255, 255], dtype=np.uint8)

    base = Image.fromarray(stroke_layer, mode="RGBA")
    base.alpha_composite(rgba)
    return base
