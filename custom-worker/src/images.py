from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
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


def _resize_rgba_premultiplied(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0

    alpha = arr[:, :, 3:4]
    premul_rgb = arr[:, :, :3] * alpha
    premul = np.concatenate((premul_rgb, alpha), axis=2)

    premul_img = Image.fromarray(np.clip(premul * 255.0, 0, 255).astype(np.uint8), mode="RGBA")
    premul_resized = premul_img.resize(size, resample=Image.Resampling.LANCZOS)
    out = np.asarray(premul_resized, dtype=np.float32) / 255.0

    out_alpha = out[:, :, 3:4]
    safe_alpha = np.where(out_alpha > 1e-6, out_alpha, 1.0)
    rgb = np.where(out_alpha > 1e-6, out[:, :, :3] / safe_alpha, 0.0)

    out_rgba = np.concatenate((np.clip(rgb, 0.0, 1.0), np.clip(out_alpha, 0.0, 1.0)), axis=2)
    return Image.fromarray(np.clip(out_rgba * 255.0, 0, 255).astype(np.uint8), mode="RGBA")


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
    resized = _resize_rgba_premultiplied(cropped, (new_w, new_h))

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


def defringe_edges_to_white(
    image: Image.Image,
    strength: float = 0.70,
    alpha_max: int = 245,
) -> Image.Image:
    rgba = image.convert("RGBA")
    if strength <= 0:
        return rgba

    arr = np.asarray(rgba, dtype=np.float32).copy()
    alpha = arr[:, :, 3] / 255.0
    alpha_limit = max(1, min(alpha_max, 254)) / 255.0
    edge = np.logical_and(alpha > 0.0, alpha < alpha_limit)
    if not np.any(edge):
        return rgba

    max_rgb = np.max(arr[:, :, :3], axis=2) / 255.0
    darkness = 1.0 - max_rgb
    weight = ((1.0 - alpha) * float(strength) * (0.55 + darkness)).clip(0.0, 1.0)
    weight = np.where(edge, weight, 0.0)
    arr[:, :, :3] = (arr[:, :, :3] * (1.0 - weight[..., None])) + (
        255.0 * weight[..., None]
    )

    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def refine_cutout_alpha(
    image: Image.Image,
    keep_components: int = 2,
    min_component_px: int = 80,
    min_component_ratio: float = 0.01,
    smooth_sigma: float = 0.9,
) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8).copy()
    alpha = arr[:, :, 3]
    mask = alpha > 20

    if int(mask.sum()) == 0:
        return rgba

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)

    labeled, count = ndimage.label(mask)
    if count > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        total = int(sizes.sum())
        min_size = max(min_component_px, int(total * max(0.0, min_component_ratio)))
        ranked = [
            (int(sizes[idx]), idx)
            for idx in range(1, len(sizes))
            if int(sizes[idx]) >= min_size
        ]
        if ranked:
            ranked.sort(reverse=True)
            keep_ids = {idx for _, idx in ranked[: max(1, keep_components)]}
            mask = np.isin(labeled, list(keep_ids))
            mask = ndimage.binary_fill_holes(mask)

    soft = ndimage.gaussian_filter(mask.astype(np.float32), sigma=max(0.0, smooth_sigma))
    new_alpha = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
    arr[:, :, 3] = new_alpha
    return Image.fromarray(arr, mode="RGBA")


def apply_head_cut_line(
    image: Image.Image,
    cut_line_y: int,
    feather_px: int = 28,
) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8).copy()
    alpha = arr[:, :, 3].astype(np.float32)
    h = alpha.shape[0]
    if h <= 0:
        return rgba

    cut_y = int(np.clip(cut_line_y, 0, h))
    if cut_y <= 0:
        arr[:, :, 3] = 0
        return Image.fromarray(arr, mode="RGBA")

    if feather_px > 0:
        start = max(0, cut_y - int(feather_px))
        end = cut_y
        if end > start:
            ys = np.arange(start, end, dtype=np.float32)
            span = max(1.0, float(end - start))
            fade = (float(end) - ys) / span
            alpha[start:end, :] *= np.clip(fade[:, None], 0.0, 1.0)

    alpha[cut_y:, :] = 0.0
    arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _circular_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array([[True]], dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= (radius * radius)


def build_white_stroke_backdrop(image: Image.Image, stroke_px: int) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    alpha = arr[:, :, 3]
    core_mask = alpha > 20

    # Smooth jagged subject edges before stroke dilation so the white contour
    # is less noisy while still tracking the same silhouette.
    smooth_radius = max(2, int(round(max(1, stroke_px) * 0.55)))
    smooth_kernel = _circular_kernel(smooth_radius)
    core_mask = ndimage.binary_closing(core_mask, structure=smooth_kernel)
    core_mask = ndimage.binary_opening(core_mask, structure=smooth_kernel)
    core_mask = ndimage.binary_fill_holes(core_mask)
    # Additional contour smoothing pass before dilation.
    pre_sigma = max(0.8, float(stroke_px) * 0.38)
    core_mask = ndimage.gaussian_filter(core_mask.astype(np.float32), sigma=pre_sigma) >= 0.40

    dilated = binary_dilation(core_mask, structure=_circular_kernel(max(0, stroke_px)))
    edge_sigma = max(1.4, float(stroke_px) * 0.52)
    dilated = ndimage.gaussian_filter(dilated.astype(np.float32), sigma=edge_sigma) >= 0.40
    backdrop = np.zeros_like(arr, dtype=np.uint8)
    backdrop[dilated] = np.array([255, 255, 255, 255], dtype=np.uint8)
    return Image.fromarray(backdrop, mode="RGBA")


def add_outer_white_stroke(image: Image.Image, stroke_px: int) -> Image.Image:
    if stroke_px <= 0:
        return image.copy()

    framed = image.convert("RGBA")
    stroke_bg = build_white_stroke_backdrop(framed, stroke_px)
    # Ensure subject never extends outside stroke backdrop shape.
    bg_alpha = np.asarray(stroke_bg, dtype=np.uint8)[:, :, 3] > 0
    framed_arr = np.asarray(framed, dtype=np.uint8).copy()
    framed_arr[~bg_alpha, 3] = 0
    framed_clipped = Image.fromarray(framed_arr, mode="RGBA")
    stroke_bg.alpha_composite(framed_clipped)
    return stroke_bg
