from pathlib import Path

import numpy as np
from PIL import Image

from src.images import (
    add_outer_white_stroke,
    alpha_from_rgba,
    defringe_edges_to_white,
    frame_cutout,
)
from src.quality import evaluate_quality_gate


FIXTURES = Path(__file__).parent / "images"


def _load(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGBA")


def test_processing_outputs_shape() -> None:
    img = _load(FIXTURES / "head" / "ok" / "sample.png")
    framed, _ = frame_cutout(img, "head", 1500)
    assert framed.size == (1500, 1500)


def test_stroke_exists() -> None:
    img = _load(FIXTURES / "body" / "ok" / "sample.png")
    framed, _ = frame_cutout(img, "body", 1500)
    stroked = add_outer_white_stroke(framed, 15)

    base_alpha = alpha_from_rgba(framed) > 20
    arr = np.asarray(stroked)
    white = (
        (arr[:, :, 0] == 255)
        & (arr[:, :, 1] == 255)
        & (arr[:, :, 2] == 255)
        & (arr[:, :, 3] > 200)
    )
    stroke_only = white & (~base_alpha)

    assert int(stroke_only.sum()) > 0


def test_negative_sets_trigger_needs_review() -> None:
    img = _load(FIXTURES / "negative" / "car_marked_head" / "sample.png")
    alpha = alpha_from_rgba(img)
    gate = evaluate_quality_gate(alpha, "head")

    assert gate.passed is False
    assert ("head_bbox_too_wide" in gate.reasons) or ("mask_area_ratio_out_of_range" in gate.reasons)


def test_defringe_lightens_dark_edge_pixels() -> None:
    arr = np.zeros((60, 60, 4), dtype=np.uint8)
    arr[15:45, 15:45, :3] = 120
    arr[15:45, 15:45, 3] = 255
    arr[12:48, 12:48, :3] = 0
    arr[12:48, 12:48, 3] = 90
    arr[15:45, 15:45, :3] = 120
    arr[15:45, 15:45, 3] = 255

    img = Image.fromarray(arr, mode="RGBA")
    out = defringe_edges_to_white(img, strength=1.0, alpha_max=245)
    out_arr = np.asarray(out, dtype=np.uint8)

    edge_mask = arr[:, :, 3] == 90
    before_edge_brightness = float(arr[:, :, :3][edge_mask].mean())
    after_edge_brightness = float(out_arr[:, :, :3][edge_mask].mean())

    assert after_edge_brightness > before_edge_brightness
