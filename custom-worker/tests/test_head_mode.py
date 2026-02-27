import numpy as np
from PIL import Image

from src.head_mode import HeadAnchors, enforce_head_only
from src.images import alpha_from_rgba


def test_enforce_head_only_cuts_below_line() -> None:
    arr = np.zeros((120, 120, 4), dtype=np.uint8)
    arr[20:100, 30:90, :3] = 200
    arr[20:100, 30:90, 3] = 255
    image = Image.fromarray(arr, mode="RGBA")

    anchors = HeadAnchors(
        face_box_xyxy=(35, 25, 85, 70),
        face_score=0.95,
        shoulder_y=78,
        cut_line_y=72,
        cut_line_source="shoulders",
    )
    out, metrics = enforce_head_only(image, anchors, feather_px=12)
    alpha = alpha_from_rgba(out)

    assert int(alpha[90:, :].sum()) == 0
    assert int(alpha[:60, :].sum()) > 0
    assert metrics["head_cut_applied"] is True


def test_enforce_head_only_uses_silhouette_fallback_when_no_anchor() -> None:
    arr = np.zeros((200, 200, 4), dtype=np.uint8)
    arr[18:96, 68:132, :3] = 210
    arr[18:96, 68:132, 3] = 255
    arr[96:165, 38:162, :3] = 210
    arr[96:165, 38:162, 3] = 255
    image = Image.fromarray(arr, mode="RGBA")

    anchors = HeadAnchors(
        face_box_xyxy=None,
        face_score=None,
        shoulder_y=None,
        cut_line_y=None,
        cut_line_source="none",
    )
    out, metrics = enforce_head_only(image, anchors, feather_px=16)
    alpha = alpha_from_rgba(out)

    assert metrics["head_cut_applied"] is True
    assert metrics["cut_line_source"] == "silhouette_fallback"
    assert metrics["silhouette_fallback_used"] is True
    assert int(alpha[130:, :].sum()) == 0
