from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .images import apply_head_cut_line, refine_cutout_alpha


@dataclass(frozen=True)
class HeadAnchors:
    face_box_xyxy: tuple[int, int, int, int] | None
    face_score: float | None
    shoulder_y: int | None
    cut_line_y: int | None
    cut_line_source: str


@dataclass(frozen=True)
class FaceDetection:
    box_xyxy: tuple[int, int, int, int]
    score: float


def detect_primary_face(
    image: Image.Image,
    min_detection_confidence: float = 0.40,
) -> FaceDetection | None:
    try:
        import mediapipe as mp
    except Exception:
        return None

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return None

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_detection_confidence,
    ) as detector:
        result = detector.process(rgb)
        detections = list(result.detections or [])
        if not detections:
            return None

        best = max(detections, key=lambda item: float((item.score or [0.0])[0]))
        score = float((best.score or [0.0])[0])
        box = best.location_data.relative_bounding_box
        x0 = int(round(max(0.0, box.xmin) * w))
        y0 = int(round(max(0.0, box.ymin) * h))
        x1 = int(round(min(1.0, box.xmin + box.width) * w))
        y1 = int(round(min(1.0, box.ymin + box.height) * h))
        if x1 <= x0 or y1 <= y0:
            return None
        return FaceDetection(box_xyxy=(x0, y0, x1, y1), score=score)


def apply_head_mask_to_cutout(cutout: Image.Image, head_mask: np.ndarray) -> Image.Image:
    rgba = cutout.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    if head_mask.shape != (h, w):
        mask_img = Image.fromarray((head_mask.astype(np.uint8) * 255), mode="L")
        mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
        head_mask = np.asarray(mask_img, dtype=np.uint8) > 20

    alpha = arr[:, :, 3]
    alpha = np.where(head_mask, alpha, 0).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def evaluate_head_specific_quality(
    alpha: np.ndarray,
    face: FaceDetection | None,
    require_face_detection: bool,
    face_coverage_min_ratio: float,
    top_clip_margin_ratio: float,
    torso_leakage_max_ratio: float,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    mask = alpha > 20
    total = int(mask.sum())
    if total <= 0:
        reasons.append("empty_mask")
        metrics["head_alpha_pixels"] = 0
        return reasons, metrics

    ys, xs = np.where(mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    metrics["head_bbox_y0"] = y0
    metrics["head_bbox_y1"] = y1
    metrics["head_alpha_pixels"] = total

    top_margin_px = max(1, int(round(alpha.shape[0] * max(0.0, top_clip_margin_ratio))))
    top_clipped = y0 <= top_margin_px
    metrics["head_top_clipped"] = top_clipped
    if top_clipped:
        reasons.append("head_top_clipped")

    if face is None:
        if require_face_detection:
            reasons.append("head_face_not_detected")
        metrics["head_face_coverage_ratio"] = 0.0
        return list(dict.fromkeys(reasons)), metrics

    fx0, fy0, fx1, fy1 = face.box_xyxy
    fx0 = int(np.clip(fx0, 0, alpha.shape[1]))
    fy0 = int(np.clip(fy0, 0, alpha.shape[0]))
    fx1 = int(np.clip(fx1, 0, alpha.shape[1]))
    fy1 = int(np.clip(fy1, 0, alpha.shape[0]))
    face_area = max(1, (fx1 - fx0) * (fy1 - fy0))
    face_mask = mask[fy0:fy1, fx0:fx1] if fy1 > fy0 and fx1 > fx0 else np.zeros((1, 1), dtype=bool)
    face_coverage = float(face_mask.sum()) / float(face_area)
    metrics["head_face_coverage_ratio"] = face_coverage
    if face_coverage < face_coverage_min_ratio:
        reasons.append("head_face_coverage_low")

    face_h = max(1, fy1 - fy0)
    neck_y = min(alpha.shape[0] - 1, fy1 + int(round(face_h * 0.45)))
    below_ratio = float(mask[neck_y:, :].sum()) / float(total)
    metrics["head_alpha_below_neck_ratio"] = below_ratio
    if below_ratio > torso_leakage_max_ratio:
        reasons.append("head_torso_leakage")

    return list(dict.fromkeys(reasons)), metrics


def _estimate_silhouette_cut_line(alpha: np.ndarray) -> tuple[int | None, dict[str, Any]]:
    mask = alpha > 20
    if int(mask.sum()) == 0:
        return None, {"silhouette_fallback_used": False, "silhouette_reason": "empty_mask"}

    row_counts = mask.sum(axis=1).astype(np.float32)
    nonzero_rows = np.where(row_counts > 0)[0]
    if nonzero_rows.size < 12:
        return None, {
            "silhouette_fallback_used": False,
            "silhouette_reason": "mask_too_small",
        }

    y_top = int(nonzero_rows.min())
    y_bottom = int(nonzero_rows.max())
    h = y_bottom - y_top + 1
    widths = row_counts[y_top : y_bottom + 1]
    if h < 30 or widths.size < 10:
        return None, {
            "silhouette_fallback_used": False,
            "silhouette_reason": "subject_too_short",
        }

    smooth = ndimage.gaussian_filter1d(widths, sigma=max(1.0, h * 0.012))
    top_band_end = max(6, int(h * 0.30))
    top_band = smooth[:top_band_end]
    top_nonzero = top_band[top_band > 0]
    if top_nonzero.size == 0:
        return None, {
            "silhouette_fallback_used": False,
            "silhouette_reason": "no_top_pixels",
        }

    head_width_ref = float(np.percentile(top_nonzero, 35))
    if head_width_ref <= 0:
        return None, {
            "silhouette_fallback_used": False,
            "silhouette_reason": "invalid_head_width",
        }

    start = int(h * 0.20)
    end = int(h * 0.78)
    if end <= start + 2:
        return None, {
            "silhouette_fallback_used": False,
            "silhouette_reason": "invalid_search_range",
        }

    expansion_ratio = 1.18
    rise_threshold = max(3.0, head_width_ref * 0.02)
    rise_window = max(4, int(h * 0.04))
    candidates: list[int] = []
    for i in range(start, end):
        if smooth[i] < (head_width_ref * expansion_ratio):
            continue
        prior = max(start, i - rise_window)
        if smooth[i] >= (smooth[prior] + rise_threshold):
            candidates.append(i)

    if not candidates:
        for i in range(start, end):
            if smooth[i] >= (head_width_ref * expansion_ratio):
                candidates.append(i)

    if not candidates:
        return None, {
            "silhouette_fallback_used": False,
            "silhouette_reason": "no_shoulder_expansion",
            "silhouette_head_width_ref": head_width_ref,
        }

    # Avoid very early cuts: choose a later shoulder candidate to preserve full head shape.
    shoulder_pick = min(len(candidates) - 1, max(0, int(round(len(candidates) * 0.35))))
    shoulder_row = int(candidates[shoulder_pick])
    cut_line_local = shoulder_row - int(round(h * 0.03))
    cut_line_local = int(np.clip(cut_line_local, int(h * 0.30), int(h * 0.74)))
    cut_line_y = y_top + cut_line_local

    metrics = {
        "silhouette_fallback_used": True,
        "silhouette_reason": "ok",
        "silhouette_head_width_ref": head_width_ref,
        "silhouette_width_at_shoulder": float(smooth[shoulder_row]),
        "silhouette_shoulder_row_local": shoulder_row,
    }
    return cut_line_y, metrics


def detect_head_anchors(
    image: Image.Image,
    shoulder_offset_ratio: float,
    face_fallback_offset_ratio: float,
) -> HeadAnchors:
    try:
        import mediapipe as mp
    except Exception:
        return HeadAnchors(
            face_box_xyxy=None,
            face_score=None,
            shoulder_y=None,
            cut_line_y=None,
            cut_line_source="none",
        )

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return HeadAnchors(
            face_box_xyxy=None,
            face_score=None,
            shoulder_y=None,
            cut_line_y=None,
            cut_line_source="none",
        )

    mp_face = mp.solutions.face_detection
    mp_pose = mp.solutions.pose

    face_box: tuple[int, int, int, int] | None = None
    face_score: float | None = None

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.45) as detector:
        face_result = detector.process(rgb)
        detections = list(face_result.detections or [])
        if detections:
            best = max(detections, key=lambda item: float((item.score or [0.0])[0]))
            score = float((best.score or [0.0])[0])
            box = best.location_data.relative_bounding_box
            x0 = int(round(max(0.0, box.xmin) * w))
            y0 = int(round(max(0.0, box.ymin) * h))
            x1 = int(round(min(1.0, box.xmin + box.width) * w))
            y1 = int(round(min(1.0, box.ymin + box.height) * h))
            if x1 > x0 and y1 > y0:
                face_box = (x0, y0, x1, y1)
                face_score = score

    shoulder_y: int | None = None
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.45,
    ) as pose:
        pose_result = pose.process(rgb)
        landmarks = getattr(pose_result, "pose_landmarks", None)
        if landmarks and landmarks.landmark:
            lm = landmarks.landmark
            left = lm[11] if len(lm) > 11 else None
            right = lm[12] if len(lm) > 12 else None
            ys: list[float] = []
            if left and float(getattr(left, "visibility", 0.0)) >= 0.35:
                ys.append(float(left.y))
            if right and float(getattr(right, "visibility", 0.0)) >= 0.35:
                ys.append(float(right.y))
            if ys:
                shoulder_y = int(round(np.mean(ys) * h))

    cut_line_y: int | None = None
    cut_line_source = "none"

    if shoulder_y is not None:
        offset = int(round(h * shoulder_offset_ratio))
        cut_line_y = shoulder_y - offset
        cut_line_source = "shoulders"
    elif face_box is not None:
        _, _, _, fy1 = face_box
        face_h = max(1, face_box[3] - face_box[1])
        cut_line_y = fy1 + int(round(face_h * face_fallback_offset_ratio))
        cut_line_source = "face_fallback"

    if cut_line_y is not None:
        cut_line_y = int(np.clip(cut_line_y, 0, h - 1))
        if face_box is not None:
            _, _, _, fy1 = face_box
            cut_line_y = max(cut_line_y, fy1 + max(4, int(round((fy1 - face_box[1]) * 0.05))))
            cut_line_y = int(np.clip(cut_line_y, 0, h - 1))

    return HeadAnchors(
        face_box_xyxy=face_box,
        face_score=face_score,
        shoulder_y=shoulder_y,
        cut_line_y=cut_line_y,
        cut_line_source=cut_line_source,
    )


def enforce_head_only(
    cutout: Image.Image,
    anchors: HeadAnchors,
    feather_px: int,
) -> tuple[Image.Image, dict[str, Any]]:
    result = cutout.convert("RGBA")
    active_cut_line = anchors.cut_line_y
    active_cut_source = anchors.cut_line_source
    fallback_metrics: dict[str, Any] = {}

    if active_cut_line is None:
        alpha = np.asarray(result, dtype=np.uint8)[:, :, 3]
        fallback_cut, fallback_metrics = _estimate_silhouette_cut_line(alpha)
        if fallback_cut is not None:
            active_cut_line = int(fallback_cut)
            active_cut_source = "silhouette_fallback"

    if active_cut_line is not None:
        result = apply_head_cut_line(result, active_cut_line, feather_px=feather_px)

    result = refine_cutout_alpha(
        result,
        keep_components=1,
        min_component_px=120,
        min_component_ratio=0.02,
        smooth_sigma=1.0,
    )

    metrics: dict[str, Any] = {
        "face_detected": bool(anchors.face_box_xyxy),
        "face_score": anchors.face_score,
        "shoulder_y": anchors.shoulder_y,
        "cut_line_y": active_cut_line,
        "cut_line_source": active_cut_source,
        "head_cut_applied": active_cut_line is not None,
    }
    metrics.update(fallback_metrics)
    return result, metrics
