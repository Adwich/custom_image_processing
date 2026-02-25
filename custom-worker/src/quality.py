from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def evaluate_quality_gate(alpha: np.ndarray, cut_option: str) -> QualityResult:
    if alpha.ndim != 2:
        raise ValueError("alpha mask must be 2D")

    mask = alpha > 20
    total = int(mask.size)
    mask_area = int(mask.sum())

    mask_area_ratio = (mask_area / total) if total else 0.0

    reasons: list[str] = []
    if mask_area_ratio < 0.10 or mask_area_ratio > 0.90:
        reasons.append("mask_area_ratio_out_of_range")

    metrics: dict[str, Any] = {
        "mask_area_ratio": mask_area_ratio,
        "mask_area_pixels": mask_area,
        "total_pixels": total,
        "cut_option": cut_option,
        "threshold_alpha": 20,
    }

    if mask_area == 0:
        reasons.append("empty_mask")
        metrics.update(
            {
                "component_count": 0,
                "large_component_count": 0,
                "bbox_width": 0,
                "bbox_height": 0,
                "bbox_aspect": 0.0,
            }
        )
        return QualityResult(False, tuple(dict.fromkeys(reasons)), metrics)

    labeled, count = ndimage.label(mask)
    component_sizes = np.bincount(labeled.ravel())[1:]
    large_threshold = max(1, int(mask_area * 0.05))
    large_component_count = int(np.sum(component_sizes > large_threshold))

    metrics["component_count"] = int(count)
    metrics["large_component_count"] = large_component_count

    if large_component_count > 2:
        reasons.append("too_many_large_components")

    box = _bbox(mask)
    if box is None:
        reasons.append("empty_bbox")
        bbox_w = 0
        bbox_h = 0
        bbox_aspect = 0.0
    else:
        x0, y0, x1, y1 = box
        bbox_w = x1 - x0
        bbox_h = y1 - y0
        bbox_aspect = float(bbox_w / bbox_h) if bbox_h > 0 else 0.0

        if bbox_w < max(5, int(mask.shape[1] * 0.03)) or bbox_h < max(
            5, int(mask.shape[0] * 0.03)
        ):
            reasons.append("bbox_too_thin")

    metrics["bbox_width"] = bbox_w
    metrics["bbox_height"] = bbox_h
    metrics["bbox_aspect"] = bbox_aspect

    if cut_option == "head" and bbox_aspect > 1.6:
        reasons.append("head_bbox_too_wide")
    if cut_option == "car" and bbox_aspect < 0.8:
        reasons.append("car_bbox_too_tall")

    return QualityResult(passed=(len(reasons) == 0), reasons=tuple(dict.fromkeys(reasons)), metrics=metrics)
