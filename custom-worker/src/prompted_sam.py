from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class DetectionResult:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]


class PromptedSamSegmenter:
    def __init__(
        self,
        detector_model: str,
        sam_model: str,
        detection_threshold: float = 0.12,
    ):
        import torch
        from transformers import SamModel, SamProcessor, pipeline

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._detector = pipeline(
            task="zero-shot-object-detection",
            model=detector_model,
            device=0 if self._device == "cuda" else -1,
        )
        self._sam_model = SamModel.from_pretrained(sam_model).to(self._device)
        self._sam_processor = SamProcessor.from_pretrained(sam_model)
        self._detection_threshold = detection_threshold

    @staticmethod
    def _normalize_box(box: dict) -> tuple[float, float, float, float]:
        xmin = float(box.get("xmin", box.get("x0", 0.0)))
        ymin = float(box.get("ymin", box.get("y0", 0.0)))
        xmax = float(box.get("xmax", box.get("x1", xmin)))
        ymax = float(box.get("ymax", box.get("y1", ymin)))
        return (xmin, ymin, xmax, ymax)

    def _best_detection(
        self,
        image: Image.Image,
        labels: Sequence[str],
        mode: str,
    ) -> DetectionResult | None:
        candidates = [lbl.strip() for lbl in labels if lbl and lbl.strip()]
        if not candidates:
            return None

        detections = self._detector(image, candidate_labels=candidates)
        if not detections:
            # Broad fallback for harder crops/backgrounds.
            detections = self._detector(
                image,
                candidate_labels=["person", "face", "head", "car", "truck", "vehicle", "object"],
            )
            if not detections:
                return None

        parsed: list[DetectionResult] = []
        for det in detections:
            score = float(det.get("score", 0.0))
            if score < (self._detection_threshold * 0.5):
                continue
            parsed.append(
                DetectionResult(
                    label=str(det.get("label", "")),
                    score=score,
                    box_xyxy=self._normalize_box(det.get("box", {})),
                )
            )

        if not parsed:
            return None

        iw, ih = image.size
        best = max(parsed, key=lambda det: self._detection_rank(det, mode, iw, ih))
        if best.score < (self._detection_threshold * 0.75):
            return None
        return best

    def _detection_rank(
        self,
        det: DetectionResult,
        mode: str,
        image_w: int,
        image_h: int,
    ) -> float:
        x0, y0, x1, y1 = det.box_xyxy
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        area_ratio = (w * h) / max(1.0, float(image_w * image_h))
        aspect = w / h
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        dx = abs((cx / max(1.0, image_w)) - 0.5)
        dy = abs((cy / max(1.0, image_h)) - 0.5)
        center_penalty = (dx + dy) * 0.40

        label = det.label.lower()
        rank = det.score - center_penalty

        if mode == "head":
            if "head" in label or "face" in label:
                rank += 0.15
            if "person" in label:
                rank += 0.05
            if area_ratio < 0.012:
                rank -= 0.65
            if area_ratio > 0.55:
                rank -= 0.25
            if aspect < 0.35 or aspect > 1.8:
                rank -= 0.45
            # A head box should usually be in upper portion, not tiny strip.
            if (cy / max(1.0, image_h)) > 0.72:
                rank -= 0.20
        elif mode == "body":
            if "person" in label:
                rank += 0.10
            if area_ratio < 0.04:
                rank -= 0.35
            if aspect < 0.18 or aspect > 1.4:
                rank -= 0.25
        elif mode == "car":
            if any(token in label for token in ("car", "truck", "vehicle", "object")):
                rank += 0.08
            if area_ratio < 0.02:
                rank -= 0.25
            if aspect < 0.30 or aspect > 3.8:
                rank -= 0.20

        return rank

    def _mask_from_box(self, image: Image.Image, box_xyxy: tuple[float, float, float, float]) -> np.ndarray:
        box = [list(box_xyxy)]
        inputs = self._sam_processor(
            images=image,
            input_boxes=[box],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._sam_model(**inputs)

        masks = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        iou = outputs.iou_scores[0, 0].cpu()
        idx = int(self._torch.argmax(iou).item())
        best_mask = masks[0][0][idx].numpy() > 0.5
        return best_mask

    def _cleanup_mask(
        self,
        mask: np.ndarray,
        box_xyxy: tuple[float, float, float, float],
        mode: str,
    ) -> np.ndarray:
        cleaned = mask.astype(bool)
        cleaned = ndimage.binary_opening(cleaned, structure=np.ones((3, 3), dtype=bool))
        cleaned = ndimage.binary_closing(cleaned, structure=np.ones((5, 5), dtype=bool))
        cleaned = ndimage.binary_fill_holes(cleaned)

        labeled, count = ndimage.label(cleaned)
        if count <= 0:
            return cleaned

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        total = int(sizes.sum())
        if total <= 0:
            return cleaned

        min_component = max(80, int(total * 0.01))
        x0, y0, x1, y1 = box_xyxy
        box_cx = (x0 + x1) / 2.0
        box_cy = (y0 + y1) / 2.0
        h, w = cleaned.shape

        ranked_components: list[tuple[float, int]] = []
        for idx in range(1, len(sizes)):
            size = int(sizes[idx])
            if size < min_component:
                continue
            ys, xs = np.where(labeled == idx)
            if xs.size == 0:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            dist = (((cx - box_cx) / max(1.0, w)) ** 2) + (((cy - box_cy) / max(1.0, h)) ** 2)
            rank = size - (dist * total * 0.9)
            ranked_components.append((rank, idx))

        if not ranked_components:
            keep = {int(np.argmax(sizes))}
        else:
            ranked_components.sort(reverse=True)
            max_keep = 1 if mode == "head" else 2
            keep = {idx for _, idx in ranked_components[:max_keep]}

        cleaned = np.isin(labeled, list(keep))
        cleaned = ndimage.binary_fill_holes(cleaned)

        # Smooth contour and keep anti-aliased alpha to avoid jagged edges.
        sigma = 1.1 if mode == "head" else 0.9
        soft = ndimage.gaussian_filter(cleaned.astype(np.float32), sigma=sigma)
        soft = np.clip(soft, 0.0, 1.0)
        return soft

    def segment(self, image: Image.Image, labels: Sequence[str], mode: str = "body") -> Image.Image:
        rgba = image.convert("RGBA")
        rgb = rgba.convert("RGB")
        detection = self._best_detection(rgb, labels, mode=mode)
        if detection is None:
            raise RuntimeError("prompted_sam_no_detection")

        raw_mask = self._mask_from_box(rgb, detection.box_xyxy)
        mask = self._cleanup_mask(raw_mask, detection.box_xyxy, mode=mode)
        solid_ratio = float(np.mean(mask > 0.20))
        if mode == "head" and solid_ratio < 0.05:
            raise RuntimeError("prompted_sam_head_too_small")
        if mode == "body" and solid_ratio < 0.045:
            raise RuntimeError("prompted_sam_body_too_small")
        arr = np.asarray(rgba, dtype=np.uint8).copy()
        arr[:, :, 3] = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")
