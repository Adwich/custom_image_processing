from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .config import AppConfig
from .db import Database
from .images import (
    add_outer_white_stroke,
    alpha_from_rgba,
    defringe_edges_to_white,
    frame_cutout,
    image_to_png_bytes,
    load_image_from_bytes,
    refine_cutout_alpha,
)
from .logging_events import EventLogger
from .quality import QualityResult, evaluate_quality_gate
from .prompted_sam import PromptedSamSegmenter
from .storage import SupabaseStorageClient


@dataclass(frozen=True)
class ProcessingOutcome:
    status: str
    error: str | None
    quality: QualityResult | None
    mask_storage_path: str | None
    processed_storage_path: str


class AssetProcessor:
    def __init__(
        self,
        config: AppConfig,
        db: Database,
        storage: SupabaseStorageClient,
        logger: EventLogger,
    ):
        self.config = config
        self.db = db
        self.storage = storage
        self.logger = logger
        self._sessions: dict[str, Any] = {}
        self._prompted_segmenter: PromptedSamSegmenter | None = None

    def _get_session(self, model_name: str):
        if model_name in self._sessions:
            return self._sessions[model_name]
        from rembg import new_session

        session = new_session(model_name)
        self._sessions[model_name] = session
        return session

    def _segment(self, image: Image.Image, cut_option: str) -> Image.Image:
        if self.config.segmentation_backend == "prompted_sam":
            try:
                return self._segment_prompted_sam(image, cut_option)
            except Exception:
                # Keep processing robust: fallback to rembg when prompted mode fails.
                pass

        if cut_option in ("head", "body"):
            model = self.config.rembg_model_human
        elif cut_option == "car":
            model = self.config.rembg_model_object
        else:
            model = self.config.rembg_model_fallback

        from rembg import remove

        payload = image_to_png_bytes(image.convert("RGBA"))
        cutout_bytes = remove(
            payload,
            session=self._get_session(model),
            alpha_matting=self.config.rembg_alpha_matting,
            alpha_matting_foreground_threshold=self.config.rembg_alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=self.config.rembg_alpha_matting_background_threshold,
            alpha_matting_erode_size=self.config.rembg_alpha_matting_erode_size,
            post_process_mask=self.config.rembg_post_process_mask,
        )
        cutout = load_image_from_bytes(cutout_bytes)
        return cutout.convert("RGBA")

    def _segment_prompted_sam(self, image: Image.Image, cut_option: str) -> Image.Image:
        if self._prompted_segmenter is None:
            self._prompted_segmenter = PromptedSamSegmenter(
                detector_model=self.config.prompted_detector_model,
                sam_model=self.config.prompted_sam_model,
                detection_threshold=self.config.prompted_detection_threshold,
            )

        if cut_option == "head":
            labels = self.config.prompted_head_labels
        elif cut_option == "body":
            labels = self.config.prompted_body_labels
        elif cut_option == "car":
            labels = self.config.prompted_car_labels
        else:
            labels = self.config.prompted_body_labels

        return self._prompted_segmenter.segment(
            image,
            labels,
            mode=cut_option,
        ).convert("RGBA")

    def process_images(self) -> int:
        assets = self.db.claim_assets_for_processing(self.config.max_process_per_run)
        if not assets:
            return 0

        for asset in assets:
            asset_id = str(asset["id"])
            self.logger.log_event(
                entity_type="asset",
                entity_id=asset_id,
                event_type="asset_processing_started",
                event_data={
                    "asset_id": asset_id,
                    "cut_option": asset.get("cut_option"),
                },
            )
            try:
                outcome = self._process_one(asset)
                self.db.mark_asset_processed(
                    asset_id=asset_id,
                    processed_storage_path=outcome.processed_storage_path,
                    mask_storage_path=outcome.mask_storage_path,
                    status=outcome.status,
                    error=outcome.error,
                )

                if outcome.status == "needs_review":
                    self.logger.log_error(
                        error_type="processing_needs_review",
                        message=outcome.error or "quality gate failed",
                        severity="warning",
                        asset_id=asset_id,
                        client_order_id=str(asset["client_order_id"]),
                        context=outcome.quality.metrics if outcome.quality else {},
                    )
                    self.logger.log_event(
                        entity_type="asset",
                        entity_id=asset_id,
                        event_type="asset_needs_review",
                        event_data={
                            "status": outcome.status,
                            "error": outcome.error,
                            "quality": outcome.quality.metrics if outcome.quality else {},
                            "processed_storage_path": outcome.processed_storage_path,
                            "mask_storage_path": outcome.mask_storage_path,
                        },
                    )
                else:
                    self.logger.log_event(
                        entity_type="asset",
                        entity_id=asset_id,
                        event_type="asset_processed",
                        event_data={
                            "status": outcome.status,
                            "quality": outcome.quality.metrics if outcome.quality else {},
                            "processed_storage_path": outcome.processed_storage_path,
                            "mask_storage_path": outcome.mask_storage_path,
                        },
                    )
            except Exception as exc:
                self.db.mark_asset_failed(asset_id=asset_id, error=str(exc))
                self.logger.log_error(
                    error_type="processing_failed",
                    message=str(exc),
                    severity="error",
                    asset_id=asset_id,
                    client_order_id=str(asset["client_order_id"]),
                    context={"asset_id": asset_id},
                )
                self.logger.log_event(
                    entity_type="asset",
                    entity_id=asset_id,
                    event_type="asset_processing_failed",
                    event_data={"error": str(exc)},
                )

        return len(assets)

    def _process_one(self, asset: dict[str, Any]) -> ProcessingOutcome:
        asset_id = str(asset["id"])
        client_order_id = str(asset["client_order_id"])
        cut_option = str(asset["cut_option"])
        original_path = str(asset["original_storage_path"])

        original_bytes = self.storage.download_bytes(original_path)
        original_image = load_image_from_bytes(original_bytes)

        if cut_option == "none":
            cutout = original_image
            quality = QualityResult(True, tuple(), {"cut_option": "none", "segmentation_skipped": True})
            mask_storage_path = None
        else:
            cutout = self._segment(original_image, cut_option)
            if cut_option == "head":
                cutout = refine_cutout_alpha(
                    cutout,
                    keep_components=1,
                    min_component_px=120,
                    min_component_ratio=0.02,
                    smooth_sigma=1.05,
                )
            else:
                cutout = refine_cutout_alpha(
                    cutout,
                    keep_components=2,
                    min_component_px=100,
                    min_component_ratio=0.01,
                    smooth_sigma=0.9,
                )
            alpha = alpha_from_rgba(cutout)
            quality = evaluate_quality_gate(alpha, cut_option)

            mask_storage_path = f"{client_order_id}/processed/{asset_id}_mask.png"
            mask_img = Image.fromarray(alpha, mode="L")
            self.storage.upload_bytes(
                mask_storage_path,
                image_to_png_bytes(mask_img.convert("RGBA")),
                "image/png",
            )

        framed, _ = frame_cutout(cutout, cut_option, self.config.image_output_size)

        if cut_option != "none":
            framed = defringe_edges_to_white(
                framed,
                strength=self.config.edge_defringe_strength,
                alpha_max=self.config.edge_defringe_alpha_max,
            )
            final_img = add_outer_white_stroke(framed, self.config.stroke_px)
        else:
            final_img = framed

        final_path = f"{client_order_id}/processed/{asset_id}_final.png"
        self.storage.upload_bytes(final_path, image_to_png_bytes(final_img), "image/png")

        if quality.passed:
            return ProcessingOutcome(
                status="processed",
                error=None,
                quality=quality,
                mask_storage_path=mask_storage_path,
                processed_storage_path=final_path,
            )

        reason = ",".join(quality.reasons)
        return ProcessingOutcome(
            status="needs_review",
            error=f"quality_gate_failed: {reason}",
            quality=quality,
            mask_storage_path=mask_storage_path,
            processed_storage_path=final_path,
        )
