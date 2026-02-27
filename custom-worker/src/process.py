from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from .ailabtools import AILabToolsHeadClient
from .config import AppConfig
from .db import Database
from .head_mode import (
    apply_head_mask_to_cutout,
    detect_primary_face,
    evaluate_head_specific_quality,
)
from .human_parsing import HumanPartParserClient, compose_head_part_mask
from .images import (
    add_outer_white_stroke,
    alpha_from_rgba,
    frame_cutout,
    image_to_png_bytes,
    load_image_from_bytes,
    refine_cutout_alpha,
)
from .logging_events import EventLogger
from .photoroom import PhotoRoomClient
from .quality import QualityResult, evaluate_quality_gate
from .prompted_sam import PromptedSamSegmenter
from .storage import SupabaseStorageClient


@dataclass(frozen=True)
class ProcessingOutcome:
    status: str
    error: str | None
    is_auto_processed: bool
    quality: QualityResult | None
    mask_storage_path: str | None
    segmentation_storage_path: str | None
    segmentation_engine: str | None
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
        self._photoroom_client: PhotoRoomClient | None = None
        self._head_part_parser: HumanPartParserClient | None = None
        self._ailabtools_head_client: AILabToolsHeadClient | None = None

    def _get_session(self, model_name: str):
        if model_name in self._sessions:
            return self._sessions[model_name]
        from rembg import new_session

        session = new_session(model_name)
        self._sessions[model_name] = session
        return session

    def _get_photoroom_client(self) -> PhotoRoomClient:
        if self._photoroom_client is None:
            self._photoroom_client = PhotoRoomClient(
                api_key=self.config.photoroom_api_key,
                api_url=self.config.photoroom_api_url,
                timeout_seconds=self.config.photoroom_timeout_seconds,
            )
        return self._photoroom_client

    def _get_head_part_parser(self) -> HumanPartParserClient:
        if self._head_part_parser is None:
            self._head_part_parser = HumanPartParserClient(
                provider=self.config.head_part_parser_provider,
                api_url=self.config.head_part_parser_api_url,
                api_key=self.config.head_part_parser_api_key,
                timeout_seconds=self.config.head_part_parser_timeout_seconds,
            )
        return self._head_part_parser

    def _get_ailabtools_head_client(self) -> AILabToolsHeadClient:
        if self._ailabtools_head_client is None:
            self._ailabtools_head_client = AILabToolsHeadClient(
                api_key=self.config.ailabtools_head_api_key,
                api_url=self.config.ailabtools_head_api_url,
                timeout_seconds=self.config.ailabtools_timeout_seconds,
                return_form=self.config.ailabtools_return_form,
            )
        return self._ailabtools_head_client

    def _segment_head_preferred(
        self,
        original_bytes: bytes,
        image: Image.Image,
        mode_metrics: dict[str, Any],
    ) -> Image.Image:
        if self.config.head_use_ailabtools:
            head_client = self._get_ailabtools_head_client()
            if head_client.enabled:
                try:
                    cutout = head_client.extract_head(
                        original_bytes,
                        filename="asset.png",
                    )
                    mode_metrics["head_segmentation_engine"] = "ailabtools"
                    mode_metrics["head_ailabtools_used"] = True
                    return cutout
                except Exception as exc:
                    mode_metrics["head_ailabtools_used"] = False
                    mode_metrics["head_ailabtools_error"] = str(exc)
            else:
                mode_metrics["head_ailabtools_used"] = False
                mode_metrics["head_ailabtools_error"] = "not_configured"

        cutout = self._segment(image, "head")
        mode_metrics["head_segmentation_engine"] = self.config.segmentation_backend
        return cutout

    def _segment(self, image: Image.Image, cut_option: str) -> Image.Image:
        if self.config.segmentation_backend == "photoroom":
            return self._segment_photoroom(image)

        if self.config.segmentation_backend == "prompted_sam":
            try:
                return self._segment_prompted_sam(image, cut_option)
            except Exception:
                # Keep processing robust: fallback to rembg when prompted mode fails.
                pass

        return self._segment_rembg(image, cut_option)

    def _segment_photoroom(self, image: Image.Image) -> Image.Image:
        payload = image_to_png_bytes(image.convert("RGBA"))
        return self._get_photoroom_client().remove_background(payload, filename="asset.png")

    def _segment_rembg(self, image: Image.Image, cut_option: str) -> Image.Image:
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

    def _apply_mode_quality(
        self,
        alpha: Any,
        cut_option: str,
        mode_metrics: dict[str, Any],
    ) -> QualityResult:
        base = evaluate_quality_gate(alpha, cut_option)
        reasons = list(base.reasons)
        metrics = dict(base.metrics)
        metrics.update(mode_metrics)

        if cut_option == "head":
            forced_reasons = mode_metrics.get("head_forced_reasons", [])
            if isinstance(forced_reasons, list):
                for reason in forced_reasons:
                    if reason:
                        reasons.append(str(reason))

        deduped = tuple(dict.fromkeys(reasons))
        return QualityResult(passed=(len(deduped) == 0), reasons=deduped, metrics=metrics)

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
                    is_auto_processed=outcome.is_auto_processed,
                )

                if outcome.error:
                    self.logger.log_error(
                        error_type="processing_needs_review",
                        message=outcome.error,
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
                        "segmentation_storage_path": outcome.segmentation_storage_path,
                        "segmentation_engine": outcome.segmentation_engine,
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

        mode_metrics: dict[str, Any] = {
            "cut_option": cut_option,
            "segmentation_backend": self.config.segmentation_backend,
        }

        if cut_option == "none":
            cutout = original_image
            quality = QualityResult(
                True,
                tuple(),
                {
                    "cut_option": "none",
                    "segmentation_skipped": True,
                    "segmentation_backend": "none",
                },
            )
            mask_storage_path = None
            segmentation_storage_path = None
            segmentation_engine = None
        else:
            raw_engine_cutout: Image.Image
            if cut_option == "head":
                cutout = self._segment_head_preferred(
                    original_bytes=original_bytes,
                    image=original_image,
                    mode_metrics=mode_metrics,
                )
                raw_engine_cutout = cutout.copy()
                segmentation_engine = str(
                    mode_metrics.get("head_segmentation_engine", "head_unknown")
                )
                head_reasons: list[str] = []
                face = detect_primary_face(
                    original_image,
                    min_detection_confidence=self.config.head_face_min_confidence,
                )
                mode_metrics["face_detected"] = face is not None
                mode_metrics["face_score"] = face.score if face else None

                parser = self._get_head_part_parser()
                head_mask = None
                if parser.enabled:
                    try:
                        part_masks = parser.parse_part_masks(
                            image_bytes=image_to_png_bytes(original_image.convert("RGBA")),
                            image_size=original_image.size,
                        )
                        mode_metrics["head_part_mask_count"] = len(part_masks)
                        head_mask = compose_head_part_mask(
                            part_masks,
                            include_labels=self.config.head_part_include_labels,
                            exclude_labels=self.config.head_part_exclude_labels,
                        )
                        mode_metrics["head_part_mask_applied"] = head_mask is not None
                    except Exception as exc:
                        mode_metrics["head_part_mask_applied"] = False
                        mode_metrics["head_part_parser_error"] = str(exc)
                        if self.config.head_require_part_parser:
                            head_reasons.append("head_part_parser_error")
                else:
                    mode_metrics["head_part_mask_applied"] = False
                    if self.config.head_require_part_parser:
                        head_reasons.append("head_part_parser_not_configured")

                if face is None and self.config.head_require_face_detection:
                    head_reasons.append("head_face_not_detected")

                if head_mask is not None and not (face is None and self.config.head_require_face_detection):
                    cutout = apply_head_mask_to_cutout(cutout, head_mask)
                elif self.config.head_require_part_parser and face is not None:
                    head_reasons.append("head_part_mask_missing")

                if not self.config.segmentation_skip_refinement:
                    cutout = refine_cutout_alpha(
                        cutout,
                        keep_components=1,
                        min_component_px=140,
                        min_component_ratio=0.01,
                        smooth_sigma=0.85,
                    )

                alpha = alpha_from_rgba(cutout)
                head_gate_reasons, head_gate_metrics = evaluate_head_specific_quality(
                    alpha=alpha,
                    face=face,
                    require_face_detection=self.config.head_require_face_detection,
                    face_coverage_min_ratio=self.config.head_face_coverage_min_ratio,
                    top_clip_margin_ratio=self.config.head_top_clip_margin_ratio,
                    torso_leakage_max_ratio=self.config.head_torso_leakage_max_ratio,
                )
                head_reasons.extend(head_gate_reasons)
                mode_metrics.update(head_gate_metrics)
                mode_metrics["head_forced_reasons"] = list(dict.fromkeys(head_reasons))
            elif cut_option == "body":
                cutout = self._segment(original_image, cut_option)
                raw_engine_cutout = cutout.copy()
                segmentation_engine = self.config.segmentation_backend
                if not self.config.segmentation_skip_refinement:
                    cutout = refine_cutout_alpha(
                        cutout,
                        keep_components=1,
                        min_component_px=140,
                        min_component_ratio=0.01,
                        smooth_sigma=0.85,
                    )
            elif cut_option == "car":
                cutout = self._segment(original_image, cut_option)
                raw_engine_cutout = cutout.copy()
                segmentation_engine = self.config.segmentation_backend
                if not self.config.segmentation_skip_refinement:
                    cutout = refine_cutout_alpha(
                        cutout,
                        keep_components=1,
                        min_component_px=140,
                        min_component_ratio=0.01,
                        smooth_sigma=0.75,
                    )
            else:
                cutout = self._segment(original_image, cut_option)
                raw_engine_cutout = cutout.copy()
                segmentation_engine = self.config.segmentation_backend
                if not self.config.segmentation_skip_refinement:
                    cutout = refine_cutout_alpha(
                        cutout,
                        keep_components=1,
                        min_component_px=100,
                        min_component_ratio=0.01,
                        smooth_sigma=0.85,
                    )
            alpha = alpha_from_rgba(cutout)
            quality = self._apply_mode_quality(alpha, cut_option, mode_metrics)

            segmentation_storage_path = (
                f"{client_order_id}/processed/{asset_id}_segmented.png"
            )
            self.storage.upload_bytes(
                segmentation_storage_path,
                image_to_png_bytes(raw_engine_cutout.convert("RGBA")),
                "image/png",
            )

            mask_storage_path = f"{client_order_id}/processed/{asset_id}_mask.png"
            mask_img = Image.fromarray(alpha, mode="L")
            self.storage.upload_bytes(
                mask_storage_path,
                image_to_png_bytes(mask_img.convert("RGBA")),
                "image/png",
            )

        framed, _ = frame_cutout(cutout, cut_option, self.config.image_output_size)

        if cut_option != "none":
            final_img = add_outer_white_stroke(
                framed,
                self.config.stroke_px,
                cut_option=cut_option,
            )
        else:
            final_img = framed

        final_path = f"{client_order_id}/processed/{asset_id}_final.png"
        self.storage.upload_bytes(final_path, image_to_png_bytes(final_img), "image/png")

        error = None
        is_auto_processed = bool(quality.passed)
        if not is_auto_processed:
            reason = ",".join(quality.reasons)
            error = f"quality_gate_failed: {reason}"

        return ProcessingOutcome(
            status="needs_review",
            error=error,
            is_auto_processed=is_auto_processed,
            quality=quality,
            mask_storage_path=mask_storage_path,
            segmentation_storage_path=segmentation_storage_path,
            segmentation_engine=segmentation_engine,
            processed_storage_path=final_path,
        )
