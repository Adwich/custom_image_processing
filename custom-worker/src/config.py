from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ELIGIBLE_STATUSES = (
    "pending,wb_submit,wb_assign,wb_success,wb_failed,to_ship_in,to_ship_oos"
)


@dataclass(frozen=True)
class AppConfig:
    supabase_url: str
    supabase_service_role_key: str
    supabase_db_url: str
    supabase_storage_bucket: str

    gdrive_service_account_json: str
    gdrive_folder_id: str
    gdrive_cursor_kv_key: str

    poll_interval_seconds: int
    max_ingest_per_run: int
    max_process_per_run: int
    max_resolve_per_run: int

    eligible_order_status_en: tuple[str, ...]
    order_status_field: str
    tie_break: str

    image_output_size: int
    stroke_px: int
    edge_defringe_strength: float
    edge_defringe_alpha_max: int
    head_cut_feather_px: int
    head_shoulder_offset_ratio: float
    head_face_fallback_offset_ratio: float
    head_require_face_detection: bool
    head_face_min_confidence: float
    head_face_coverage_min_ratio: float
    head_top_clip_margin_ratio: float
    head_torso_leakage_max_ratio: float
    head_require_part_parser: bool
    head_part_parser_provider: str
    head_part_parser_api_url: str
    head_part_parser_api_key: str
    head_part_parser_timeout_seconds: int
    head_part_include_labels: tuple[str, ...]
    head_part_exclude_labels: tuple[str, ...]
    head_use_ailabtools: bool
    ailabtools_head_api_key: str
    ailabtools_head_api_url: str
    ailabtools_timeout_seconds: int
    ailabtools_return_form: str

    rembg_model_human: str
    rembg_model_object: str
    rembg_model_fallback: str
    rembg_alpha_matting: bool
    rembg_alpha_matting_foreground_threshold: int
    rembg_alpha_matting_background_threshold: int
    rembg_alpha_matting_erode_size: int
    rembg_post_process_mask: bool
    photoroom_api_key: str
    photoroom_api_url: str
    photoroom_timeout_seconds: int
    segmentation_backend: str
    prompted_detector_model: str
    prompted_sam_model: str
    prompted_detection_threshold: float
    prompted_head_labels: tuple[str, ...]
    prompted_body_labels: tuple[str, ...]
    prompted_car_labels: tuple[str, ...]


class ConfigError(RuntimeError):
    pass


_ENV_LOADED = False


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]
    return key, value


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if not parsed:
            continue
        key, value = parsed
        # Keep shell-provided env values as highest priority.
        os.environ.setdefault(key, value)


def _load_local_env_files_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    project_root = Path(__file__).resolve().parent.parent
    # Base defaults first, then local overrides.
    _load_env_file(project_root / ".env")
    _load_env_file(project_root / ".env.local")
    _ENV_LOADED = True


def _get_required(name: str, strict: bool) -> str:
    value = os.getenv(name, "").strip()
    if strict and not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _get_optional(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() if isinstance(value, str) else str(value)


def _parse_csv(value: str) -> tuple[str, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_labels(value: str) -> tuple[str, ...]:
    parts = [p.strip() for p in value.split(";") if p.strip()]
    return tuple(parts)


def _load_service_account_json(strict: bool) -> str:
    inline = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    file_path = os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE", "").strip()
    if inline:
        return inline
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as exc:
            if strict:
                raise ConfigError(f"Unable to read GDRIVE_SERVICE_ACCOUNT_FILE: {exc}") from exc
    if strict:
        raise ConfigError(
            "Provide GDRIVE_SERVICE_ACCOUNT_JSON or GDRIVE_SERVICE_ACCOUNT_FILE"
        )
    return ""


def load_config(strict: bool = True) -> AppConfig:
    _load_local_env_files_once()

    segmentation_backend = _get_optional("SEGMENTATION_BACKEND", "photoroom").lower()
    photoroom_api_key = _get_optional("PHOTOROOM_API_KEY", "")
    if strict and segmentation_backend == "photoroom" and not photoroom_api_key:
        raise ConfigError(
            "Missing required environment variable: PHOTOROOM_API_KEY (SEGMENTATION_BACKEND=photoroom)"
        )

    eligible_csv = _get_optional("ELIGIBLE_ORDER_STATUS_EN", DEFAULT_ELIGIBLE_STATUSES)
    eligible = _parse_csv(eligible_csv)
    if strict and not eligible:
        raise ConfigError("ELIGIBLE_ORDER_STATUS_EN resolved to an empty set")

    return AppConfig(
        supabase_url=_get_required("SUPABASE_URL", strict),
        supabase_service_role_key=_get_required("SUPABASE_SERVICE_ROLE_KEY", strict),
        supabase_db_url=_get_required("SUPABASE_DB_URL", strict),
        supabase_storage_bucket=_get_optional("SUPABASE_STORAGE_BUCKET", "customization"),
        gdrive_service_account_json=_load_service_account_json(strict),
        gdrive_folder_id=_get_required("GDRIVE_FOLDER_ID", strict),
        gdrive_cursor_kv_key=_get_optional("GDRIVE_CURSOR_KV_KEY", "custom_worker_gdrive_cursor"),
        poll_interval_seconds=int(_get_optional("POLL_INTERVAL_SECONDS", "5")),
        max_ingest_per_run=int(_get_optional("MAX_INGEST_PER_RUN", "50")),
        max_process_per_run=int(_get_optional("MAX_PROCESS_PER_RUN", "25")),
        max_resolve_per_run=int(_get_optional("MAX_RESOLVE_PER_RUN", "50")),
        eligible_order_status_en=eligible,
        order_status_field=_get_optional("ORDER_STATUS_FIELD", "order_status_en"),
        tie_break=_get_optional("TIE_BREAK", "created_at_desc"),
        image_output_size=int(_get_optional("IMAGE_OUTPUT_SIZE", "1500")),
        stroke_px=int(_get_optional("STROKE_PX", "15")),
        edge_defringe_strength=float(_get_optional("EDGE_DEFRINGE_STRENGTH", "0.70")),
        edge_defringe_alpha_max=int(_get_optional("EDGE_DEFRINGE_ALPHA_MAX", "245")),
        head_cut_feather_px=int(_get_optional("HEAD_CUT_FEATHER_PX", "28")),
        head_shoulder_offset_ratio=float(
            _get_optional("HEAD_SHOULDER_OFFSET_RATIO", "0.015")
        ),
        head_face_fallback_offset_ratio=float(
            _get_optional("HEAD_FACE_FALLBACK_OFFSET_RATIO", "0.22")
        ),
        head_require_face_detection=_parse_bool(
            _get_optional("HEAD_REQUIRE_FACE_DETECTION", "true")
        ),
        head_face_min_confidence=float(_get_optional("HEAD_FACE_MIN_CONFIDENCE", "0.40")),
        head_face_coverage_min_ratio=float(
            _get_optional("HEAD_FACE_COVERAGE_MIN_RATIO", "0.65")
        ),
        head_top_clip_margin_ratio=float(
            _get_optional("HEAD_TOP_CLIP_MARGIN_RATIO", "0.01")
        ),
        head_torso_leakage_max_ratio=float(
            _get_optional("HEAD_TORSO_LEAKAGE_MAX_RATIO", "0.42")
        ),
        head_require_part_parser=_parse_bool(
            _get_optional("HEAD_REQUIRE_PART_PARSER", "false")
        ),
        head_part_parser_provider=_get_optional(
            "HEAD_PART_PARSER_PROVIDER", "none"
        ).lower(),
        head_part_parser_api_url=_get_optional("HEAD_PART_PARSER_API_URL", ""),
        head_part_parser_api_key=_get_optional("HEAD_PART_PARSER_API_KEY", ""),
        head_part_parser_timeout_seconds=int(
            _get_optional("HEAD_PART_PARSER_TIMEOUT_SECONDS", "45")
        ),
        head_part_include_labels=_parse_labels(
            _get_optional("HEAD_PART_INCLUDE_LABELS", "hair;face;head;neck")
        ),
        head_part_exclude_labels=_parse_labels(
            _get_optional(
                "HEAD_PART_EXCLUDE_LABELS",
                "upper_clothes;torso;upper_body;left_arm;right_arm;arms",
            )
        ),
        head_use_ailabtools=_parse_bool(_get_optional("HEAD_USE_AILABTOOLS", "true")),
        ailabtools_head_api_key=_get_optional("AILABTOOLS_HEAD_API_KEY", ""),
        ailabtools_head_api_url=_get_optional(
            "AILABTOOLS_HEAD_API_URL",
            "https://www.ailabapi.com/api/cutout/portrait/avatar-extraction",
        ),
        ailabtools_timeout_seconds=int(_get_optional("AILABTOOLS_TIMEOUT_SECONDS", "60")),
        ailabtools_return_form=_get_optional("AILABTOOLS_RETURN_FORM", ""),
        rembg_model_human=_get_optional("REMBG_MODEL_HUMAN", "u2net_human_seg"),
        rembg_model_object=_get_optional("REMBG_MODEL_OBJECT", "isnet-general-use"),
        rembg_model_fallback=_get_optional("REMBG_MODEL_FALLBACK", "u2net"),
        rembg_alpha_matting=_parse_bool(_get_optional("REMBG_ALPHA_MATTING", "true")),
        rembg_alpha_matting_foreground_threshold=int(
            _get_optional("REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD", "240")
        ),
        rembg_alpha_matting_background_threshold=int(
            _get_optional("REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD", "10")
        ),
        rembg_alpha_matting_erode_size=int(
            _get_optional("REMBG_ALPHA_MATTING_ERODE_SIZE", "5")
        ),
        rembg_post_process_mask=_parse_bool(
            _get_optional("REMBG_POST_PROCESS_MASK", "true")
        ),
        photoroom_api_key=photoroom_api_key,
        photoroom_api_url=_get_optional(
            "PHOTOROOM_API_URL", "https://sdk.photoroom.com/v1/segment"
        ),
        photoroom_timeout_seconds=int(_get_optional("PHOTOROOM_TIMEOUT_SECONDS", "60")),
        segmentation_backend=segmentation_backend,
        prompted_detector_model=_get_optional(
            "PROMPTED_DETECTOR_MODEL", "google/owlvit-base-patch32"
        ),
        prompted_sam_model=_get_optional("PROMPTED_SAM_MODEL", "facebook/sam-vit-base"),
        prompted_detection_threshold=float(
            _get_optional("PROMPTED_DETECTION_THRESHOLD", "0.12")
        ),
        prompted_head_labels=_parse_labels(
            _get_optional(
                "PROMPTED_HEAD_LABELS",
                "person head only, no shoulders;human head;face",
            )
        ),
        prompted_body_labels=_parse_labels(
            _get_optional(
                "PROMPTED_BODY_LABELS",
                "person full body;person",
            )
        ),
        prompted_car_labels=_parse_labels(
            _get_optional(
                "PROMPTED_CAR_LABELS",
                "car;truck;vehicle;object",
            )
        ),
    )
