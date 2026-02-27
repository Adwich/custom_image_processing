# custom-worker

Fly worker service for customization pipeline Points 4/5/6:
- Drive ingest (UploadKit files)
- strict `dx_order_id` resolution using `orders.order_status_en`
- image processing (PhotoRoom bg removal + mode logic + framing + outer stroke)
- structured operational logs in `custom_errors` and `custom_events`
- JSON stdout/stderr logging for Fly machine logs
- optional Sentry error/check-in integration

## Status conventions

This worker uses:
- `needs_review` for all successfully processed assets (dashboard/manual review later promotes to `processed`)
- `failed` only for runtime exceptions
- `needs_manual_link` for strict dx resolution ambiguity (`eligible_count != 1`)

## Environment variables

Required:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_DB_URL`
- `GDRIVE_FOLDER_ID`
- `GDRIVE_SERVICE_ACCOUNT_JSON` or `GDRIVE_SERVICE_ACCOUNT_FILE`

Optional:
- `SUPABASE_STORAGE_BUCKET=customization`
- `GDRIVE_CURSOR_KV_KEY=custom_worker_gdrive_cursor`
- `POLL_INTERVAL_SECONDS=5`
- `MAX_INGEST_PER_RUN=50`
- `MAX_PROCESS_PER_RUN=25`
- `MAX_RESOLVE_PER_RUN=50`
- `MAX_EXPORT_PER_RUN=1`
- `EXPORT_SIGNED_URL_TTL_SECONDS=86400`
- `CONTROL_API_ENABLED=false`
- `CONTROL_API_HOST=0.0.0.0`
- `CONTROL_API_PORT=8787`
- `CONTROL_API_TOKEN=...` (required when `CONTROL_API_ENABLED=true`)
- `ELIGIBLE_ORDER_STATUS_EN=pending,wb_submit,wb_assign,wb_success,wb_failed,to_ship_in,to_ship_oos`
- `ORDER_STATUS_FIELD=order_status_en`
- `TIE_BREAK=created_at_desc`
- `IMAGE_OUTPUT_SIZE=1500`
- `STROKE_PX=15`
- `EDGE_DEFRINGE_STRENGTH=0.70` (reduces dark halo before white stroke)
- `EDGE_DEFRINGE_ALPHA_MAX=245`
- `HEAD_CUT_FEATHER_PX=28`
- `HEAD_SHOULDER_OFFSET_RATIO=0.015`
- `HEAD_FACE_FALLBACK_OFFSET_RATIO=0.22`
- `HEAD_REQUIRE_FACE_DETECTION=true`
- `HEAD_FACE_MIN_CONFIDENCE=0.40`
- `HEAD_FACE_COVERAGE_MIN_RATIO=0.65`
- `HEAD_TOP_CLIP_MARGIN_RATIO=0.01`
- `HEAD_TORSO_LEAKAGE_MAX_RATIO=0.42`
- `HEAD_REQUIRE_PART_PARSER=false` (`true` for strict production head-only mode)
- `HEAD_PART_PARSER_PROVIDER=none` (`api` when using hosted human parsing)
- `HEAD_PART_PARSER_API_URL=...`
- `HEAD_PART_PARSER_API_KEY=...`
- `HEAD_PART_PARSER_TIMEOUT_SECONDS=45`
- `HEAD_PART_INCLUDE_LABELS=hair;face;head;neck`
- `HEAD_PART_EXCLUDE_LABELS=upper_clothes;torso;upper_body;left_arm;right_arm;arms`
- `HEAD_USE_AILABTOOLS=true` (for HEAD mode, try AILabTools first)
- `AILABTOOLS_HEAD_API_KEY=...`
- `AILABTOOLS_HEAD_API_URL=https://www.ailabapi.com/api/cutout/portrait/avatar-extraction`
- `AILABTOOLS_TIMEOUT_SECONDS=60`
- `AILABTOOLS_RETURN_FORM=` (optional; leave empty for standard URL response)
- `SEGMENTATION_BACKEND=photoroom` (`rembg` and `prompted_sam` remain supported fallback backends)
- `SEGMENTATION_SKIP_REFINEMENT=false` (when `true`, skip `refine_cutout_alpha` and go directly from raw segmented image to mask/framing/stroke)
- `PHOTOROOM_API_KEY=...`
- `PHOTOROOM_API_URL=https://sdk.photoroom.com/v1/segment`
- `PHOTOROOM_TIMEOUT_SECONDS=60`
- `REMBG_MODEL_HUMAN=u2net_human_seg`
- `REMBG_MODEL_OBJECT=isnet-general-use`
- `REMBG_MODEL_FALLBACK=u2net`
- `REMBG_ALPHA_MATTING=true`
- `REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD=240`
- `REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD=10`
- `REMBG_ALPHA_MATTING_ERODE_SIZE=5`
- `REMBG_POST_PROCESS_MASK=true`
- `PROMPTED_DETECTOR_MODEL=google/owlvit-base-patch32`
- `PROMPTED_SAM_MODEL=facebook/sam-vit-base`
- `PROMPTED_DETECTION_THRESHOLD=0.12`
- `PROMPTED_HEAD_LABELS=person head only, no shoulders;human head;face`
- `PROMPTED_BODY_LABELS=person full body;person`
- `PROMPTED_CAR_LABELS=car;truck;vehicle;object`
- `SENTRY_DSN=...`
- `SENTRY_ENVIRONMENT=production` (defaults to `FLY_APP_NAME`/`ENV`/`NODE_ENV`)
- `SENTRY_RELEASE=...` (defaults to Fly/Git metadata when present)
- `SENTRY_TRACES_SAMPLE_RATE=0`
- `SENTRY_HEARTBEAT_INTERVAL_SECONDS=60`
- `SENTRY_MONITOR_SLUG=custom-worker`

## Local run

```bash
cd custom-worker
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env.local
# edit .env.local and provide real credentials/paths
python -m src.main
```

`src.config.load_config()` automatically loads `.env` and `.env.local` from the
`custom-worker/` directory, so you do not need to export variables manually.

Runtime logs are JSON lines with keys like `level`, `msg`, `ts`, and context
fields, matching the style used by your existing Fly workers.

Processing modes:
- `head`: AILabTools head extraction is attempted first (head + bg removal in one pass). If it fails, fallback to configured segmentation backend (for example PhotoRoom). Then MediaPipe face safety gate + optional human parsing part-mask merge (`hair/face/head/neck` minus torso/arms), single-component cleanup, 1500x1500 framing, white outer stroke, strict head review gate.
- `body`: PhotoRoom segmentation, single-component cleanup, 1500x1500 framing, white outer stroke, body gate.
- `car`: PhotoRoom segmentation, single-component cleanup, 1500x1500 framing, white outer stroke, object gate.
- `none`: no segmentation, frame only, no stroke.

Set `SEGMENTATION_SKIP_REFINEMENT=true` if you want a raw path from `*_segmented.png` straight to mask/framing/stroke.

Processing artifacts:
- `.../{asset_id}_segmented.png`: raw image returned by AILabTools/PhotoRoom before post-processing.
- `.../{asset_id}_mask.png`: mask after pipeline refinements.
- `.../{asset_id}_final.png`: framed output with white outer stroke.

Stroke behavior:
- The stroke is generated from a dilated outer mask and the masked subject is composited on top.
- Subject alpha is preserved while edge RGB is white-matted to reduce dark halo artifacts.

Run tests:

```bash
pytest
```

## Fly deploy

```bash
cd custom-worker
fly launch --no-deploy
fly secrets set \
  SUPABASE_URL=... \
  SUPABASE_SERVICE_ROLE_KEY=... \
  SUPABASE_DB_URL=... \
  GDRIVE_FOLDER_ID=... \
  GDRIVE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
fly deploy
```

`fly.toml` runs:

```text
python -m src.main
```

No public ports are exposed.

## Manual export jobs

The worker exposes a control API for manual export jobs (when `CONTROL_API_ENABLED=true`):
- `POST /exports` with body `{ "asset_ids": ["uuid"], "requested_by": "admin" }`
- `GET /exports/{id}`
- `GET /exports?limit=30`

The worker itself processes jobs from `public.custom_exports`:
- Claims `pending` jobs and marks them `running`
- Collects assets eligible for export (`status='processed'` and `processed_storage_path is not null`)
- Builds a ZIP and uploads to `customization/exports/{job_id}/...`
- Writes a signed URL and marks job `completed`
- Marks job `failed` on errors

Job payload contract (`request_payload` json):
- `asset_ids: string[]` (optional): if provided, export only these asset IDs (still requires status `processed`)
- if omitted, exports all currently eligible processed assets

## Eligible status maintenance

Update `ELIGIBLE_ORDER_STATUS_EN` with comma-separated `orders.order_status_en` values.
The strict resolver only sets `dx_order_id` when exactly one eligible candidate exists.
If `eligible_count == 0` or `eligible_count > 1`, the worker sets `needs_manual_link` and writes a `custom_errors` row.

## needs_review interpretation

`needs_review` means image output exists and is awaiting dashboard/manual review.
If quality checks fail, `custom_assets.error` is set with `quality_gate_failed: ...` and a warning is logged.
Quality checks include:
- mask area ratio out of range
- too many large connected components
- bbox too thin/empty
- heuristic mismatch (e.g. `head` too wide, `car` too tall)

Details are stored in:
- `custom_assets.error`
- `custom_errors` (`error_type=processing_needs_review`)
- `custom_events` (`event_type=asset_needs_review`)
