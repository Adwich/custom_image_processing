# custom-worker

Fly worker service for customization pipeline Points 4/5/6:
- Drive ingest (UploadKit files)
- strict `dx_order_id` resolution using `orders.order_status_en`
- image processing (rembg + framing + outer stroke)
- structured operational logs in `custom_errors` and `custom_events`
- JSON stdout/stderr logging for Fly machine logs
- optional Sentry error/check-in integration

## Status conventions

This worker uses:
- `processed` for successful quality gate
- `needs_review` when quality gate fails (not `failed`)
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
- `ELIGIBLE_ORDER_STATUS_EN=pending,wb_submit,wb_assign,wb_success,wb_failed,to_ship_in,to_ship_oos`
- `ORDER_STATUS_FIELD=order_status_en`
- `TIE_BREAK=created_at_desc`
- `IMAGE_OUTPUT_SIZE=1500`
- `STROKE_PX=15`
- `EDGE_DEFRINGE_STRENGTH=0.70` (reduces dark halo before white stroke)
- `EDGE_DEFRINGE_ALPHA_MAX=245`
- `REMBG_MODEL_HUMAN=u2net_human_seg`
- `REMBG_MODEL_OBJECT=isnet-general-use`
- `REMBG_MODEL_FALLBACK=u2net`
- `REMBG_ALPHA_MATTING=true`
- `REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD=240`
- `REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD=10`
- `REMBG_ALPHA_MATTING_ERODE_SIZE=5`
- `REMBG_POST_PROCESS_MASK=true`
- `SEGMENTATION_BACKEND=rembg` (`prompted_sam` for text-prompt segmentation)
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

## Eligible status maintenance

Update `ELIGIBLE_ORDER_STATUS_EN` with comma-separated `orders.order_status_en` values.
The strict resolver only sets `dx_order_id` when exactly one eligible candidate exists.
If `eligible_count == 0` or `eligible_count > 1`, the worker sets `needs_manual_link` and writes a `custom_errors` row.

## needs_review interpretation

`needs_review` means image output exists, but one or more quality gate checks failed:
- mask area ratio out of range
- too many large connected components
- bbox too thin/empty
- heuristic mismatch (e.g. `head` too wide, `car` too tall)

Details are stored in:
- `custom_assets.error`
- `custom_errors` (`error_type=processing_needs_review`)
- `custom_events` (`event_type=asset_needs_review`)
