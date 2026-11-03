# lt-preview — Lemon Tree live cut-out preview

Fly app `lt-cutout-preview` (arn, always on, shared-cpu-2x 2 GB). `POST /preview {slot_id}` from the storefront
(origins = upload service's `LT_ALLOWED_ORIGINS`) → the four Utskärning stickers as WebP data-URLs (600 px, 19 px
outline = 48 px at 1500) + review flags. The photo is read from the private `lemontree-uploads` bucket by slot id
(`lt_slots_get`, status uploaded/bound); results cached in memory by the file's sha256.

Flags (shown to staff, not shoppers): `low_res` (short side < 800 px), `small_face` (< 250 px), `no_face`
(whole subject used for Huvud), `head_unclear` (face found but the head cut keeps < 60 % of it: hands, animal,
mask → whole subject), `group_N` (N heads moved together on one backdrop).

Secrets: `PHOTOROOM_API_KEY`, `SUPABASE_URL`, `LT_SUPABASE_SECRET_KEY`, `LT_ALLOWED_ORIGINS`, `LT_PREVIEW_ADMIN_KEY`
(header `x-lt-admin` skips the cache; local copy in `.env`, git-ignored).

    cd lt-preview && fly deploy --remote-only --ha=false     # always from this folder (build context = cwd)
    python -m pytest -q tests                                # model-free tests

Measured 2026-11-03 on Fly (uncached): 2.8–5.6 s wall (PhotoRoom 0.6–1.4 s, models 0.2 s, render 1.5–3.2 s);
cached 0.35 s. Pipeline origin: bake-off ~/.cache/lt-bakeoff (fix2.py) on 81 real orders.
