# Local Processing Logbook

This logbook summarizes each local process run from this chat, including what changed before the run and the observed result.

## 1) `photoroom-sandbox-20260226-181650`
- Change before run:
  - First PhotoRoom sandbox local simulation with parsed Drive-like files.
  - Head pipeline used face/pose anchor approach; no silhouette fallback yet.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
  - Head files flagged: `head_face_not_detected`, `head_cut_not_applied`
  - Body one file processed; one body file failed quality (`mask_area_ratio_out_of_range`)
- User feedback context:
  - Head extraction still included wrong region / not robust.

## 2) `photoroom-sandbox-20260226-182246`
- Change before run:
  - Added silhouette fallback for head cut when face/shoulder anchors fail.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
  - Fallback did not activate (`no_shoulder_expansion`), so head cut still not applied.
- User feedback context:
  - Needed stronger head-only behavior.

## 3) `photoroom-sandbox-20260226-182501`
- Change before run:
  - Tuned silhouette fallback thresholds to detect gradual shoulder expansion.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
  - Head fallback activated (`cut_line_source=silhouette_fallback`).
  - Head outputs still got quality failures (`mask_area_ratio_out_of_range`, `head_bbox_too_wide`).
- User feedback context:
  - “Head cut in half horizontally” issue was raised.

## 4) `photoroom-sandbox-20260226-183048`
- Change before run:
  - Made fallback cut less aggressive (lower/later cut-line selection).
- Result:
  - `total=4`, `processed=1`, `needs_review=1`, `failed=2`
  - Failures were not model logic; they were API billing errors:
    - PhotoRoom `402` (“exhausted the number of images in your plan”).
- User feedback context:
  - Run interrupted after errors; moved toward more robust architecture.

## 5) `local-try-20260227-094805`
- Change before run:
  - Implemented robust head pipeline foundation:
    - AILabTools-first for head extraction (with fallback),
    - face safety gate,
    - strict head quality metrics.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
  - Engines: `head=ailabtools`, `body=photoroom`
  - Head still flagged by quality gate (`head_face_not_detected`, `head_top_clipped`).
- User feedback context:
  - Requested saving returned segmented artifacts and stroke-layer rework.

## 6) `API-process-output`
- Change before run:
  - Added saving of raw segmentation artifact:
    - `{asset}_segmented.png` (raw API return),
    - `{asset}_mask.png`,
    - `{asset}_final.png`.
  - Initial white-stroke compositor update.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
  - Same quality reasons as previous run.
- User feedback context:
  - Reported persistent thin black halo.

## 7) `API-process-output-v2`
- Change before run:
  - Versioned run naming introduced (`v2`, `v3`, ...).
  - Same processing logic as previous run.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`

## 8) `API-process-output-v3`
- Change before run:
  - Stroke compositor changed to full-dilated white backdrop + alpha restoration.
  - Removed pre-stroke defringe from processing path.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
- User feedback context:
  - Halo still reported.

## 9) `API-process-output-v4`
- Change before run:
  - Added explicit inner-edge whitening band in stroke compositor.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
- User feedback context:
  - Halo still matched semi-transparent edge/mask boundary.

## 10) `API-process-output-v5`
- Change before run:
  - Stronger halo cleanup:
    - force white-matte for all semi-transparent pixels,
    - widen inner contour whitening and increase whitening strength.
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
- User feedback context:
  - Requested segmented-only stroke test with saved white backdrop file.

## 11) `API-process-output-v5-segmented-stroke-test`
- Change before run:
  - No API calls.
  - Used existing `*_segmented.png` as source.
  - Added exported stroke backdrop artifact `*_stroke_bg.png`.
- Result:
  - `processed=4`, `failed=0`
  - Produced:
    - `*_framed.png`
    - `*_stroke_bg.png`
    - `*_final_from_segmented.png`
- User feedback context:
  - Confirmed need to isolate layering from segmentation source.

## 12) `API-process-output-v6`
- Change before run:
  - Stroke compositor simplified exactly to:
    - build `stroke_bg`,
    - overlay `framed` directly on top (no extra edge whitening math).
- Result:
  - `total=4`, `processed=1`, `needs_review=3`, `failed=0`
- User feedback context:
  - Current baseline after simplification request.

---

## Current recurring quality outcomes across recent runs
- Head files: `head_face_not_detected`, `head_top_clipped`
- Body (one specific file): `mask_area_ratio_out_of_range`

These outcomes are independent from stroke layering and come from head/body quality gate metrics.
