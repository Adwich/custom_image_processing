"""Lemon Tree live cut-out preview: one photo -> the four Utskärning stickers + review flags.

The pipeline tuned on 81 real orders (bake-off 2026-11-02/03, ~/.cache/lt-bakeoff):
  PhotoRoom background matte (one paid call covers all four cuts)
  Huvud  = hair + face + accessories, no neck/shoulders (MediaPipe multiclass gate); 2+ faces -> every head,
           each trimmed at its own jaw, moved closer together; no human face -> the whole subject
  Kropp / Bil / Objekt = the matte (every person kept), Hela Bilden = the photo, no outline
  white outline that follows the shape and never clips thin parts (smoothing may only ADD white),
  separate pieces joined by white so the sticker stays one piece.
Framing presets come from custom-worker/src/images.py (FRAMING_PRESETS, frame_cutout), unchanged.
"""
from __future__ import annotations

import io
import os
import threading
import time

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from framing import frame_cutout, refine_cutout_alpha

try:  # iPhone photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

MODELS = os.environ.get("LT_MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
WORK = 1024          # every model runs on this copy; the preview needs no more
SIZE = 600           # preview square
OUTLINE = 19         # 48 px at 1500 (owner, 2026-11-02) scaled to 600
CUTS = {"Huvud": "head", "Kropp": "body", "Bil / Objekt": "car", "Hela Bilden": "none"}
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LOW_RES_PX = 800     # shorter side of the original
SMALL_FACE_PX = 250  # face height in the original

_lock = threading.Lock()  # MediaPipe sessions are not thread-safe
_sessions: dict = {}


class PreviewError(Exception):
    pass


# ---------- models ----------
def _mp_image(img: Image.Image):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img.convert("RGB"))))


def class_probs(img: Image.Image) -> np.ndarray:
    """(6, H, W) confidences: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions

    if "seg" not in _sessions:
        _sessions["seg"] = ImageSegmenter.create_from_options(ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=f"{MODELS}/selfie_multiclass.tflite"),
            output_confidence_masks=True, output_category_mask=False))
    res = _sessions["seg"].segment(_mp_image(img))
    out = [np.asarray(Image.fromarray((np.clip(m.numpy_view(), 0, 1) * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR),
                      np.float32) / 255.0 for m in res.confidence_masks]
    return np.stack(out)


def _faces_in(img: Image.Image) -> list[np.ndarray]:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

    if "faces" not in _sessions:
        _sessions["faces"] = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=f"{MODELS}/face_landmarker.task"), num_faces=8,
            min_face_detection_confidence=0.5))
    res = _sessions["faces"].detect(_mp_image(img))
    return [np.array([[lm[i].x * img.width, lm[i].y * img.height] for i in FACE_OVAL], np.float32) for lm in res.face_landmarks]


def faces(img: Image.Image, box) -> list[np.ndarray]:
    """Face outlines on the whole photo, the subject box and overlapping tiles of it (small faces); duplicates merged."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    crops = [(0, 0, img.width, img.height), box]
    crops += [(int(x0 + fx * bw), int(y0 + fy * bh), int(x0 + (fx + 0.5) * bw), int(y0 + (fy + 0.5) * bh))
              for fx in (0, 0.25, 0.5) for fy in (0, 0.25, 0.5)]
    found = []
    for c in crops:
        crop = img.crop(c)
        if min(crop.size) >= 64:
            found += [o + np.array([c[0], c[1]], np.float32) for o in _faces_in(crop)]
    out: list[np.ndarray] = []
    for o in sorted(found, key=lambda q: -(np.ptp(q[:, 0]) * np.ptp(q[:, 1]))):
        if all(np.linalg.norm(o.mean(0) - q.mean(0)) > 0.5 * max(np.ptp(o[:, 0]), np.ptp(q[:, 0])) for q in out):
            out.append(o)
    return out


def photoroom_matte(img: Image.Image) -> np.ndarray:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    last = None
    for attempt in range(2):
        try:
            r = requests.post(os.environ.get("PHOTOROOM_API_URL", "https://sdk.photoroom.com/v1/segment"),
                              headers={"x-api-key": os.environ["PHOTOROOM_API_KEY"], "Accept": "image/png"},
                              files={"image_file": ("photo.jpg", buf.getvalue(), "image/jpeg")}, timeout=30)
            if r.status_code < 400 and r.headers.get("content-type", "").startswith("image/"):
                m = Image.open(io.BytesIO(r.content)).convert("RGBA").getchannel("A")
                return np.asarray(m.resize(img.size, Image.BILINEAR), np.float32)
            last = f"PhotoRoom {r.status_code}: {r.text[:200]}"
        except requests.RequestException as exc:
            last = f"PhotoRoom {type(exc).__name__}"
        time.sleep(1 + attempt)
    raise PreviewError(last or "PhotoRoom failed")


# ---------- cuts ----------
def _largest(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    return lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))))


def _rgba(img: Image.Image, alpha: np.ndarray) -> Image.Image:
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)))
    return out


def head_only(img: Image.Image, matte: np.ndarray, P: np.ndarray) -> Image.Image | None:
    """Hair + face (+ glasses/hat/accessories). No neck, no shoulders, no clothes. None if no face skin."""
    cls = P.argmax(0)
    face = _largest(cls == 3)
    if not face.any():
        return None
    ys, xs = np.where(face)
    top, chin = int(ys.min()), int(ys.max())
    fh, fw = chin - top, int(xs.max() - xs.min())
    cx0, cx1 = int(xs.min()), int(xs.max())
    h, w = cls.shape
    yy, xx = np.mgrid[0:h, 0:w]
    near = (xx > cx0 - 1.0 * fw) & (xx < cx1 + 1.0 * fw) & (yy < chin + 1.5 * fh)
    hat = (cls == 4) & (yy < top + 0.1 * fh) & near  # caps/hats sit above the face
    keep = ((cls == 1) | (cls == 3) | (cls == 5) | hat) & near
    upper = ndimage.binary_fill_holes(ndimage.binary_closing(keep & (yy <= chin), iterations=2))
    lower = (cls == 1) & (yy > chin) & near  # below the chin only real hair
    head = _largest(upper | lower)
    p_keep = P[1] + P[3] + P[5] + np.where(hat, P[4], 0)
    p_drop = P[2] + np.where(hat, 0, P[4])
    soft = np.clip((p_keep - p_drop) * 2.0 + 0.5, 0, 1)
    region = ndimage.binary_dilation(head, iterations=max(2, int(0.006 * max(h, w))))
    gate = ndimage.gaussian_filter((region * soft).astype(np.float32), sigma=max(1.0, 0.002 * max(h, w)))
    a = matte * np.clip(gate, 0, 1)
    a[a < 0.12 * 255] = 0  # faint neck residue would bend the outline
    return _rgba(img, a)


def _jaw(oval: np.ndarray, width: int, drop: float) -> np.ndarray:
    """Per-column jawline y (+drop) between the jaw corners; inf elsewhere."""
    i234, i454 = FACE_OVAL.index(234), FACE_OVAL.index(454)
    lower = oval[i454:i234 + 1]
    lower = lower[np.argsort(lower[:, 0])]
    xs = np.arange(width, dtype=np.float32)
    y = np.interp(xs, lower[:, 0], lower[:, 1] + drop)
    cx = sorted([oval[FACE_OVAL.index(365), 0], oval[FACE_OVAL.index(136), 0]])
    y[(xs < cx[0]) | (xs > cx[1])] = np.inf
    return y


def heads_group(img: Image.Image, matte: np.ndarray, P: np.ndarray, ovals: list[np.ndarray]) -> Image.Image:
    """Every head; each face's jaw trim applies only inside that face's own zone (else it slices a neighbour)."""
    cls = P.argmax(0)
    h, w = cls.shape
    yy, xx = np.mgrid[0:h, 0:w]
    best = np.full((h, w), np.inf, np.float32)
    owner = np.zeros((h, w), np.int16)
    near = np.zeros((h, w), bool)
    seeds = Image.new("L", img.size, 0)
    for i, o in enumerate(ovals):
        x0, x1, y0, y1 = o[:, 0].min(), o[:, 0].max(), o[:, 1].min(), o[:, 1].max()
        fw, fh = x1 - x0, y1 - y0
        near |= (xx > x0 - 0.8 * fw) & (xx < x1 + 0.8 * fw) & (yy > y0 - 0.9 * fh) & (yy < y1 + 0.6 * fh)
        ImageDraw.Draw(seeds).polygon([tuple(p) for p in o], fill=255)
        d = np.hypot(xx - o[:, 0].mean(), yy - o[:, 1].mean()).astype(np.float32) / max(1.0, float(fh))
        owner = np.where(d < best, i, owner)
        best = np.minimum(best, d)
    seed = np.asarray(seeds) > 0
    keep = ((cls == 1) | (cls == 3) | (cls == 5) | seed) & near
    keep = ndimage.binary_fill_holes(ndimage.binary_closing(keep, iterations=3))
    lab, _ = ndimage.label(keep)
    region = np.isin(lab, np.unique(lab[seed & (lab > 0)]))
    region = ndimage.binary_dilation(region, iterations=max(2, int(0.004 * max(h, w))))
    soft = np.clip((P[1] + P[3] + P[5] + seed - P[2] - P[4]) * 2.0 + 0.5, 0, 1)
    gate = ndimage.gaussian_filter((region * soft).astype(np.float32), sigma=max(1.0, 0.002 * max(h, w)))
    a = matte * np.clip(gate, 0, 1)
    hair_acc = ndimage.binary_dilation((cls == 1) | (cls == 5), iterations=2)
    for i, o in enumerate(ovals):
        fh = float(np.ptp(o[:, 1]))
        lim = np.where(owner == i, _jaw(o, w, 0.015 * fh)[None, :], np.inf)
        feather = max(2.0, 0.006 * fh)
        a = np.where(hair_acc, a, a * np.clip((lim + feather - yy) / feather, 0, 1))
    return _rgba(img, a)


def drop_specks(cut: Image.Image, min_ratio: float = 0.02) -> Image.Image:
    a = np.asarray(cut)[:, :, 3]
    lab, n = ndimage.label(a > 30)
    if n <= 1:
        return cut
    sizes = ndimage.sum(a > 30, lab, range(1, n + 1))
    keep = ndimage.binary_dilation(np.isin(lab, 1 + np.where(sizes >= min_ratio * sizes.max())[0]), iterations=3)
    out = cut.copy()
    out.putalpha(Image.fromarray(np.where(keep, a, 0).astype(np.uint8)))
    return out


def pack(cut: Image.Image, gap_frac: float = 0.02) -> Image.Image:
    """Move separate head pieces toward their common centre (layout kept) until they nearly touch."""
    arr = np.asarray(cut)
    a = arr[:, :, 3] > 30
    h, w = a.shape
    g = max(2, int(gap_frac * max(h, w)))
    lab, n = ndimage.label(ndimage.binary_dilation(a, iterations=g))
    if n <= 1:
        return cut
    pieces = []
    for i in range(1, n + 1):
        m = (lab == i) & a
        ys, xs = np.where(m)
        pieces.append((m, np.array([xs.mean(), ys.mean()]), (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    wts = np.array([p[0].sum() for p in pieces], np.float64)
    centre = (np.array([p[1] for p in pieces]) * wts[:, None]).sum(0) / wts.sum()

    def shift(c, f):
        return np.round((centre - c) * (1 - f)).astype(int)

    best = 1.0
    for f in np.arange(1.0, 0.0, -0.02):
        cv = np.zeros((h, w), np.int32)
        for m, c, (x0, y0, x1, y1) in pieces:
            dx, dy = shift(c, f)
            cv[y0 + dy:y1 + dy, x0 + dx:x1 + dx] += m[y0:y1, x0:x1]
        if (cv > 1).any():
            break
        best = f
        if ndimage.label(ndimage.binary_dilation(cv > 0, iterations=g))[1] == 1:
            break
    out = np.zeros_like(arr)
    for m, c, (x0, y0, x1, y1) in pieces:
        dx, dy = shift(c, best)
        sub = arr[y0:y1, x0:x1].copy()
        sub[~m[y0:y1, x0:x1]] = 0
        dst = out[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        np.copyto(dst, sub, where=(sub[:, :, 3:4] > dst[:, :, 3:4]))
    return Image.fromarray(out, "RGBA")


# ---------- outline ----------
def backdrop(alpha: np.ndarray, px: int) -> np.ndarray:
    """Full ring around every visible pixel + smoothing that may only add white; pieces joined into one."""
    core = alpha > 20
    d_out = ndimage.distance_transform_edt(~core)
    ring = d_out <= px
    sr = max(2.0, px * 0.55)
    closed = ndimage.binary_fill_holes(ndimage.distance_transform_edt(d_out <= sr) > sr)
    smooth = ndimage.distance_transform_edt(~closed) <= px
    smooth = ndimage.gaussian_filter(smooth.astype(np.float32), sigma=max(1.4, px * 0.52)) >= 0.5
    back = ndimage.binary_fill_holes(ring | smooth)
    back |= ndimage.gaussian_filter(back.astype(np.float32), sigma=max(1.0, px * 0.25)) >= 0.5
    if ndimage.label(back)[1] > 1:
        d = ndimage.distance_transform_edt(~back)

        def join(rad):
            return ndimage.binary_fill_holes((ndimage.distance_transform_edt(d <= rad) > rad) | back)

        lo, hi = 0, max(2, px // 2)  # smallest closing that joins every piece: double, then bisect
        while hi <= 40 * px and ndimage.label(join(hi))[1] > 1:
            lo, hi = hi, hi * 2
        if hi <= 40 * px:
            while hi - lo > max(2, px // 4):
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if ndimage.label(join(mid))[1] == 1 else (mid, hi)
            joined = join(hi)
            return joined | (ndimage.gaussian_filter(joined.astype(np.float32), sigma=max(1.0, px * 0.25)) >= 0.5)
    return back


def _frame(cut: Image.Image, code: str, scale: float) -> Image.Image:
    import framing

    saved = framing.FRAMING_PRESETS[code]
    framing.FRAMING_PRESETS[code] = {**saved, "target_ratio": float(saved["target_ratio"]) * scale}
    try:
        return frame_cutout(cut, code, SIZE)[0]
    finally:
        framing.FRAMING_PRESETS[code] = saved


def sticker(cut: Image.Image, code: str) -> Image.Image:
    """Frame like the worker, add the outline, shrink the framing until the outline fits the square."""
    if code == "none":
        return _frame(cut, "none", 1.0)
    pad = 3 * OUTLINE
    inside = np.zeros((SIZE + 2 * pad, SIZE + 2 * pad), bool)
    inside[pad:pad + SIZE, pad:pad + SIZE] = True
    scale, tries = 1.0, 0
    while True:
        big = np.zeros((SIZE + 2 * pad, SIZE + 2 * pad, 4), np.uint8)
        big[pad:pad + SIZE, pad:pad + SIZE] = np.asarray(_frame(cut, code, scale))
        m = backdrop(big[:, :, 3], OUTLINE)
        tries += 1
        if not (m & ~inside).any() or scale <= 0.6 or tries >= 4:
            break
        ys, xs = np.where(m)  # one jump to the scale where the outlined shape fits, instead of small steps
        extent = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        scale = max(0.6, min(scale - 0.03, scale * (SIZE - 4) / extent))
    out = np.zeros_like(big)
    out[m] = [255, 255, 255, 255]
    img = Image.fromarray(out, "RGBA")
    img.alpha_composite(Image.fromarray(big, "RGBA"))
    return img.crop((pad, pad, pad + SIZE, pad + SIZE))


def _face_kept(cut: Image.Image, oval: np.ndarray) -> float:
    """Share of the detected face outline that the head cut actually keeps."""
    m = Image.new("L", cut.size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in oval], fill=255)
    face = np.asarray(m) > 0
    return float((np.asarray(cut)[:, :, 3][face] > 128).mean()) if face.any() else 0.0


def warm_up() -> None:
    """Load and run both models once on a blank image."""
    blank = Image.new("RGB", (256, 256), (128, 128, 128))
    with _lock:
        class_probs(blank)
        _faces_in(blank)


# ---------- entry point ----------
def previews(photo: bytes) -> dict:
    """{'cuts': {cut name: WebP bytes}, 'flags': [...], 'faces': n, 'head_route': ..., 'ms': {...}}."""
    t0 = time.perf_counter()
    try:
        src = ImageOps.exif_transpose(Image.open(io.BytesIO(photo))).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise PreviewError(f"not an image: {type(exc).__name__}") from exc
    img = ImageOps.contain(src, (WORK, WORK))
    k = max(src.size) / max(img.size)
    ms = {}
    matte = photoroom_matte(img)
    ms["photoroom"] = round((time.perf_counter() - t0) * 1000)
    if (matte > 30).sum() < 0.002 * matte.size:
        raise PreviewError("no subject found")
    t1 = time.perf_counter()
    with _lock:
        P = class_probs(img)
        ys, xs = np.where(matte > 30)
        ovals = faces(img, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    ms["models"] = round((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    flags = []
    if min(src.size) < LOW_RES_PX:
        flags.append("low_res")
    if ovals and max(float(np.ptp(o[:, 1])) for o in ovals) * k < SMALL_FACE_PX:
        flags.append("small_face")
    whole = refine_cutout_alpha(_rgba(img, matte), keep_components=8, min_component_px=140, min_component_ratio=0.05, smooth_sigma=0.85)
    if not ovals:
        head, head_code, route = whole, "body", "no_face_whole_subject"
        flags.append("no_face")
    elif len(ovals) >= 2:
        head, head_code, route = pack(drop_specks(heads_group(img, matte, P, ovals))), "head", f"group_{len(ovals)}_heads"
        flags.append(f"group_{len(ovals)}")
    else:
        head, head_code, route = head_only(img, matte, P), "head", "single_head"
        if head is not None and _face_kept(head, ovals[0]) < 0.6:
            head = None  # hands over the face, an animal's face, a mask: the head cut would be a scrap
        if head is None:
            head, head_code, route = whole, "body", "head_unclear_whole_subject"
            flags.append("head_unclear")
        else:
            head = drop_specks(head)
    out = {}
    for name, code in CUTS.items():
        cut = head if code == "head" else img.convert("RGBA") if code == "none" else whole
        st = sticker(cut, head_code if code == "head" else code)
        buf = io.BytesIO()
        st.save(buf, "WEBP", quality=82, method=4)
        out[name] = buf.getvalue()
    ms["render"] = round((time.perf_counter() - t2) * 1000)
    ms["total"] = round((time.perf_counter() - t0) * 1000)
    return {"cuts": out, "flags": flags, "faces": len(ovals), "head_route": route, "ms": ms,
            "size": [src.width, src.height]}
