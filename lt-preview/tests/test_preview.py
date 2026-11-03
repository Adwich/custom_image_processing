"""Model-free tests: outline, joining, packing, request guards. Run: python -m pytest tests (from lt-preview/)."""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("LT_SUPABASE_SECRET_KEY", "test")
os.environ.setdefault("LT_ALLOWED_ORIGINS", "https://lemon-tree.store,https://jntft2-dr.myshopify.com")

import cutout  # noqa: E402
import server  # noqa: E402


def blob(h, w, boxes):
    a = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        a[y0:y1, x0:x1] = 255
    return a


def test_outline_never_clips_a_thin_part():
    a = blob(300, 300, [(100, 100, 200, 200), (149, 40, 151, 100)])  # a 2px strand sticking out
    back = cutout.backdrop(a, 19)
    assert back[a > 20].all()
    dist = ndimage.distance_transform_edt(back)
    assert dist[40, 150] >= 18  # the strand's tip still has a full outline


def test_separate_pieces_are_joined_into_one_sticker():
    a = blob(400, 400, [(40, 150, 120, 250), (260, 150, 340, 250)])
    back = cutout.backdrop(a, 19)
    assert ndimage.label(back)[1] == 1


def test_pack_moves_heads_closer_without_overlap():
    rgba = np.zeros((400, 600, 4), np.uint8)
    rgba[150:250, 40:140] = [200, 50, 50, 255]
    rgba[150:250, 460:560] = [50, 50, 200, 255]
    out = np.asarray(cutout.pack(Image.fromarray(rgba, "RGBA")))
    a = out[:, :, 3] > 30
    xs = np.where(a.any(0))[0]
    assert xs.max() - xs.min() < 520 - 20  # closer than before
    assert a.sum() == (rgba[:, :, 3] > 30).sum()  # nothing lost, nothing overlapped
    assert ndimage.label(a)[1] == 2


def test_sticker_fits_the_square():
    cut = Image.fromarray(np.dstack([np.full((500, 500, 3), 120, np.uint8), blob(500, 500, [(50, 20, 450, 480)])]), "RGBA")
    st = np.asarray(cutout.sticker(cut, "head"))
    assert st.shape[:2] == (cutout.SIZE, cutout.SIZE)
    edge = np.concatenate([st[0, :, 3], st[-1, :, 3], st[:, 0, 3], st[:, -1, 3]])
    assert (edge == 0).all()  # outline inside the print square


def test_origin_and_rate_guards():
    assert server.allowed("https://lemon-tree.store")
    assert not server.allowed("https://evil.example") and not server.allowed(None)
    ip = "203.0.113.9"
    assert all(server.rate_ok(ip) for _ in range(server.RATE[0]))
    assert not server.rate_ok(ip)
    assert server.UUID.match("43c28668-1111-2222-3333-444455556666") and not server.UUID.match("../etc/passwd")
