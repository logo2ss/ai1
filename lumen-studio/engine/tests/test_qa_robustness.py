"""QA robustness tests: realistic user inputs in the user's mode (compat face path, CPU renderer).

Fast (< ~15 s together). Owner: QA.
"""

import math

import cv2
import numpy as np
import pytest

from conftest import has_model, load, needs_depth, needs_testdata

needs_compat = pytest.mark.skipif(
    not (has_model("face-detector-yunet") and has_model("face-landmarks-lbf")), reason="compat face models not installed"
)


def _ellipse(shape, cx, cy, rx, ry):
    """A loop the user drew around an object (normalised coords), possibly running off the image edge."""
    h, w = shape[:2]
    pts = np.array([[(cx + rx * math.cos(t)) * w, (cy + ry * math.sin(t)) * h] for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)])
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(np.float32)


@needs_testdata
@needs_compat
@pytest.mark.parametrize("variant", [
    pytest.param("gray", marks=pytest.mark.xfail(reason="QA 06:20: colour leaks into B&W targets (p99 chroma 0.068) - reported to F2", strict=False)),
    "rgba_alpha_kept",
])
def test_qa_compat_face_swap_on_grayscale_and_transparent_targets(variant):
    """Old B&W scans and PNGs with transparency are common inputs; the compat path must handle them."""
    from lumen_engine.face import fallback
    from lumen_engine.face.swap import FaceSwapSettings, swap_face

    src = load("faces/obama_small.jpg")
    tgt = load("faces/lin-manuel-miranda.png")
    tgt, _ = tgt[::2, ::2].copy(), None  # keep it small and fast
    if variant == "gray":
        g = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
        tgt = np.ascontiguousarray(np.dstack([g, g, g]))
    fs, ft = fallback.detect_faces(src), fallback.detect_faces(tgt)
    assert fs and ft, "compat detector must find the faces"
    out, rep, dbg = swap_face(src, max(fs, key=lambda f: f.size), tgt, max(ft, key=lambda f: f.size), FaceSwapSettings(force=True))
    assert out.shape == tgt.shape and np.isfinite(out).all()
    m = dbg["mask"]
    far = cv2.dilate((m > 0.01).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    assert np.abs(out[far] - tgt[far]).max() < 1e-4, "pixels far from the face must stay untouched"
    if variant == "gray":
        # a grey photo must stay grey: no colour cast introduced by the skin-tone transfer
        sat = np.abs(out - out.mean(axis=2, keepdims=True)).max(axis=2)
        assert float(np.percentile(sat[m > 0.5], 99)) < 0.04


@needs_testdata
def test_qa_region_swap_with_loops_running_off_the_image_edge():
    """Users draw around objects cut by the frame (the lime at the right edge of fruits.jpg)."""
    from lumen_engine.region.swap import RegionSwapSettings, swap_regions

    a = cv2.resize(load("scenes/fruits.jpg"), (256, 240), interpolation=cv2.INTER_AREA)
    b = cv2.resize(load("scenes/home.jpg"), (256, 192), interpolation=cv2.INTER_AREA)
    ma = _ellipse(a.shape, 0.86, 0.43, 0.18, 0.17)  # crosses the right border
    mb = _ellipse(b.shape, 0.30, 0.95, 0.12, 0.12)  # crosses the bottom border
    res = swap_regions(a, ma, b, mb, RegionSwapSettings(), "both")
    for key, img, m in (("a", a, ma), ("b", b, mb)):
        out = res[key][0]
        assert out.shape == img.shape and np.isfinite(out).all()
        far = cv2.dilate((m > 0).astype(np.uint8), np.ones((31, 31), np.uint8)) == 0
        assert np.abs(out[far] - img[far]).max() < 1e-4, f"{key}: outside the selection must stay bit-identical"
        assert np.abs(out[m > 0] - img[m > 0]).mean() > 0.01, f"{key}: the selection must actually change"


@needs_testdata
@needs_depth
def test_qa_reconstruct_and_render_tiny_photo_and_extreme_cameras():
    """A thumbnail-sized photo must reconstruct, and any camera the UI can send must render (no crash, no NaN)."""
    from lumen_engine.recon import render as R
    from lumen_engine.recon.scene import ReconSettings, reconstruct

    rgb = cv2.resize(load("scenes/home.jpg"), (120, 90), interpolation=cv2.INTER_AREA)
    out = reconstruct(rgb, ReconSettings(quality="fast"))
    sc = R.parse_scene(out["scene"])
    tex = [R.decode_texture(out[k]) for k in ("main_texture", "bg_texture", "back_texture")]
    cams = [((0, 0, 0), 0.0, 0.0), ((0, 0, 0), math.pi, 0.0), ((0, 0, 0), 0.0, 1.5), ((3.0, -2.0, -9.0), 0.7, -0.4), ((0, 0, -0.999), 0.0, 0.0)]
    for pos, yaw, pitch in cams:
        img, info = R.render_scene(sc, *tex, R.SceneParams(position=pos, yaw=yaw, pitch=pitch, width=64, height=40, quality="full"))
        assert img.shape == (40, 64, 3) and img.dtype == np.uint8
    ref, _ = R.render_scene(sc, *tex, R.SceneParams(width=120, height=90, quality="full", margin_fade=0.0))
    clear = np.all(np.abs(ref.astype(int) - np.array(R.SCENE_CLEAR)) <= 3, axis=2).mean()
    assert clear < 0.02, "the reference view of a tiny photo must be fully covered"
