"""Face parsing (BiSeNet ONNX) and the parsing-based face-swap masks."""

import cv2
import numpy as np
import pytest

from conftest import has_model, load, needs_testdata

needs_fallback = pytest.mark.skipif(
    not (has_model("face-detector-yunet") and (has_model("face-mesh-onnx") or has_model("face-landmarks-lbf"))), reason="compat face models not installed"
)
needs_parsing = pytest.mark.skipif(not has_model("face-parsing"), reason="face-parsing model not installed")


def _small(rgb, side=480):
    h, w = rgb.shape[:2]
    s = min(1.0, side / max(h, w))
    return np.ascontiguousarray(cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)) if s < 1 else rgb


def _face(rgb, monkeypatch):
    from lumen_engine.face import backend

    monkeypatch.setenv("LUMEN_FACE_BACKEND", "opencv")  # the user's (no-AVX) compatibility path
    faces, _ = backend.detect(rgb)
    return max(faces, key=lambda f: f.size)


@needs_fallback
@needs_testdata
def test_align_matrix_levels_eyes(monkeypatch):
    from lumen_engine.face import parsing
    from lumen_engine.face.landmarks import topology

    img = load("faces/obama_small.jpg")
    f = _face(img, monkeypatch)
    M = parsing.align_matrix(f.xy)
    t = topology()
    e0 = f.xy[t["right_eye"]].mean(0) @ M[:, :2].T + M[:, 2]
    e1 = f.xy[t["left_eye"]].mean(0) @ M[:, :2].T + M[:, 2]
    assert abs(e0[1] - e1[1]) < 4.0  # eyes (nearly) level in the crop (FFHQ axis also uses the mouth)
    assert 0.2 * parsing.SIZE < e1[0] - e0[0] < 0.3 * parsing.SIZE  # CelebA-HQ scale
    assert 0.3 * parsing.SIZE < e0[1] < 0.5 * parsing.SIZE


@needs_parsing
@needs_fallback
@needs_testdata
def test_parsing_labels_face_hair_background(monkeypatch):
    from lumen_engine.face import parsing
    from lumen_engine.face.geometry import poly_mask
    from lumen_engine.face.landmarks import topology

    img = load("faces/obama_small.jpg")
    f = _face(img, monkeypatch)
    assert parsing.available()
    fp = parsing.parse_region(img, f.points)
    assert fp.probs.shape == img.shape[:2] + (19,)
    assert np.allclose(fp.probs.astype(np.float32).sum(-1), 1.0, atol=0.02)
    oval = poly_mask(img.shape[:2], f.xy[topology()["oval"]]) > 0.5
    face = fp.face() > 0.5
    assert face[oval].mean() > 0.8  # an unoccluded face is face inside its oval
    assert face[~oval].mean() < 0.05
    x0, y0, x1, y1 = f.bbox
    above = fp.group("hair")[max(0, int(y0 - 0.3 * f.size)) : max(1, int(y0 - 0.08 * f.size)), int(x0 + 0.3 * (x1 - x0)) : int(x1 - 0.3 * (x1 - x0))]
    assert above.mean() > 0.5  # hair above the forehead
    assert fp.group("background")[:, :5].mean() > 0.9
    # cached: a second call on a sub-image (same face) reuses the inference
    n = len(parsing._cache)
    sub = img[10:-10, 10:-10]
    fp2 = parsing.parse_region(sub, f.points, (10, img.shape[0] - 10, 10, img.shape[1] - 10), offset=(10, 10))
    assert len(parsing._cache) == n
    assert np.abs(fp2.face() - fp.face()[10:-10, 10:-10]).mean() < 0.01


@needs_parsing
@needs_fallback
@needs_testdata
def test_parsed_masks_respect_occluders(monkeypatch):
    """Target hair/hat stays on top; the source's microphone/hand is not transplanted."""
    from lumen_engine.face import masking, swap
    from lumen_engine.face.settings import FaceSwapSettings

    cases = [("faces/lin-manuel-miranda.png", "faces/obama_small.jpg"), ("faces/obama_small.jpg", "faces/lena.jpg")]
    for src_name, dst_name in cases:
        S, T = _small(load(src_name)), _small(load(dst_name))
        fs, ft = _face(S, monkeypatch), _face(T, monkeypatch)
        cap = {}
        orig = masking._face_masks

        def spy(*a, **k):
            cap["r"] = orig(*a, **k)
            return cap["r"]

        monkeypatch.setattr(swap, "_face_masks", spy)
        out, rep, dbg = swap.swap_face(S, fs, T, ft, FaceSwapSettings(force=True))
        d = masking.LAST_DEBUG
        assert d["method"] == "parsing" and d["source_parse"] == "source photo"
        soft, skin, feat, mouth_kept, skin_both = cap["r"]
        assert soft.dtype == np.float32 and 0 <= soft.min() and soft.max() <= 1 and soft.max() > 0.99
        lt, ls = d["labels_t"], d["labels_s"]
        k3 = np.ones((3, 3), np.uint8)
        hair_t = cv2.erode(np.isin(lt, (6, 17, 18)).astype(np.uint8), k3) > 0  # target glasses / hair / hat (interior)
        assert soft[hair_t].max(initial=0) < 0.3 and soft[hair_t].mean() < 0.01, (dst_name, soft[hair_t].max(initial=0))
        # source background (mic, hand), glasses, cloth, hair, hat are not transplanted
        foreign_s = cv2.erode(np.isin(ls, (0, 6, 16, 17, 18)).astype(np.uint8), k3) > 0
        assert soft[foreign_s].max(initial=0) < 0.3 and soft[foreign_s].mean() < 0.01, (src_name, soft[foreign_s].max(initial=0))
        assert skin.shape == soft.shape and skin.sum() > 100 and (skin_both[skin > 0.5] > 0.5).all()
        assert rep["quality"]["skin_delta_e_after"] < rep["quality"]["skin_delta_e_before"]
        # nothing outside the processing region changes
        y0, y1, x0, x1 = dbg["roi"]
        outside = np.ones(T.shape[:2], bool)
        outside[y0:y1, x0:x1] = False
        assert np.array_equal(out[outside], T[outside])


@needs_fallback
@needs_testdata
def test_masks_fall_back_without_parsing_model(monkeypatch):
    from lumen_engine.face import masking, parsing, swap
    from lumen_engine.face.settings import FaceSwapSettings
    from lumen_engine.models import manager

    real = manager.is_installed
    monkeypatch.setattr(manager, "is_installed", lambda mid: False if mid == parsing.MODEL_ID else real(mid))
    assert not parsing.available()
    S, T = _small(load("faces/obama_small.jpg")), _small(load("faces/lin-manuel-miranda.png"))
    fs, ft = _face(S, monkeypatch), _face(T, monkeypatch)
    out, rep, _ = swap.swap_face(S, fs, T, ft, FaceSwapSettings())
    assert masking.LAST_DEBUG["method"] == "heuristic"
    assert rep["analysis"]["compatible"] and rep["mask"]["coverage_px"] > 1000
    assert out.shape == T.shape


def test_polar_seam_follows_low_cost_path():
    """The fade is placed on the cheapest closed path inside the band (here a ring at r~42)."""
    from lumen_engine.face.masking import _polar_seam

    yy, xx = np.mgrid[0:120, 0:120]
    r = np.hypot(xx - 60, yy - 60)
    cost = np.ones((120, 120), np.float32)
    cost[(r > 40) & (r < 44)] = 0.0
    reg = _polar_seam(r < 50, r >= 50, cost, (60, 60), 2, 16)
    assert reg is not None and 40 <= r[reg].max() <= 45
    assert 0.9 < reg.sum() / (np.pi * 42**2) < 1.15
