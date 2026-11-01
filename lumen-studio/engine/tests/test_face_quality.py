"""Face-swap quality regression tests (ground-truth self-swaps from engine/tools/face_train).

The source is the target photo with known photometric/geometric changes, so the
ideal result is the target itself. Errors are measured as low-frequency
CIEDE2000 (colour/shading), large-scale log-luminance (lighting), texture-energy
mismatch (blur/over-sharpening) and seam-band error; thresholds sit ~40 % above
the values measured when the benchmark was built (see tools/face_train/README.md).
Runs in compat mode (YuNet + LBF, what CPUs without AVX get).
"""

import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from conftest import has_model, needs_testdata

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

needs_fallback = pytest.mark.skipif(
    not (has_model("face-detector-yunet") and (has_model("face-mesh-onnx") or has_model("face-landmarks-lbf"))), reason="fallback face models not installed"
)

MILD = {"scale": 1.3, "rot": 5.0, "tx": 0.02, "ty": -0.01, "light_az": 0.6, "light_el": 0.7, "light_amb": 0.5,
        "light_amount": 0.6, "light_tint": [0.05, 0.0, -0.05], "grad": 0.15, "grad_dir": 2.0, "temp": 0.12, "tint": -0.03,
        "ev": 0.35, "gamma": 1.1, "contrast": 0.9, "saturation": 1.1, "blur": 0.0, "sharpen": 0.0, "noise": 0.0, "jpeg": 0}
HARD = {**MILD, "scale": 0.6, "rot": -8.0, "light_amount": 0.9, "light_az": 3.5, "temp": -0.2, "ev": -0.5, "gamma": 0.85,
        "contrast": 1.2, "saturation": 0.8, "blur": 0.8, "noise": 3.0 / 255, "jpeg": 70}

# (image, augmentation, max de_lf, max shade, max |hf_mid|, max seam_de)
CASES = [
    ("obama_small.jpg", MILD, 3.2, 0.075, 0.30, 2.6),
    ("lena.jpg", HARD, 4.5, 0.10, 0.45, 3.5),
]


@pytest.fixture(scope="module")
def compat(tmp_path_factory):
    import os

    from tools.face_train import common

    old = {k: os.environ.get(k) for k in ("LUMEN_ASSUME_NO_AVX", "LUMEN_FACE_BACKEND")}
    os.environ["FACE_TRAIN_CACHE"] = str(tmp_path_factory.mktemp("ftcache"))
    common.CACHE = Path(os.environ["FACE_TRAIN_CACHE"])
    common.set_backend_env("compat")
    yield common
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _target(common, name):
    from lumen_engine.face import fallback

    img = common.load_image(name)
    faces = fallback.detect_faces(img)
    assert faces, f"no face in {name}"
    return common.crop_face(img, max(faces, key=lambda f: f.size))


@needs_fallback
@needs_testdata
@pytest.mark.parametrize("name,aug,max_de,max_shade,max_hf,max_seam", CASES, ids=[c[0] for c in CASES])
def test_self_swap_recovers_target(compat, name, aug, max_de, max_shade, max_hf, max_seam):
    from lumen_engine.face.swap import FaceSwapSettings, swap_face

    common = compat
    T, fT = _target(common, name)
    S, fS = common.augment(T, fT, aug, seed=3)
    s = FaceSwapSettings()
    s.force = True
    out, rep, dbg = swap_face(S, fS, T, fT, s)
    m = common.selfswap_metrics(out, T, dbg["mask"], fT)
    # the raw (uncorrected) warp is far from the target: the pipeline must remove most of the change
    raw = common.warp_source(S, fS, T, fT)
    m_raw = common.selfswap_metrics(raw, T, dbg["mask"], fT)
    assert m["de_lf"] < 0.6 * m_raw["de_lf"], (m, m_raw)
    assert m["de_lf"] < max_de, m
    assert m["shade"] < max_shade, m
    assert abs(m["hf_mid"]) < max_hf, m  # neither blurred nor over-sharpened
    assert m["seam_de"] < max_seam, m
    assert m["cov"] > 0.6, m  # the face was actually replaced


def test_settings_defaults_round_trip():
    from lumen_engine.face import settings as fset
    from lumen_engine.face.settings import FaceSwapSettings

    s = FaceSwapSettings()
    assert FaceSwapSettings.from_dict(asdict(s)) == s
    assert FaceSwapSettings.from_dict(None) == s
    t = FaceSwapSettings.from_dict({"color_match": "0.5", "match_grain": "false", "keep_target_mouth": 0, "bogus": 1, "force": None})
    assert t.color_match == 0.5 and t.match_grain is False and t.keep_target_mouth is False and t.force is False
    for k, v in asdict(s).items():  # defaults are sane
        if isinstance(v, float) and k != "mask_extent":
            assert 0.0 <= v <= 1.0, k
    assert 0.8 <= s.mask_extent <= 1.1
    # internal constants: override > tuned > code default
    name = "_test_constant"
    assert fset.tv(name, 0.25) == 0.25
    fset.set_overrides({name: 0.5})
    try:
        assert fset.tv(name, 0.25) == 0.5
    finally:
        fset.set_overrides(None)
    assert fset.tv(name, 0.25) == fset.TUNED.get(name, 0.25)
    assert name in fset.seen_constants()


@needs_fallback
@needs_testdata
def test_learned_color_model(compat):
    from lumen_engine.face import learned

    if not learned.available():
        pytest.skip("learned colour weights not present")
    common = compat
    T, fT = _target(common, "obama_small.jpg")
    S, fS = common.augment(T, fT, MILD, seed=3)
    W = common.warp_source(S, fS, T, fT)
    soft = common.face_regions(T.shape[:2], fT)[0]
    t0 = time.time()
    C = learned.color_transfer(W, T, fT.xy, (0, 0), fT.size)
    dt = time.time() - t0
    assert C.shape == W.shape and C.dtype == np.float32 and np.isfinite(C).all()
    before = common.selfswap_metrics(W, T, soft, fT)
    after = common.selfswap_metrics(C, T, soft, fT)
    assert after["de_lf"] < 0.6 * before["de_lf"], (before, after)
    assert after["shade"] < before["shade"]
    assert abs(after["hf_mid"]) < 0.3  # detail kept
    assert dt < 1.0  # (< 0.3 s typical; generous for loaded CI machines)
