import cv2
import numpy as np

from lumen_engine.imaging import blend, color, matting
from lumen_engine.imaging.analysis import match_noise, noise_sigma, sharpness
from lumen_engine.imaging.inpaint import patchmatch_inpaint, scaled_patchmatch
from lumen_engine.imaging.io import decode_image, encode_image


def _ramp(h=120, w=160, c=(0.2, 0.4, 0.6)):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.dstack([np.clip(ci + 0.3 * xx / w + 0.1 * yy / h, 0, 1) for ci in c]).astype(np.float32)


def test_io_roundtrip_png_and_16bit():
    img = _ramp()
    out, alpha, _ = decode_image(encode_image(img, "png"))
    assert alpha is None and np.abs(out - img).max() < 1 / 255 + 1e-6
    out16, _, _ = decode_image(encode_image(img, "png", bit_depth=16))
    assert np.abs(out16 - img).max() < 2e-4


def test_poisson_matches_boundary_and_keeps_outside():
    dst = _ramp()
    src = _ramp(c=(0.7, 0.1, 0.3))  # very different colour, same gradients
    mask = np.zeros(dst.shape[:2], np.float32)
    cv2.circle(mask, (80, 60), 35, 1, -1)
    out = blend.poisson_blend(src, dst, mask, robust=False)
    # same gradients + Dirichlet boundary => the solution equals dst everywhere
    assert np.abs(out - dst).max() < 2e-3
    outside = cv2.dilate(mask, np.ones((3, 3))) == 0
    assert np.array_equal(out[outside], dst[outside])


def test_poisson_mixed_mode_runs():
    dst = _ramp()
    src = np.random.default_rng(0).random(dst.shape).astype(np.float32)
    mask = np.zeros(dst.shape[:2], np.float32)
    mask[30:90, 40:120] = 1
    out = blend.poisson_blend(src, dst, mask, mode="mixed")
    assert out.shape == dst.shape and np.isfinite(out).all()


def test_multiband_untouched_outside_support():
    bg = _ramp()
    fg = np.zeros_like(bg)
    m = np.zeros(bg.shape[:2], np.float32)
    m[40:80, 50:110] = 1
    m = cv2.GaussianBlur(m, (0, 0), 3)
    out = blend.multiband_blend(fg, bg, m)
    far = cv2.dilate((m > 1e-3).astype(np.uint8), np.ones((21, 21))) == 0
    assert np.array_equal(out[far], bg[far])
    assert out[60, 80].mean() < 0.05  # inside: foreground


def test_reinhard_transfers_statistics():
    rng = np.random.default_rng(1)
    src = np.clip(rng.normal(0.3, 0.05, (80, 80, 3)), 0, 1).astype(np.float32)
    ref = np.clip(rng.normal([0.7, 0.5, 0.4], 0.08, (80, 80, 3)), 0, 1).astype(np.float32)
    w = np.ones((80, 80), np.float32)
    out = color.reinhard_transfer(src, w, ref, w, std_clamp=(0.1, 10))
    m_out, _ = color.weighted_stats(color.rgb_to_lab(out), w)
    m_ref, _ = color.weighted_stats(color.rgb_to_lab(ref), w)
    assert np.abs(m_out - m_ref).max() < 1.5


def test_noise_estimation_and_matching():
    clean = _ramp(200, 200)
    rng = np.random.default_rng(2)
    noisy = np.clip(clean + rng.normal(0, 6 / 255, clean.shape), 0, 1).astype(np.float32)
    assert noise_sigma(clean) < 1 / 255
    assert 4 / 255 < noise_sigma(noisy) < 8 / 255
    w = np.ones(clean.shape[:2], np.float32)
    matched, rep = match_noise(clean, noisy, w, w)
    assert rep["action"] == "add_grain"
    assert abs(noise_sigma(matched) - noise_sigma(noisy)) < 2 / 255


def test_sharpness_metric_orders_blur():
    rng = np.random.default_rng(3)
    tex = cv2.resize(rng.random((50, 50, 3)).astype(np.float32), (200, 200), interpolation=cv2.INTER_NEAREST)
    assert sharpness(tex) > sharpness(cv2.GaussianBlur(tex, (0, 0), 2))


def test_patchmatch_completes_texture_and_keeps_known_pixels():
    rng = np.random.default_rng(4)
    tile = rng.random((8, 8, 3)).astype(np.float32)
    img = np.tile(tile, (20, 25, 1))  # periodic texture
    hole = np.zeros(img.shape[:2], bool)
    hole[60:100, 80:130] = True
    out = patchmatch_inpaint(img, hole)
    assert np.array_equal(out[~hole], img[~hole])
    err = np.abs(out[hole] - img[hole]).mean()
    assert err < 0.08, err  # a periodic texture can be completed almost exactly


def test_scaled_patchmatch_large_hole_bounded():
    img = np.tile(np.linspace(0, 1, 400, dtype=np.float32)[None, :, None], (300, 1, 3))
    hole = np.zeros(img.shape[:2], bool)
    hole[20:280, 50:350] = True
    out = scaled_patchmatch(img, hole, max_extent=128)
    assert np.array_equal(out[~hole], img[~hole]) and np.isfinite(out).all()


def test_alpha_refinement_soft_edges():
    img = np.zeros((100, 100, 3), np.float32)
    img[:, 50:] = 1
    rough = np.zeros((100, 100), np.float32)
    rough[:, 45:] = 1  # selection overshoots the real edge
    a = matting.refine_alpha(img, rough, radius=8, eps=1e-4)
    assert a[:, 40].mean() < 0.3 and a[:, 60].mean() > 0.9
