"""Fast synthetic tests for the 3D depth/geometry helpers (planes, slopes, depth edges, sky)."""

import math

import numpy as np

from lumen_engine.recon import geometry as G

H, W = 160, 240


def _room(noise: float = 0.0, seed: int = 0):
    """Disparity of a level camera (fov 60) in front of a floor (1 below) and a frontal wall at z=6."""
    zfar = G.REF_ZFAR
    t = math.tan(math.radians(30))
    xs, ys = G.norm_coords(H, W)
    y = np.broadcast_to(ys[:, None], (H, W))
    v = y * t / (W / H)  # downwards ray slope
    z_floor = np.where(v > 1e-6, 1.0 / np.maximum(v, 1e-6), np.inf)
    z = np.minimum(z_floor, 6.0)
    d = (1.0 / z - 1.0 / zfar) / (1.0 - 1.0 / zfar)
    rng = np.random.default_rng(seed)
    d = d + noise * rng.standard_normal(d.shape)
    return d.astype(np.float32), z_floor < 6.0


def test_plane_disparity_extent_continues_plane():
    p = G.Plane(0.1, 0.5, 0.3)
    d = G.plane_disparity(p, (H, W))
    assert abs(d[0, 0] - (0.3 - 0.1 - 0.5)) < 1e-5 and abs(d[-1, -1] - (0.3 + 0.1 + 0.5)) < 1e-5
    m = 20  # a grid extending the frame by m px per side, as the scene container does
    big = G.plane_disparity(p, (H + 2 * m, W + 2 * m), ((W + 2 * m) / W, (H + 2 * m) / H))
    assert np.abs(big[m : m + H, m : m + W] - d).max() < 0.02


def test_fit_planes_finds_floor_and_wall():
    d, floor = _room(noise=0.002)
    planes = G.fit_planes(d)
    kinds = [p.kind for p in planes]
    assert "floor" in kinds and "wall" in kinds, kinds
    fl = planes[kinds.index("floor")]
    assert fl.normal[1] < -0.95  # horizontal, below the camera
    assert (fl.mask & floor).sum() > 0.7 * floor.sum()
    assert (fl.mask & ~floor).sum() < 0.05 * floor.sum()
    wall = planes[kinds.index("wall")]
    assert abs(wall.normal[2]) > 0.95  # frontal


def test_flatten_planes_removes_bulge_keeps_objects():
    d, floor = _room()
    yy, xx = np.mgrid[0:H, 0:W]
    bulge = 0.012 * np.exp(-(((xx - 120) / 50.0) ** 2 + ((yy - 125) / 18.0) ** 2))
    box = (yy > 105) & (yy < 125) & (xx > 30) & (xx < 60)  # an object standing on the floor
    noisy = (d + bulge * floor).astype(np.float32)
    noisy[box] = noisy[box] + 0.25
    planes = G.fit_planes(noisy)
    out = G.flatten_planes(noisy, planes)
    inner = floor & ~G._dil(box, 6) & (yy < H - 4) & (yy > 90)

    def planarity(a):  # rms deviation from the best affine fit d = ax + by + c over the floor
        A = np.stack([xx[inner], yy[inner], np.ones(int(inner.sum()))], 1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, a[inner].astype(np.float64), rcond=None)
        return float(np.sqrt(np.mean((A @ coef - a[inner]) ** 2)))

    assert planarity(out) < 0.5 * planarity(noisy)
    assert np.abs(out[box] - noisy[box]).max() < 1e-4  # the object is not flattened into the floor


def test_flatten_fills_far_pockets_in_planes():
    d, floor = _room()
    noisy = d.copy()
    pock = np.zeros((H, W), bool)
    pock[134:139, 80:88] = True  # reflection: looks much farther than the floor
    pock[142:146, 150:160] = True
    noisy[pock] -= 0.2
    planes = G.fit_planes(noisy)
    out = G.flatten_planes(noisy, planes)
    assert np.abs(out[pock] - d[pock]).max() < 0.03


def test_steep_floor_is_not_torn_but_steps_are_snapped():
    rng = np.random.default_rng(1)
    rgb = (0.4 + 0.25 * rng.random((H, W, 3))).astype(np.float32)  # textured (grass-like) ground
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    ramp = np.clip(0.2 + (yy - 40) / (H - 40) * 0.8, 0.2, 1.0).astype(np.float32)  # steep near ground
    ramp[:40] = 0.2
    out = G._snap_ramps(rgb, G._sharpen_depth_edges(rgb, ramp, 0.045), 0.045)
    ch, cv_, _ = G._cuts(out, 0.045)
    assert cv_[60:].mean() < 0.002 and ch[60:].mean() < 0.002  # the ground stays one surface
    # a blurred step between two differently coloured flat regions becomes a clean step
    rgb2 = np.zeros((H, W, 3), np.float32)
    rgb2[:, : W // 2] = (0.8, 0.2, 0.2)
    rgb2[:, W // 2 :] = (0.2, 0.3, 0.8)
    xx = np.mgrid[0:H, 0:W][1].astype(np.float32)
    step = (0.2 + 0.6 / (1 + np.exp((xx - W / 2) / 3.0))).astype(np.float32)
    snapped = G._snap_ramps(rgb2, step, 0.045)
    row = snapped[H // 2]
    assert np.sum(np.abs(np.diff(row)) > 0.05) <= 2  # at most a two-pixel transition
    assert row[W // 2 - 8] > 0.75 and row[W // 2 + 8] < 0.25


def test_planar_slopes_vs_blurred_step():
    xx = np.mgrid[0:H, 0:W][1].astype(np.float32)
    ramp = 0.002 * xx
    step = 0.6 / (1 + np.exp(-(xx - W / 2) / 2.0))
    assert G.planar_slopes(ramp, 10)[20:-20, 20:-20].all()
    assert not G.planar_slopes(step.astype(np.float32), 10)[H // 2, W // 2 - 3 : W // 2 + 3].any()


def test_detect_sky_uses_network_hint():
    rng = np.random.default_rng(2)
    yy = np.mgrid[0:H, 0:W][0]
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = (0.55, 0.7, 0.9)  # smooth blue sky in the upper half ...
    ground = yy >= H // 2
    rgb[ground] = (0.3 + 0.3 * rng.random((int(ground.sum()), 3))).astype(np.float32)  # ... textured ground below
    disp = np.where(ground, 0.3 + 0.7 * (yy - H // 2) / (H // 2), 0.045).astype(np.float32)  # sky lifted off 0 by filtering
    hint = (~ground).astype(np.float32)
    sky = G.detect_sky(rgb, disp, hint=hint)
    assert sky[: H // 2 - 2].mean() > 0.95 and sky[H // 2 + 2 :].mean() < 0.01
    # a plain white wall at the far end of a room: normalised to disparity 0 like a sky, but the network's raw
    # output is not zero there -> no sky
    wall = rgb.copy()
    wall[~ground] = (0.92, 0.9, 0.88)
    d2 = np.where(ground, disp, 0.0).astype(np.float32)
    assert G.detect_sky(wall, d2, hint=np.zeros((H, W), np.float32)).mean() == 0.0


def test_optional_depth_model_falls_back_to_small(monkeypatch):
    from conftest import has_model

    import pytest

    if not has_model("depth-anything-v2-small"):
        pytest.skip("depth model not installed")
    from lumen_engine.depth import estimator
    from lumen_engine.models import manager

    real = manager.is_installed
    monkeypatch.setattr(manager, "is_installed", lambda mid: False if mid == "depth-anything-v2-base" else real(mid))
    rng = np.random.default_rng(3)
    rgb = rng.random((48, 64, 3)).astype(np.float32)
    info: dict = {}
    d = estimator.estimate_depth(rgb, "depth-anything-v2-base", detail=False, info=info)
    assert info["model"] == "depth-anything-v2-small"
    assert d.shape == (48, 64) and np.isfinite(d).all()
    assert info["sky_hint"].shape == (48, 64)
