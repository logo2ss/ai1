"""Unit tests for the physically based face appearance transfer (face/appearance.py)."""

import numpy as np
import pytest

ap = pytest.importorskip("lumen_engine.face.appearance")


def _template_face(scale=200.0, off=(60.0, 40.0)):
    can = ap.canonical()
    if can is None:
        pytest.skip("face template not available")
    v = can["vertices"]
    xy = (v[:, :2] - v[:, :2].min(0)) / np.ptp(v[:, 1]) * scale + np.asarray(off)
    return xy


def test_canonical_mirror_and_normals():
    can = ap.canonical()
    if can is None:
        pytest.skip("face template not available")
    m = can["mirror"]
    assert can["mirror_ok"] > 0.95
    assert m[33] == 263 and m[263] == 33 and m[61] == 291 and m[1] == 1
    n = can["normals"]
    assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-6)
    # frontal face: nose tip and forehead face the camera (z points away from it)
    nv = ap.vertex_normals(_template_face())
    assert nv[1, 2] < -0.5 and nv[10, 2] < -0.5
    # left and right cheeks point to opposite sides
    assert nv[234, 0] * nv[454, 0] < 0


def test_mesh_raster_interpolates_positions():
    xy = _template_face()
    shape = (300, 330)
    r = ap.MeshRaster(xy, shape, (0, 0))
    pos = r.interp(xy.astype(np.float32))
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    err = np.hypot(pos[..., 0] - xx, pos[..., 1] - yy)[r.inside]
    assert r.inside.sum() > 10000 and np.median(err) < 0.05


def test_sh_fit_is_robust_to_dark_outliers():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(5000, 3))
    n[:, 2] = -np.abs(n[:, 2]) - 0.3  # front hemisphere
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    Y = ap.sh_basis(n)
    L = np.zeros((9, 3))
    L[0] = [0.5, 0.4, 0.3]
    L[1] = [0.2, 0.16, 0.12]  # light from +x
    img = np.clip(Y @ L, 0.01, None)
    img[:600] *= 0.2  # "beard": dark albedo outliers
    Lf, rw = ap.fit_sh(img, Y, np.ones(len(img)))
    pred = Y[600:] @ Lf
    rel = np.abs(pred - img[600:]) / img[600:]
    assert np.median(rel) < 0.03
    assert rw[:600].mean() < 0.5 * rw[600:].mean()


def _sphere_scene(light_x, albedo, size=160):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2
    r = size * 0.45
    u, v = (xx - c) / r, (yy - c) / r
    inside = (u * u + v * v) < 1
    z = -np.sqrt(np.clip(1 - u * u - v * v, 0, 1))
    n = np.stack([u, v, z], -1)
    n[~inside] = [0, 0, -1]
    shade = np.clip(0.55 - 0.45 * z * 0 + light_x * u, 0.05, None)[..., None]
    rng = np.random.default_rng(1)
    tex = 1 + 0.03 * rng.standard_normal((size, size, 1)).astype(np.float32)
    lin = np.asarray(albedo, np.float32) * shade * tex
    from lumen_engine.imaging import color

    return color.linear_to_srgb(np.clip(lin, 0, 1)), n.astype(np.float32), inside.astype(np.float32)


def test_appearance_transfer_relights_and_matches_colour():
    from lumen_engine.imaging import color

    src, n, inside = _sphere_scene(-0.35, (0.75, 0.55, 0.45))  # lit from the left, pale skin
    dst, _, _ = _sphere_scene(+0.35, (0.45, 0.30, 0.20))  # lit from the right, darker skin
    zeros = np.zeros(inside.shape, np.float32)
    parts = {"eyes": zeros, "brows": zeros, "lips": zeros, "mouth": zeros}
    out, rep = ap.appearance_transfer(src, dst, n, n, inside, parts, 140.0, lighting=1.0, color_match=1.0)
    assert rep["method"] == "sh"
    lo, lt, ls = color.luminance(out), color.luminance(dst), color.luminance(src)
    sel = inside > 0.5
    # shading now follows the target (bright on the right), not the source
    assert np.corrcoef(lo[sel], lt[sel])[0, 1] > 0.9
    assert np.corrcoef(lo[sel], ls[sel])[0, 1] < 0
    de = np.linalg.norm(color.rgb_to_lab(out)[sel].mean(0) - color.rgb_to_lab(dst)[sel].mean(0))
    assert de < 4.0


def test_warp_quality_flags_folded_and_stretched_triangles():
    xy = _template_face()
    r = ap.MeshRaster(xy, (300, 330), (0, 0))
    q_same = ap.warp_quality(r, xy, xy, None)
    assert q_same[r.inside].mean() > 0.95
    squashed = xy.copy()
    cx = xy[:, 0].mean()
    right = xy[:, 0] > cx
    squashed[right, 0] = cx + (xy[right, 0] - cx) * 0.12  # source shows the right half foreshortened
    q = ap.warp_quality(r, squashed, xy, None)
    assert q[r.inside & (np.mgrid[0:300, 0:330][1] > cx + 30)].mean() < 0.5
    assert q[r.inside & (np.mgrid[0:300, 0:330][1] < cx - 30)].mean() > 0.9


def test_identical_appearance_does_not_double_highlights():
    src, n, inside = _sphere_scene(0.2, (0.7, 0.5, 0.35))
    yy, xx = np.mgrid[:160, :160]
    highlight = (0.18*np.exp(-((xx-95)**2+(yy-65)**2)/100)).astype(np.float32)
    src = np.clip(src + highlight[..., None], 0, 1)
    z = np.zeros(inside.shape, np.float32)
    parts = {k: z for k in ('eyes','brows','lips','mouth')}
    out, _ = ap.appearance_transfer(src, src, n, n, inside, parts, 140, 0.7, 0.9)
    assert np.abs(out-src)[inside > 0.5].mean() < 0.003
