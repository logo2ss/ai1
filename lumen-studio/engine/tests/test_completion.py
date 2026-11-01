"""Hidden-content completion (D2): learned inpainting crops/context, planar hidden depth, model mirrors."""

import numpy as np
import pytest

from lumen_engine.imaging import inpaint as IP


def _lama_installed() -> bool:
    try:
        return "lama" in IP.available_learned()
    except Exception:  # noqa: BLE001
        return False


def test_crops_cover_every_hole_pixel_with_context():
    hole = np.zeros((300, 400), bool)
    hole[40:80, 30:60] = True  # two separate regions far apart
    hole[200:260, 300:360] = True
    rects = IP._crops(hole)
    cover = np.zeros_like(hole)
    for y0, y1, x0, x1 in rects:
        cover[y0:y1, x0:x1] = True
        assert (y1 - y0) * (x1 - x0) < 300 * 400  # local context, not the whole image
    assert cover[hole].all() and len(rects) == 2
    # a ring (out-of-frame margins) is one rectangle around everything
    ring = np.ones((120, 160), bool)
    ring[10:-10, 10:-10] = False
    assert IP._crops(ring) == [(0, 120, 0, 160)]


def test_untrusted_context_only_near_the_hole():
    hole = np.zeros((200, 200), bool)
    hole[80:120, 80:120] = True
    source = np.ones_like(hole)
    source[60:140, 60:140] = False  # fringe around the hole is not trusted
    source[0:10, 0:10] = False  # far away: irrelevant
    m = IP._untrusted_context(hole, source)
    assert m[hole].all() and m[76, 100] and not m[70, 100] and not m[5, 5]
    assert IP._untrusted_context(hole, None) is None


def test_hidden_depth_continues_floor_and_wall_planes():
    from lumen_engine.recon.completion import _harmonic_fill, _hidden_depth

    h, w = 160, 240
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    y = 2 * ys / (h - 1) - 1
    floor = 0.25 + 0.55 * (y + 1) / 2  # disparity grows towards the bottom (floor recedes upwards)
    wall = np.full((h, w), 0.42, np.float32)  # back wall
    truth = np.maximum(floor, wall).astype(np.float32)  # visible surface: the nearer plane
    hole = np.zeros((h, w), bool)
    hole[40:150, 90:150] = True  # a standing person in front of the floor-wall corner
    border = np.zeros_like(hole)
    border[39:151, 89:151] = True
    border &= ~hole
    d = truth.copy()
    d[hole] = 0.9  # the person (near)
    mem = _harmonic_fill(d, hole, border)
    pl = _hidden_depth(d, hole, border)
    e_mem = np.abs(mem - truth)[hole].mean()
    e_pl = np.abs(pl - truth)[hole].mean()
    assert np.isfinite(pl).all() and np.array_equal(pl[~hole], d[~hole])
    assert e_pl < 0.5 * e_mem + 1e-4, (e_pl, e_mem)


def test_hidden_depth_falls_back_to_membrane_without_planes():
    from lumen_engine.recon.completion import _harmonic_fill, _hidden_depth

    rng = np.random.default_rng(0)
    d = (0.3 + 0.4 * rng.random((100, 120))).astype(np.float32)  # no planar structure at all
    hole = np.zeros((100, 120), bool)
    hole[30:70, 40:80] = True
    border = np.zeros_like(hole)
    border[29:71, 39:81] = True
    border &= ~hole
    assert np.allclose(_hidden_depth(d, hole, border), _harmonic_fill(d, hole, border))


def test_model_download_tries_mirrors(monkeypatch):
    from lumen_engine.models import manager
    from lumen_engine.models.registry import MODELS

    spec = MODELS["lama"]
    srcs = manager._sources(spec)
    assert srcs[0][0].startswith("https://storage.googleapis.com/") and srcs[0][1] == spec.sha256
    assert len(srcs) >= 2 and srcs[1][1] is None
    tried = []

    def fake(sp, fresh=False):
        tried.append((sp.url, sp.sha256, fresh))
        if len(tried) == 1:
            raise RuntimeError("primary down")

    monkeypatch.setattr(manager, "_download_from", fake)
    manager.download("lama")
    assert [t[0] for t in tried] == [s[0] for s in srcs[:2]] and tried[1][2] is True


@pytest.mark.skipif(not _lama_installed(), reason="LaMa model not installed")
def test_lama_fills_hole_plausibly_and_keeps_known_pixels():
    h, w = 192, 256
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.dstack([0.2 + 0.5 * xs / w, 0.3 + 0.3 * ys / h, np.full((h, w), 0.5, np.float32)]).astype(np.float32)
    img[:, 120:126] = [0.9, 0.9, 0.9]  # a vertical line that continues behind the hole
    hole = np.zeros((h, w), bool)
    hole[60:130, 90:160] = True
    out, method = IP.inpaint(img, hole, method="lama")
    assert method == "lama" and out.shape == img.shape
    assert np.array_equal(out[~hole], img[~hole])
    err = np.abs(out[hole] - img[hole]).mean()
    assert err < 0.06, err  # smooth gradient + straight line are continued
    assert out[95, 121:125].mean() > 0.7  # the line goes on through the hole


def _stub_model(path, fixed: int | None, uint8_io: bool, hole_is_one: bool):
    """Tiny ONNX stand-in with the I/O signature of a real export: fills the hole with grey (0.5)
    and reproduces the known pixels (Carve LaMa: float 0..1 in, 0..255 out, mask 1 = hole, fixed 512;
    MI-GAN pipeline v2: uint8 NCHW in/out, mask 255 = known, any size)."""
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h

    hw = [fixed, fixed] if fixed else ["height", "width"]
    et = T.UINT8 if uint8_io else T.FLOAT
    img = h.make_tensor_value_info("image", et, [1, 3, *hw])
    msk = h.make_tensor_value_info("mask", et, [1, 1, *hw])
    out = h.make_tensor_value_info("output", T.UINT8 if uint8_io else T.FLOAT, [1, 3, *hw])

    def c(n, v):
        return h.make_node("Constant", [], [n], value=h.make_tensor(n, T.FLOAT, [], [v]))

    mscale = 255.0 if uint8_io else 1.0
    iscale = 1.0 if uint8_io else 255.0  # LaMa takes 0..1 and returns 0..255
    nodes = [c("ms", mscale), c("is", iscale), c("one", 1.0), c("grey", 127.5),
             h.make_node("Cast", ["image"], ["imf0"], to=T.FLOAT), h.make_node("Mul", ["imf0", "is"], ["imf"]),
             h.make_node("Cast", ["mask"], ["mf0"], to=T.FLOAT), h.make_node("Div", ["mf0", "ms"], ["mk"])]
    if hole_is_one:
        nodes += [h.make_node("Sub", ["one", "mk"], ["keepw"]), h.make_node("Identity", ["mk"], ["holew"])]
    else:
        nodes += [h.make_node("Identity", ["mk"], ["keepw"]), h.make_node("Sub", ["one", "mk"], ["holew"])]
    nodes += [h.make_node("Mul", ["imf", "keepw"], ["keep"]), h.make_node("Mul", ["holew", "grey"], ["fill"]),
              h.make_node("Add", ["keep", "fill"], ["of"])]
    nodes.append(h.make_node("Cast", ["of"], ["output"], to=T.UINT8) if uint8_io else h.make_node("Identity", ["of"], ["output"]))
    m = h.make_model(h.make_graph(nodes, "stub", [img, msk], [out]), opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


@pytest.mark.parametrize("kind", ["carve_lama_fixed", "migan_pipeline"])
def test_learned_plumbing_with_stub_exports(tmp_path, monkeypatch, kind):
    from lumen_engine.models import manager

    fixed, u8, hole_one, mid = (64, False, True, "lama") if kind == "carve_lama_fixed" else (None, True, False, "migan")
    p = tmp_path / f"{kind}.onnx"
    _stub_model(p, fixed, u8, hole_one)
    monkeypatch.setattr(manager, "require", lambda model_id: p)
    ys, xs = np.mgrid[0:90, 0:150].astype(np.float32)
    img = np.dstack([0.8 + 0.1 * xs / 150, 0.75 + 0.1 * ys / 90, np.full((90, 150), 0.85, np.float32)]).astype(np.float32)
    hole = np.zeros((90, 150), bool)
    hole[30:60, 5:40] = True  # near the left border: the square crop is mirrored in
    hole[70:72, 100:140] = True  # a thin strip: must survive the downscale
    source = np.ones_like(hole)
    source[25:65, 0:45] = False  # untrusted fringe around the first hole
    out, method = IP.inpaint(img, hole, source=source, method=mid)
    assert method == mid and out.shape == img.shape
    assert np.array_equal(out[~hole], img[~hole])  # known pixels untouched
    from scipy import ndimage

    inner = ndimage.binary_erosion(hole, iterations=3)
    assert np.abs(out[inner] - 0.5).max() < 0.03  # the stub's grey fill landed in the hole
    assert out[70:72, 100:140].mean() < 0.72  # the thin strip was filled too (not dropped by the downscale)
