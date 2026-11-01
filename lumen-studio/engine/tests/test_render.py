"""CPU renderer for the 3D scene and the 3D body model (used when the browser has no WebGL)."""

import io
import json
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import load, needs_depth, needs_testdata

TOKEN = "t0k"  # same token as test_api.py (create_app sets a module-wide token)
HDR = {"X-Lumen-Token": TOKEN}


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-12)))


def _u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def _non_bg(img: np.ndarray, bg) -> float:
    return float(np.any(np.abs(img.astype(int) - np.array(bg)) > 6, axis=2).mean())


@pytest.fixture(scope="module")
def home_recon():
    from lumen_engine.recon.scene import ReconSettings, reconstruct

    rgb = load("scenes/home.jpg")
    return rgb, reconstruct(rgb, ReconSettings(quality="fast"))


@pytest.fixture(scope="module")
def home_scene(home_recon):
    from lumen_engine.recon import render as R

    _, out = home_recon
    tex = [R.decode_texture(out[k]) for k in ("main_texture", "bg_texture", "back_texture")]
    return (R.parse_scene(out["scene"]), *tex)


@pytest.fixture(scope="module")
def body_model():
    """A real body container (build_body3d) from a photo with a simple person-shaped mask."""
    from lumen_engine.recon.body3d import build_body3d

    rgb = load("faces/biden.jpg")
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, int(h * 0.55)), (int(w * 0.3), int(h * 0.42)), 0, 0, 360, 1, -1)
    cv2.circle(mask, (w // 2, int(h * 0.14)), int(w * 0.12), 1, -1)
    return build_body3d(rgb, mask)


@pytest.fixture(scope="module")
def client():
    from lumen_engine.server.app import create_app

    return TestClient(create_app(token=TOKEN))


# ----------------------------------------------------------------------------- scene


@needs_depth
@needs_testdata
def test_reference_view_reproduces_photo(home_recon, home_scene):
    from lumen_engine.recon import render as R

    rgb, _ = home_recon
    sc, tm, tb, tk = home_scene
    h = sc.header
    W = 512
    H = int(round(W / h["aspect"]))
    ref = cv2.resize(_u8(rgb), (W, H), interpolation=cv2.INTER_AREA)
    for quality in ("full", "preview"):
        p = R.SceneParams(width=W, height=H, quality=quality, fov=h["default_fov"], zfar=h["default_zfar"], margin_fade=0.0)
        img, info = R.render_scene(sc, tm, tb, tk, p)
        assert img.shape == (H, W, 3) and img.dtype == np.uint8
        assert info["triangles"] > 100_000
        assert _psnr(img, ref) > 24.0, quality
    # a larger frame (texture magnified) still matches
    W2, H2 = 960, int(round(960 / h["aspect"]))
    img, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(width=W2, height=H2, quality="full", margin_fade=0.0))
    assert _psnr(img, cv2.resize(_u8(rgb), (W2, H2), interpolation=cv2.INTER_AREA)) > 24.0


@needs_depth
@needs_testdata
def test_camera_motion_and_modes(home_scene):
    from lumen_engine.recon import render as R

    sc, tm, tb, tk = home_scene
    base = dict(width=480, height=360, quality="preview")
    ref, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(**base))
    moved, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(position=(0.3, 0.0, 0.0), **base))
    assert np.abs(moved.astype(int) - ref.astype(int)).mean() > 4  # parallax changes the image
    turned, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(yaw=0.2, pitch=-0.1, **base))
    assert np.abs(turned.astype(int) - ref.astype(int)).mean() > 4
    # look_at overrides yaw/pitch: looking straight ahead from the origin == reference view
    ahead, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(look_at=(0.0, 0.0, -3.0), yaw=1.0, **base))
    assert np.abs(ahead.astype(int) - ref.astype(int)).mean() < 0.5
    looked, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(position=(0.25, 0.1, 0.2), look_at=(0.0, 0.0, -2.5), **base))
    assert _non_bg(looked, R.SCENE_CLEAR) > 0.5
    for mode in ("depth", "layers"):
        img, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(position=(0.3, 0.0, 0.0), mode=mode, **base))
        assert img.shape == (360, 480, 3) and _non_bg(img, R.SCENE_CLEAR) > 0.5, mode
        assert np.abs(img.astype(int) - moved.astype(int)).mean() > 1, mode
    # layer tint: revealed background/backdrop regions turn pink-ish in 'layers' mode
    layers, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(position=(0.3, 0.0, 0.0), mode="layers", **base))
    pink = (layers[..., 0].astype(int) - layers[..., 1] > 60) & (layers[..., 2].astype(int) - layers[..., 1] > 20)
    assert pink.mean() > 0.002
    # without the hidden layers, disocclusions show the clear colour
    nobg, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(position=(0.3, 0.0, 0.0), show_background=False, **base))
    assert _non_bg(nobg, R.SCENE_CLEAR) < _non_bg(moved, R.SCENE_CLEAR)
    # margin fade darkens the outpainted border only
    faded, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(yaw=0.3, margin_fade=1.0, **base))
    unfaded, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(yaw=0.3, margin_fade=0.0, **base))
    assert faded.astype(int).sum() < unfaded.astype(int).sum()


@needs_depth
@needs_testdata
def test_preview_frame_time_is_interactive(home_scene):
    from lumen_engine.recon import render as R

    sc, tm, tb, tk = home_scene
    R.render_scene(sc, tm, tb, tk, R.SceneParams())  # JIT + caches warm
    t = time.perf_counter()
    R.render_scene(sc, tm, tb, tk, R.SceneParams(width=960, height=600, position=(0.1, 0.0, 0.0)))
    assert time.perf_counter() - t < 2.0  # typically ~30 ms; generous bound for loaded CI machines


def test_synthetic_scene_geometry_and_serial_fallback():
    """Exact projection on a flat plane, and the serial kernels (workqueue fallback) give the same frame."""
    from lumen_engine.recon import render as R

    sc, tm, tb, tk = R._synthetic_scene()
    p = R.SceneParams(width=96, height=72, quality="full", position=(0.05, 0.02, 0.0), mode="layers")
    par, _ = R.render_scene(sc, tm, tb, tk, p)
    saved = R._parallel_ok
    R._parallel_ok = False
    try:
        ser, _ = R.render_scene(sc, tm, tb, tk, p)
    finally:
        R._parallel_ok = saved
    assert np.abs(par.astype(int) - ser.astype(int)).max() <= 1
    # a fronto-parallel plane at the reference view fills exactly the frame of the photo
    gh, gw = 21, 31
    disp = np.full((gh, gw), 0.5, np.float32)
    keep = np.ones((gh - 1, gw - 1), np.uint8)
    header = {"grid_w": gw, "grid_h": gh, "extent_x": 1.25, "extent_y": 1.25, "aspect": 1.5}
    plane = R.SceneData(header, disp, keep, np.full_like(disp, -1.0), np.zeros_like(keep), None)
    img, _ = R.render_scene(plane, tm, tb, None, R.SceneParams(width=150, height=100, margin_fade=0.0, quality="full"))
    assert _non_bg(img, R.SCENE_CLEAR) > 0.999  # the grid spans 1.25x the frame
    img, _ = R.render_scene(plane, tm, tb, None, R.SceneParams(width=150, height=100, position=(0.0, 0.0, 3.0), margin_fade=0.0))
    covered = np.any(img != np.array(R.SCENE_CLEAR, np.uint8), axis=2)
    ys, xs = np.nonzero(covered)
    # plane at z = 1 / (0.5 * (1 - 1/8) + 1/8) = 1.778; seen from z=3 its half-width shrinks by 1.778 / 4.778
    expect_w = 150 * 1.25 * 1.778 / 4.778
    assert abs((xs.max() - xs.min() + 1) - expect_w) <= 2.5
    assert abs((xs.max() + xs.min()) / 2 - 74.5) <= 1.5 and abs((ys.max() + ys.min()) / 2 - 49.5) <= 1.5


# ----------------------------------------------------------------------------- body


@needs_depth
@needs_testdata
def test_body_styles(body_model):
    from lumen_engine.recon import render as R

    bd = R.parse_body(body_model["body"])
    tex = R.decode_texture(body_model["texture"])
    frames = {}
    for style in ("wireframe", "scanlines", "points", "solid"):
        for colors in ("depth", "photo", "hologram"):
            img, _ = R.render_body(bd, tex, R.BodyParams(width=480, height=360, style=style, colors=colors, quality="full"))
            assert img.shape == (360, 480, 3)
            assert _non_bg(img, R.BODY_CLEAR) > 0.02, (style, colors)
            frames[style, colors] = img
    # styles and colour schemes really differ
    assert np.abs(frames["solid", "depth"].astype(int) - frames["wireframe", "depth"]).mean() > 2
    assert np.abs(frames["solid", "photo"].astype(int) - frames["solid", "hologram"]).mean() > 10
    # orbiting the camera changes the view; hidden-line removal hides lines behind the surface
    side = dict(position=(2.2, 0.2, 0.3), width=480, height=360, style="wireframe", line_step=8)
    hidden, _ = R.render_body(bd, tex, R.BodyParams(hide_hidden=True, **side))
    visible, _ = R.render_body(bd, tex, R.BodyParams(hide_hidden=False, **side))
    assert _non_bg(hidden, R.BODY_CLEAR) > 0.01
    assert np.abs(hidden.astype(int) - frames["wireframe", "depth"]).mean() > 2
    changed = np.abs(hidden.astype(int) - visible.astype(int)).max(axis=2) > 20
    assert changed.mean() > 0.001 and _non_bg(hidden, R.BODY_CLEAR) <= _non_bg(visible, R.BODY_CLEAR)
    # ... but seen from the front nothing is behind the surface: no lines may vanish (depth bias)
    front = dict(width=480, height=360, style="wireframe", line_step=8)
    hidden, _ = R.render_body(bd, tex, R.BodyParams(hide_hidden=True, **front))
    visible, _ = R.render_body(bd, tex, R.BodyParams(hide_hidden=False, **front))
    assert (np.abs(hidden.astype(int) - visible.astype(int)).max(axis=2) > 20).mean() < 0.003
    # relief exaggeration changes the silhouette seen from the side
    flat, _ = R.render_body(bd, tex, R.BodyParams(position=(2.3, 0.0, 0.0), style="solid", relief=0.2, width=320, height=320))
    deep, _ = R.render_body(bd, tex, R.BodyParams(position=(2.3, 0.0, 0.0), style="solid", relief=2.0, width=320, height=320))
    assert _non_bg(deep, R.BODY_CLEAR) > _non_bg(flat, R.BODY_CLEAR)


def test_body_solid_is_lit_from_the_front():
    from lumen_engine.recon import render as R

    bd, tex = R._synthetic_body()
    img, _ = R.render_body(bd, tex, R.BodyParams(width=160, height=160, style="solid", colors="hologram", position=(0, 0, 1.4)))
    back, _ = R.render_body(bd, tex, R.BodyParams(width=160, height=160, style="solid", colors="hologram", position=(0, 0, -1.4)))
    lit = img[np.any(img != np.array(R.BODY_CLEAR, np.uint8), axis=2)].mean()
    dark = back[np.any(back != np.array(R.BODY_CLEAR, np.uint8), axis=2)].mean()
    assert lit > 150 and dark < lit  # the key light comes from the front


# ----------------------------------------------------------------------------- API


def _scene_files(home_recon) -> dict:
    from lumen_engine import store

    _, out = home_recon
    return {k: store.put_file(out[k], ".lsc" if k == "scene" else ".jpg") for k in ("scene", "main_texture", "bg_texture", "back_texture")}


def _jpeg(resp) -> np.ndarray:
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert float(resp.headers["X-Render-Ms"]) > 0
    return np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))


@needs_depth
@needs_testdata
def test_scene_render_api(client, home_recon):
    files = _scene_files(home_recon)
    req = {"files": files, "camera": {"position": [0.3, 0, 0], "yaw": 0.1, "pitch": 0, "look_at": None}, "fov": 60, "zfar": 8,
           "width": 320, "height": 200, "quality": "preview", "show_background": True, "mode": "color", "margin_fade": 0.35}
    r = client.post("/api/recon/render", headers=HDR, json=req)
    img = _jpeg(r)
    assert img.shape == (200, 320, 3) and int(r.headers["X-Triangles"]) > 100_000
    # full quality, look-at camera, every mode, and no backdrop texture (falls back to bg)
    for mode in ("color", "depth", "layers"):
        req2 = dict(req, quality="full", mode=mode, camera={"position": [0.1, 0.05, 0.1], "look_at": [0, 0, -2]},
                    files=dict(files, back_texture=None))
        assert _jpeg(client.post("/api/recon/render", headers=HDR, json=req2)).shape == (200, 320, 3)
    # sizes are clamped to [64, 2048]
    small = client.post("/api/recon/render", headers=HDR, json=dict(req, width=10, height=5000))
    assert _jpeg(small).shape == (2048, 64, 3)
    # bad / unknown file names -> 404
    for bad in ("../x.lsc", "nope.lsc", "..\\x.lsc"):
        r = client.post("/api/recon/render", headers=HDR, json=dict(req, files=dict(files, scene=bad)))
        assert r.status_code == 404, bad
    assert client.post("/api/recon/render", headers=HDR, json=dict(req, files=dict(files, main_texture="../etc.jpg"))).status_code == 404
    # a file that is not a scene -> 400
    assert client.post("/api/recon/render", headers=HDR, json=dict(req, files=dict(files, scene=files["main_texture"]))).status_code == 400
    # non-finite camera values -> 400 (Python's JSON accepts NaN/Infinity literals)
    raw = json.dumps(req).replace('"position": [0.3, 0, 0]', '"position": [NaN, 0, 0]')
    r = client.post("/api/recon/render", headers={**HDR, "content-type": "application/json"}, content=raw)
    assert r.status_code == 400 and r.json()["code"] == "invalid_input"
    raw = json.dumps(req).replace('"yaw": 0.1', '"yaw": Infinity')
    assert client.post("/api/recon/render", headers={**HDR, "content-type": "application/json"}, content=raw).status_code == 400
    assert client.post("/api/recon/render", headers=HDR, json=dict(req, camera={"position": [0, 0]})).status_code == 400
    assert client.post("/api/recon/render", headers={**HDR, "content-type": "application/json"}, content="{nope").status_code == 400
    # the session token is required
    assert client.post("/api/recon/render", json=req).status_code == 401


@needs_depth
@needs_testdata
def test_scene_render_concurrent_requests(client, home_recon):
    files = _scene_files(home_recon)
    results = []

    def go(i):
        req = {"files": files, "camera": {"position": [0.05 * i, 0, 0], "yaw": 0, "pitch": 0}, "width": 256, "height": 192}
        results.append(client.post("/api/recon/render", headers=HDR, json=req).status_code)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200] * 4


@needs_depth
@needs_testdata
def test_body_render_api(client, body_model):
    from lumen_engine import store

    files = {"body": store.put_file(body_model["body"], ".lbd"), "texture": store.put_file(body_model["texture"], ".jpg")}
    req = {"files": files, "camera": {"position": [0, 0.05, 2.3], "target": [0, 0, 0], "fov": 32}, "width": 300, "height": 400,
           "quality": "preview", "style": "wireframe", "colors": "depth", "line_step": 4, "hide_hidden": True, "relief": 1.0}
    for style in ("wireframe", "scanlines", "points", "solid"):
        r = client.post("/api/body3d/render", headers=HDR, json=dict(req, style=style, colors="photo"))
        img = _jpeg(r)
        assert img.shape == (400, 300, 3) and _non_bg(img, (7, 9, 13)) > 0.02, style
        assert int(r.headers["X-Triangles"]) >= 0
    for bad in ("../x.lbd", "missing.lbd"):
        assert client.post("/api/body3d/render", headers=HDR, json=dict(req, files=dict(files, body=bad))).status_code == 404
    assert client.post("/api/body3d/render", headers=HDR, json=dict(req, files=dict(files, texture="../t.jpg"))).status_code == 404
    raw = json.dumps(req).replace('"target": [0, 0, 0]', '"target": [0, -Infinity, 0]')
    assert client.post("/api/body3d/render", headers={**HDR, "content-type": "application/json"}, content=raw).status_code == 400
    assert client.post("/api/body3d/render", headers=HDR, json=dict(req, files=dict(files, body=files["texture"]))).status_code == 400
