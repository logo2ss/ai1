"""Out-of-frame surroundings: plane-continued depth, constrained outpainting, far environment, rubber-sheet removal."""

import math

import numpy as np


def test_extend_depth_continues_planes_exactly():
    from lumen_engine.recon.outpaint import _extend_depth

    yy, xx = np.mgrid[0:60, 0:80].astype(np.float32)
    floor = 0.2 + 0.01 * yy  # disparity of a plane is affine in the image coordinates
    e = _extend_depth(floor, 20)
    assert e.shape == (100, 120)
    np.testing.assert_allclose(e[80:, 60], 0.2 + 0.01 * np.arange(60, 80), atol=1e-4)  # keeps coming closer below
    wall = 0.3 + 0.004 * xx
    e = _extend_depth(wall, 20)
    np.testing.assert_allclose(e[50, 100:], 0.3 + 0.004 * np.arange(80, 100), atol=1e-4)
    # a noisy border with an object edge does not produce streaks or wild slopes
    rng = np.random.default_rng(0)
    noisy = (0.4 + 0.02 * rng.standard_normal((60, 80))).astype(np.float32)
    noisy[:, 70:] = 0.9
    e = _extend_depth(noisy, 20)
    assert np.isfinite(e).all() and e[20:80, 100:].std(axis=0).max() < 0.25
    # The first continuation level must be smooth too, including corners.
    assert np.abs(np.diff(e[:20, 20:100], axis=1)).max() < 0.08  # top margin smooth along the border


def test_outpaint_patchmatch_respects_sky_and_avoid(monkeypatch):
    from lumen_engine.recon import outpaint as O

    monkeypatch.setattr(O, "available_learned", lambda: [])
    h, w, m = 60, 90, 20
    img = np.zeros((h, w, 3), np.float32)
    img[:25] = (0.55, 0.7, 0.95)  # sky
    img[25:] = (0.35, 0.3, 0.25)  # ground
    img[30:50, 40:55] = (1.0, 0.0, 0.0)  # a red "person" to avoid
    avoid = np.zeros((h, w), bool)
    avoid[28:52, 38:57] = True
    sky = np.zeros((h, w), bool)
    sky[:25] = True
    guide = O._embed(np.where(sky, O.SKY_DISP, 0.4).astype(np.float32), m)
    big, pbig, method = O._outpaint(img, img.copy(), guide, m, np.zeros((h + 2 * m, w + 2 * m), bool), "auto", lambda *_: None, avoid=avoid, sky=sky)
    assert method == "patchmatch" and big.shape == (h + 2 * m, w + 2 * m, 3)
    top = big[: m // 2, m:-m]
    bottom = big[-m // 2 :, m:-m]
    assert np.abs(top.mean((0, 1)) - (0.55, 0.7, 0.95)).max() < 0.08  # sky continues as sky
    assert np.abs(bottom.mean((0, 1)) - (0.35, 0.3, 0.25)).max() < 0.08  # ground as ground
    redness = (big[..., 0] - big[..., 1] > 0.5)
    redness[m : m + h, m : m + w] = False
    assert redness.mean() < 0.002  # the avoided object is never copied into the margin


def test_environment_panorama_wraps_and_covers_everything():
    from lumen_engine.recon.environment import build_environment, env_uv

    rng = np.random.default_rng(1)
    canvas = np.zeros((300, 450, 3), np.float32)
    canvas[:120] = (0.5, 0.7, 1.0)
    canvas[120:] = (0.3, 0.25, 0.2)
    canvas += 0.05 * rng.random(canvas.shape).astype(np.float32)
    env = build_environment(canvas, 1.5, 1.5, 1.5, 60.0, width=512)
    assert env.shape == (256, 512, 3) and np.isfinite(env).all()
    assert env.min() >= 0 and env.max() <= 1
    # seamless behind the viewer (longitude wrap) and plausible: sky-ish above, ground-ish below
    assert np.abs(env[:, 0] - env[:, -1]).max() < 0.06
    assert env[10].mean(0)[2] > env[-10].mean(0)[2]
    # forward direction maps to the centre, up to row 0
    u, v = env_uv(np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]))
    assert abs(u[0] - 0.5) < 1e-9 and abs(v[0] - 0.5) < 1e-9 and abs(v[1]) < 1e-9


def test_render_environment_fills_every_direction_and_drops_rubber_sheets():
    from lumen_engine.recon import render as R

    sc, tm, tb, tk = R._synthetic_scene()
    env = R.MipTexture(np.full((32, 64, 3), 150, np.uint8))
    p = dict(width=96, height=72, quality="full")
    back, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(yaw=math.pi, **p), env=env)
    assert (np.abs(back.astype(int) - 150) <= 2).all()  # turning around: environment, no void
    noenv, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(yaw=math.pi, **p))
    assert (noenv == np.array(R.SCENE_CLEAR, np.uint8)).all()
    side, _ = R.render_scene(sc, tm, tb, tk, R.SceneParams(yaw=0.9, **p), env=env)
    assert not (side == np.array(R.SCENE_CLEAR, np.uint8)).all(axis=2).any()
    # rubber sheets: a quad bridging a big depth step is dropped, a grazing plane is kept
    gh, gw = 20, 20
    yy, xx = np.mgrid[0:gh, 0:gw]
    disp = np.where(xx < 10, 0.9, 0.05).astype(np.float32)  # uncut step between columns 9 and 10
    keep = np.ones((gh - 1, gw - 1), np.uint8)
    X, Y, Z = R._unproject(R.SceneData({"aspect": 1.0, "extent_x": 1.0, "extent_y": 1.0}, disp, keep, disp, keep, None),
                           disp, np.arange(gh), np.arange(gw), gw, gh, 60.0, 8.0)
    out = np.empty_like(keep)
    R._drop_stretched(X, Y, Z, keep, R.STRETCH_MIN_COS, out)
    assert out[:, 9].sum() == 0 and out[:, :8].all() and out[:, 11:].all()
    floor = (0.1 + 0.8 * yy / (gh - 1)).astype(np.float32)  # a floor seen at grazing angles stays
    X, Y, Z = R._unproject(R.SceneData({"aspect": 1.0, "extent_x": 1.0, "extent_y": 1.0}, floor, keep, floor, keep, None),
                           floor, np.arange(gh), np.arange(gw), gw, gh, 60.0, 8.0)
    R._drop_stretched(X, Y, Z, keep, R.STRETCH_MIN_COS, out)
    assert out.all()
    assert R.preview_stride(1400, 1000) >= 3 and R.preview_stride(300, 200) == 1
