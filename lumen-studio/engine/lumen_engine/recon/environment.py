"""Far environment around a single-photo 3D scene.

The reconstruction covers the photo plus an outpainted margin; everything
beyond (turning around, looking up at the ceiling or down at the floor) is
unknown. Instead of a black void the viewers show a *far environment*: an
equirectangular panorama at infinity that

* reproduces the (outpainted) canvas, lightly blurred, where it is visible,
* continues the edge bands of the canvas around the viewer at the same
  heights (mirrored, so the horizon, floor and wall colours wrap around),
  meeting smoothly behind the viewer,
* fades towards the average ceiling / sky colour above and the floor /
  ground colour below,
* gets blurrier, slightly darker and less saturated the farther it is from
  the photo (it is a soft, plausible guess, not content).

Both renderers (CPU ``render.py`` and the WebGL viewer, which fetches the
same panorama from ``POST /api/recon/environment``) look it up by view
direction with :func:`env_uv`'s convention:
``u = 0.5 + atan2(x, -z) / 2pi`` (forward = centre, right = +x),
``v = 0.5 - asin(y) / pi`` (row 0 = straight up).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

ENV_W, ENV_H = 1024, 512
SKY_DISP = -0.08  # same constant as recon.geometry (kept local: this module stays import-light)


def env_uv(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Panorama coordinates of unit view directions ``d`` (..., 3)."""
    lon = np.arctan2(d[..., 0], -d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return 0.5 + lon / (2 * math.pi), 0.5 - lat / math.pi


def _directions(W: int, H: int) -> np.ndarray:
    u = (np.arange(W, dtype=np.float64) + 0.5) / W
    v = (np.arange(H, dtype=np.float64) + 0.5) / H
    lon = (u - 0.5) * 2 * math.pi
    lat = (0.5 - v) * math.pi
    cl = np.cos(lat)[:, None]
    return np.dstack([cl * np.sin(lon)[None, :], np.repeat(np.sin(lat)[:, None], W, 1), -cl * np.cos(lon)[None, :]])


def _wrap_blur_rows(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur along axis 1 with wrap-around (the panorama is periodic in longitude)."""
    if sigma <= 0.3:
        return a.copy()
    W = a.shape[1]
    pad = min(W, int(3 * sigma) + 1)
    ext = np.concatenate([a[:, W - pad :], a, a[:, :pad]], axis=1)
    out = cv2.GaussianBlur(ext, (0, 1), sigmaX=sigma, sigmaY=0.01, borderType=cv2.BORDER_REPLICATE)
    return out[:, pad : pad + W]


def _wrap_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """2-D Gaussian blur, wrapping in longitude and replicating at the poles (large blurs run on
    a downscaled copy: same look, a fraction of the cost)."""
    if sigma <= 0.3:
        return a.copy()
    H, W = a.shape[:2]
    f = 1
    while sigma / (2 * f) >= 3.0 and W // (2 * f) >= 32:
        f *= 2
    small = a if f == 1 else cv2.resize(a, (W // f, H // f), interpolation=cv2.INTER_AREA)
    sw = small.shape[1]
    sg = sigma / f
    pad = min(sw, int(3 * sg) + 1)
    ext = np.concatenate([small[:, sw - pad :], small, small[:, :pad]], axis=1)
    out = cv2.GaussianBlur(ext, (0, 0), sg, borderType=cv2.BORDER_REPLICATE)[:, pad : pad + sw]
    if f > 1:
        # upsample with wrap-around so the seam behind the viewer stays continuous
        out = np.concatenate([out[:, -1:], out, out[:, :1]], axis=1)
        out = cv2.resize(out, ((sw + 2) * f, H), interpolation=cv2.INTER_LINEAR)[:, f : f + W]
    return out


def build_environment(canvas: np.ndarray, extent_x: float, extent_y: float, aspect: float, fov: float,
                      sky: np.ndarray | None = None, width: int = ENV_W) -> np.ndarray:
    """Equirectangular panorama (width x width/2 x 3, float32 0..1, sRGB-encoded values) of the
    far surroundings for a canvas whose texture spans ``[-extent_x, extent_x] x [-extent_y,
    extent_y]`` normalised image coordinates of a photo with horizontal field of view ``fov``.

    ``canvas``: H x W x 3 (uint8 or float 0..1), the outpainted photo (main texture).
    ``sky``: optional boolean mask (any resolution) of sky pixels on the canvas.
    """
    W = int(max(64, width)) // 2 * 2
    H = W // 2
    c = canvas.astype(np.float32)
    if c.max() > 1.5:
        c = c / 255.0
    c = c[..., :3]
    # the canvas spans ~ 2 atan(extent_x tan(fov/2)) of the 360 degrees: sample it at about the
    # panorama's resolution (area filter, no aliasing)
    tanx = math.tan(math.radians(max(5.0, min(170.0, fov))) / 2)
    ex, ey = max(1.0, float(extent_x)), max(1.0, float(extent_y))
    span = 2 * math.atan(ex * tanx) / (2 * math.pi)
    side = int(np.clip(span * W * 1.5, 32, 1024))
    ch0, cw0 = c.shape[:2]
    s = side / max(ch0, cw0)
    if s < 1:
        c = cv2.resize(c, (max(4, int(cw0 * s)), max(4, int(ch0 * s))), interpolation=cv2.INTER_AREA)
    chh, cww = c.shape[:2]

    d = _directions(W, H)
    front = d[..., 2] < -1e-4
    nz = np.where(front, -d[..., 2], 1.0)
    xn = d[..., 0] / (nz * tanx)
    yn = -d[..., 1] * aspect / (nz * tanx)
    cu = (xn / ex + 1) * 0.5
    cv_ = (yn / ey + 1) * 0.5
    cover = front & (cu >= 0) & (cu <= 1) & (cv_ >= 0) & (cv_ <= 1)
    mx = np.where(cover, cu * cww - 0.5, -10).astype(np.float32)
    my = np.where(cover, cv_ * chh - 0.5, -10).astype(np.float32)
    P = cv2.remap(c, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---- reference colours: ceiling / sky above, floor / ground below (band averages of the canvas)
    band = max(2, chh // 8)
    top_c = c[:band].reshape(-1, 3)
    bot_c = c[-band:].reshape(-1, 3)
    if sky is not None and sky.any():
        sk = cv2.resize(sky.astype(np.uint8), (cww, chh), interpolation=cv2.INTER_NEAREST) > 0
        if sk[:band].mean() > 0.3:
            top_c = c[:band][sk[:band]]
    zenith = np.median(top_c, axis=0)
    nadir = np.median(bot_c, axis=0) * 0.85

    # ---- longitude: every row with a wide enough covered interval continues around the viewer by
    # mirroring the edge bands of that interval; the two sides cross-fade behind the viewer
    cols = np.arange(W)
    has = cover.any(1)
    a = np.where(has, np.argmax(cover, 1), -1)
    b = np.where(has, W - 1 - np.argmax(cover[:, ::-1], 1), -1)
    width_row = np.where(has, b - a + 1, 0)
    wide = width_row >= 0.45 * max(1, width_row.max())
    F = P.copy()
    filled = cover.copy()
    for r in np.nonzero(wide)[0]:
        ar, br = int(a[r]), int(b[r])
        U = W - (br - ar + 1)
        if U <= 0:
            continue
        k = max(4, int(0.35 * (br - ar + 1)))
        t_r = (cols - br) % W  # 1..U on the uncovered arc
        out = (t_r >= 1) & (t_r <= U)
        t_r = t_r[out]
        t_l = U + 1 - t_r
        ph_r = (t_r - 1) % (2 * k)
        off_r = np.where(ph_r < k, ph_r, 2 * k - 1 - ph_r)
        ph_l = (t_l - 1) % (2 * k)
        off_l = np.where(ph_l < k, ph_l, 2 * k - 1 - ph_l)
        src_r = np.clip(br - off_r, ar, br)
        src_l = np.clip(ar + off_l, ar, br)
        wr = np.clip((t_l - t_r) / max(1.0, 0.35 * U) + 0.5, 0.0, 1.0)[:, None]
        F[r, out] = P[r, src_r] * wr + P[r, src_l] * (1 - wr)
        filled[r, out] = True

    # ---- latitude: above the highest / below the lowest wide row, blur the boundary row more and
    # more and fade it towards the zenith / nadir colour
    rows_w = np.nonzero(wide)[0]
    lat = (0.5 - (np.arange(H) + 0.5) / H) * math.pi
    if rows_w.size:
        r_top, r_bot = int(rows_w[0]), int(rows_w[-1])
        top_row = F[r_top : r_top + 1].copy()
        bot_row = F[r_bot : r_bot + 1].copy()
        sig_levels = (2.0, 8.0, 24.0, 64.0, 160.0)
        tops = [_wrap_blur_rows(top_row, sg)[0] for sg in sig_levels]
        bots = [_wrap_blur_rows(bot_row, sg)[0] for sg in sig_levels]

        def ramp(levels, f):
            x = np.clip(f, 0, 1) * (len(levels) - 1)
            i0 = min(int(x), len(levels) - 2)
            t = x - i0
            return levels[i0] * (1 - t) + levels[i0 + 1] * t

        for r in range(0, r_top):
            f = (lat[r] - lat[r_top]) / max(1e-6, math.pi / 2 - lat[r_top])
            row = ramp(tops, f ** 0.7)
            g = f * f * (3 - 2 * f)
            keep = filled[r]
            new = row * (1 - g) + zenith[None, :] * g
            F[r] = np.where(keep[:, None], F[r], new)
            filled[r] = True
        for r in range(r_bot + 1, H):
            f = (lat[r_bot] - lat[r]) / max(1e-6, math.pi / 2 + lat[r_bot])
            row = ramp(bots, f ** 0.7)
            g = f * f * (3 - 2 * f)
            keep = filled[r]
            new = row * (1 - g) + nadir[None, :] * g
            F[r] = np.where(keep[:, None], F[r], new)
            filled[r] = True
    else:  # pragma: no cover - degenerate (canvas covers almost nothing)
        gy = np.clip((np.arange(H) + 0.5) / H, 0, 1)[:, None, None]
        F = (zenith[None, None, :] * (1 - gy) + nadir[None, None, :] * gy) * np.ones((1, W, 1), np.float32)

    # narrow rows at the top/bottom of the coverage keep their samples (the blur below softens the seams)
    F = np.where(cover[..., None], P, F)

    # ---- blur, darken and desaturate with the angular distance from the photo
    dist = cv2.distanceTransform((~cover).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    dist = np.minimum(dist, np.roll(cv2.distanceTransform(np.roll(~cover, W // 2, 1).astype(np.uint8), cv2.DIST_L2, 5), -W // 2, 1))
    levels = [(0.0, F), (1.5, None), (5.0, None), (14.0, None), (36.0, None)]
    blurred = [F if sg == 0 else _wrap_blur(F, sg * W / 1024) for sg, _ in levels]
    # within the coverage a light blur (it is seen only through the faded margin and holes)
    key = np.clip(dist / (W / 1024 * 22.0), 0.0, None)  # 0 inside, grows outside
    x = np.clip(np.where(cover, 1.0, 1.0 + np.sqrt(key)), 0, len(blurred) - 1).astype(np.float32)
    out = np.zeros_like(F)
    for i in range(len(blurred)):
        wgt = np.clip(1.0 - np.abs(x - i), 0, 1)[..., None]
        out += blurred[i] * wgt
    far = np.clip(dist / (W * 0.25), 0, 1)[..., None]
    grey = out.mean(-1, keepdims=True)
    out = out + (grey - out) * (0.15 * far)
    out = out * (1.0 - 0.08 * far)
    return np.clip(out, 0, 1).astype(np.float32)


def scene_environment(disp: np.ndarray, canvas: np.ndarray, header: dict, fov: float, width: int = ENV_W) -> np.ndarray:
    """Panorama for a parsed 'LSC1' scene: the main texture is the canvas, sky = sky disparity."""
    sky = disp <= SKY_DISP + 1e-4
    return build_environment(canvas, float(header["extent_x"]), float(header["extent_y"]), float(header["aspect"]), fov, sky=sky, width=width)
