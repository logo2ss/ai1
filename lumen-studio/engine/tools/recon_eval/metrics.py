"""Objective metrics for novel views of a reconstructed scene.

Per rendered view (from the CPU renderer's G-buffer: layer id + texture coordinates per pixel):

* ``empty``     fraction of pixels where nothing was drawn (void / clear colour)
* ``backdrop``  fraction showing the coarse blurred backdrop (layer 2) - a low-quality guess
* ``revealed``  fraction showing the hidden background layer *behind* the main surface (disoccluded)
* ``margin``    fraction showing out-of-frame (outpainted) content of the main/background layer
* ``stretched`` fraction covered by "rubber sheet" quads: projected quads elongated > 3x relative to
                the reference view (or magnified > 4x relative to the median) and longer than 2 px
* ``guessed``   union of all of the above (everything that is not the photo seen as it was)
* ``streak``    structure-tensor coherence excess of textured pixels in guessed areas over the photo
                areas of the same frame (smears / streaks / repeated lines push it up; ~0 is ideal)

Per scene, from the container (grid resolution):

* ``tex_bleed``    fraction of the background plate behind foreground whose colour is explained by the
                   foreground palette much better than by the surrounding background palette (or equals the
                   foreground colour) - "foreground colours on the background plate"
* ``tex_streak_behind`` / ``tex_streak_margin``  coherence excess of the plate behind foreground / of the
                   outpainted margin over the photo interior
* ``tex_detail_behind`` / ``tex_detail_margin``  median gradient magnitude ratio fill / photo (1 = as
                   detailed as the photo, << 1 = blurry fill)
* ``behind_fraction`` share of the grid where a background surface lies behind the main one
"""

from __future__ import annotations

import math

import cv2
import numpy as np

EMPTY, MAIN, BG, BACK, ENV = 255, 0, 1, 2, 3


# ----------------------------------------------------------------------------- helpers


def lab(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb01, np.float32), cv2.COLOR_RGB2LAB)


def coherence(gray: np.ndarray, sigma: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """Structure-tensor coherence in [0, 1] and gradient energy (trace) per pixel."""
    g = np.ascontiguousarray(gray, np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    k = (0, 0)
    jxx = cv2.GaussianBlur(gx * gx, k, sigma)
    jyy = cv2.GaussianBlur(gy * gy, k, sigma)
    jxy = cv2.GaussianBlur(gx * gy, k, sigma)
    tr = jxx + jyy
    diff = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    return diff / (tr + 1e-6), tr


def _gray(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    return cv2.cvtColor(np.ascontiguousarray(a, np.float32), cv2.COLOR_RGB2GRAY)


def streak_excess(img, fill: np.ndarray, ref: np.ndarray, min_px: int = 200) -> float:
    """Mean coherence of textured pixels in ``fill`` minus the same in ``ref`` (NaN if too few)."""
    g = _gray(img)
    coh, en = coherence(g)
    thr = 2e-3  # ~ gradient of 0.045 per pixel: below this a region is flat (coherence is noise)
    tf = fill & (en > thr)
    tr = ref & (en > thr)
    if tf.sum() < min_px or tr.sum() < min_px:
        return float("nan")
    return float(coh[tf].mean() - coh[tr].mean())


def detail_ratio(img, fill: np.ndarray, ref: np.ndarray, min_px: int = 200) -> float:
    g = _gray(img)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    if fill.sum() < min_px or ref.sum() < min_px:
        return float("nan")
    return float((np.mean(gm[fill]) + 1e-4) / (np.mean(gm[ref]) + 1e-4))


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = np.mean(d[mask] ** 2) if mask is not None else np.mean(d**2)
    return float(10 * math.log10(255.0**2 / max(mse, 1e-12)))


def _palette(samples: np.ndarray, k: int = 6) -> np.ndarray:
    samples = np.ascontiguousarray(samples, np.float32)
    if len(samples) > 6000:
        idx = np.random.default_rng(0).choice(len(samples), 6000, replace=False)
        samples = samples[idx]
    k = int(min(k, max(1, len(samples) // 20)))
    cv2.setRNGSeed(0)
    _, _, centres = cv2.kmeans(samples, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.3), 1,
                               cv2.KMEANS_PP_CENTERS)
    return centres


def _min_dist(x: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.sqrt(((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)).min(axis=1)


def _ring(m: np.ndarray, r: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m


# ----------------------------------------------------------------------------- container-level


class SceneGeometry:
    """Grid-level masks derived from an LSC1 container (works for any version that keeps the
    disp / bg_disp grids and the header frame)."""

    def __init__(self, sc, tex_main_rgb: np.ndarray, tex_bg_rgb: np.ndarray):
        h = sc.header
        self.sc = sc
        gh, gw = sc.disp.shape
        self.gh, self.gw = gh, gw
        fr = h.get("frame") or {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
        self.frame = (float(fr["x0"]), float(fr["y0"]), float(fr["x1"]), float(fr["y1"]))
        self.behind = (sc.bg_disp > -0.5) & (sc.bg_disp < sc.disp - 1e-4)
        u = np.arange(gw) / (gw - 1.0)
        v = np.arange(gh) / (gh - 1.0)
        x0, y0, x1, y1 = self.frame
        self.in_frame = ((v[:, None] >= y0 - 1e-6) & (v[:, None] <= y1 + 1e-6)) & ((u[None, :] >= x0 - 1e-6) & (u[None, :] <= x1 + 1e-6))
        self.main_grid = cv2.resize(tex_main_rgb, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        self.bg_grid = cv2.resize(tex_bg_rgb, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def container_metrics(self) -> dict:
        """Fill quality is judged against the *local* visible surroundings (a ring around the
        hidden region, the band just inside the frame border), not the whole photo."""
        out: dict = {}
        B = self.behind & self.in_frame
        out["behind_fraction"] = float(B.mean())
        L = max(self.gw, self.gh)
        not_behind = ~cv2.dilate(self.behind.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ring = _ring(B, max(4, int(0.04 * L))) & self.in_frame & not_behind
        out["tex_bleed"] = self._bleed(B)
        out["tex_streak_behind"] = streak_excess(self.bg_grid, B, ring)
        out["tex_detail_behind"] = detail_ratio(self.bg_grid, B, ring)
        margin = ~self.in_frame
        band = _ring(margin, max(4, int(0.06 * L))) & self.in_frame & not_behind
        out["tex_streak_margin"] = streak_excess(self.main_grid, margin, band)
        out["tex_detail_margin"] = detail_ratio(self.main_grid, margin, band)
        return out

    def _bleed(self, B: np.ndarray) -> float:
        if B.sum() < 100:
            return float("nan")
        r = max(3, int(round(0.02 * max(self.gw, self.gh))))
        ring = cv2.dilate(B.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
        ring &= ~self.behind & self.in_frame
        if ring.sum() < 50:
            return float("nan")
        L_main = lab(self.main_grid)
        L_bg = lab(self.bg_grid)
        F = L_main[B]
        G = L_bg[B]
        pf = _palette(F)
        pb = _palette(L_main[ring])
        dF = _min_dist(G, pf)
        dB = _min_dist(G, pb)
        same = np.sqrt(((G - F) ** 2).sum(-1)) < 4.0
        bleed = (dB > 10.0) & ((dF < 0.6 * dB) | same)
        return float(bleed.mean())


# ----------------------------------------------------------------------------- view-level


def quad_stretch(sx: np.ndarray, sy: np.ndarray, ok: np.ndarray, sx0: np.ndarray, sy0: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Per-quad 'rubber sheet' flag for one layer: elongation > 3x vs the reference projection or
    magnification > 4x vs the median, and longer than 2 px on screen."""

    def lens(x, y):
        ex = np.hypot(x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1])
        ey = np.hypot(x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :])
        return 0.5 * (ex[:-1, :] + ex[1:, :]), 0.5 * (ey[:, :-1] + ey[:, 1:])

    lu, lv = lens(sx, sy)
    lu0, lv0 = lens(sx0, sy0)
    okq = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:] & (keep > 0)
    su = lu / np.maximum(lu0, 1e-6)
    sv = lv / np.maximum(lv0, 1e-6)
    elong = np.maximum(su, sv) / np.maximum(np.minimum(su, sv), 1e-6)
    mag = np.sqrt(su * sv)
    med = float(np.median(mag[okq])) if okq.any() else 1.0
    longest = np.maximum(lu, lv)
    return okq & (longest > 2.0) & ((elong > 3.0) | (mag > 4.0 * med))


def view_metrics(img: np.ndarray, layer: np.ndarray, u: np.ndarray, v: np.ndarray, geo: SceneGeometry,
                 stretched_main: np.ndarray | None, stretched_bg: np.ndarray | None, env_shown: bool = False) -> tuple[dict, dict]:
    """Metrics for one rendered frame + the per-pixel class masks (for the diagnostic overlay).
    ``env_shown``: pixels without a surface show the far-environment panorama (counted as ``env``,
    not as void)."""
    H, W = img.shape[:2]
    layer = layer.reshape(H, W)
    u = u.reshape(H, W)
    v = v.reshape(H, W)
    drawn = layer != EMPTY
    gw, gh = geo.gw, geo.gh
    # (texture coordinates are only defined where a surface was drawn; guard against NaN/garbage)
    uu = np.clip(np.nan_to_num(np.where(drawn, u, 0.0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    vv = np.clip(np.nan_to_num(np.where(drawn, v, 0.0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    ci = np.clip(np.rint(uu * (gw - 1)), 0, gw - 1).astype(np.int64)
    ri = np.clip(np.rint(vv * (gh - 1)), 0, gh - 1).astype(np.int64)
    qx = np.clip(np.floor(uu * (gw - 1)), 0, gw - 2).astype(np.int64)
    qy = np.clip(np.floor(vv * (gh - 1)), 0, gh - 2).astype(np.int64)
    x0, y0, x1, y1 = geo.frame
    grid_layer = (layer == MAIN) | (layer == BG)
    margin = grid_layer & ((uu < x0) | (uu > x1) | (vv < y0) | (vv > y1))
    revealed = (layer == BG) & geo.behind[ri, ci]
    # layer ids: 0 main, 1 bg, 2 backdrop, 3 far environment (D3 marks env pixels in the G-buffer),
    # 255 nothing drawn (shows the environment when one is given, else the clear colour)
    env = (layer == ENV) | (~drawn if env_shown else np.zeros_like(drawn))
    backdrop = (layer >= BACK) & drawn & (layer != ENV)
    nothing = ~drawn | env
    empty = ~drawn & ~env
    stretched = np.zeros((H, W), bool)
    if stretched_main is not None:
        stretched |= (layer == MAIN) & stretched_main[qy, qx]
    if stretched_bg is not None:
        stretched |= (layer == BG) & stretched_bg[qy, qx]
    guessed = margin | revealed | backdrop | nothing | stretched
    photo = (layer == MAIN) & ~margin & ~stretched
    m = {
        "empty": float(empty.mean()),
        "env": float(env.mean()),
        "backdrop": float(backdrop.mean()),
        "revealed": float((revealed & ~margin).mean()),
        "margin": float(margin.mean()),
        "stretched": float(stretched.mean()),
        "guessed": float(guessed.mean()),
    }
    fill = (revealed | backdrop | stretched | margin | env) & ~empty
    # ignore a 1-pixel rim around the void: the clear colour makes a strong straight edge
    fill &= ~cv2.dilate(empty.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    near_photo = photo & _ring(fill, 24)  # the visible surroundings of the guessed areas
    m["streak"] = streak_excess(img, fill, near_photo if near_photo.sum() >= 400 else photo)
    masks = {"empty": empty, "env": env, "backdrop": backdrop, "revealed": revealed & ~margin, "margin": margin, "stretched": stretched}
    return m, masks


def overlay(img: np.ndarray, masks: dict) -> np.ndarray:
    """Diagnostic tint: revealed bg = magenta, backdrop = cyan, margin = yellow, stretched = green,
    empty = red."""
    out = img.astype(np.float32).copy()
    tints = [("env", (60, 60, 255), 0.55), ("margin", (255, 220, 0), 0.35), ("revealed", (255, 0, 200), 0.45), ("backdrop", (0, 200, 255), 0.5),
             ("stretched", (0, 255, 0), 0.6), ("empty", (255, 0, 0), 0.8)]
    for k, col, a in tints:
        mk = masks.get(k)
        if mk is not None and mk.any():
            out[mk] = out[mk] * (1 - a) + np.array(col, np.float32) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def summarize(views: dict) -> dict:
    """Mean of the per-view metrics over the novel views (everything except 'ref')."""
    keys = ["empty", "env", "backdrop", "revealed", "margin", "stretched", "guessed", "streak"]
    out = {}
    for k in keys:
        vals = [v[k] for n, v in views.items() if n != "ref" and v.get(k) is not None and np.isfinite(v[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    from .cameras import SMALL_VIEWS

    for k in keys:
        vals = [views[n][k] for n in SMALL_VIEWS if n in views and views[n].get(k) is not None and np.isfinite(views[n][k])]
        out["small_" + k] = float(np.mean(vals)) if vals else float("nan")
    return out
