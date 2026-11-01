"""Surroundings outside the photo frame: depth extension and outpainting of the margins."""

from __future__ import annotations

import io
import math
import json
import struct
from dataclasses import asdict, dataclass
import cv2
import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from ..depth.estimator import estimate_depth
from ..imaging.filters import gaussian, normalized_blur
from ..imaging.inpaint import available_learned, inpaint, learned_inpaint, patchmatch_inpaint
from ..imaging.io import encode_image, resize_max
from .geometry import (SKY_DISP, REF_ZFAR, disp_to_z, z_to_disp, _behind, detect_sky, _cuts, _quad_keep, _disk, _dil, _ero, _fill_small_holes, _sharpen_depth_edges, _flip_small, _snap_ramps, _snap_inside, _near_side, _cut_sides)  # noqa: F401
from .completion import (_separated, _foreground, _occlusion_band, _background_border, _harmonic_fill)  # noqa: F401


def _robust_border_fit(strip: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per border position (columns of ``strip``, rows = distance from the border, row 0 at the
    border): robust straight-line fit d(t) = a + b t over the first ``2k`` rows.

    Returns (a, b, confidence). Two re-weighted least-squares passes (Cauchy weights) ignore
    an object edge or a ramp inside the strip; the confidence is the weighted share of samples
    that agree with the line."""
    n = min(strip.shape[0], 2 * k)
    D = strip[:n].astype(np.float64)
    t = np.arange(n, dtype=np.float64)[:, None]
    w = np.ones_like(D)
    a = np.median(D[: max(2, n // 4)], axis=0)
    b = np.zeros(D.shape[1])
    sig = np.full(D.shape[1], 0.02)
    for _ in range(3):
        sw = w.sum(0) + 1e-9
        mt = (w * t).sum(0) / sw
        md = (w * D).sum(0) / sw
        vt = (w * (t - mt) ** 2).sum(0) + 1e-9
        b = (w * (t - mt) * (D - md)).sum(0) / vt
        a = md - b * mt
        r = D - (a + b * t)
        sig = np.maximum(1.4826 * np.median(np.abs(r), axis=0), 0.004)
        w = 1.0 / (1.0 + (r / (2.0 * sig)) ** 2)
    conf = np.clip((w > 0.5).mean(0), 0.0, 1.0)
    return a.astype(np.float32), b.astype(np.float32), conf.astype(np.float32)


def _extend_depth(d: np.ndarray, m: int) -> np.ndarray:
    """Extend a (background) disparity map ``m`` px beyond every side.

    Planes stay planes: in a pinhole camera the disparity (1/z) of a plane is an affine function of
    the image coordinates, so continuing the disparity *linearly* across the border continues a
    floor, ceiling or wall in correct perspective (a floor keeps coming closer below the frame, a
    wall keeps its tilt). The slope perpendicular to each border is fitted robustly over a strip
    along it, smoothed along the border (weighted by how well a line fits) and shrunk towards a
    constant continuation where the fit is poor. The continuation is then blurred *along* the border
    more and more with distance (no streaks, the perpendicular slope is kept). Corners follow from
    extending the rows first, then the columns.
    """
    if m <= 0:
        return d.astype(np.float32).copy()
    L = max(d.shape)

    def extend_lr(a: np.ndarray) -> np.ndarray:
        hh, ww = a.shape
        k = int(np.clip(round(0.06 * L), 3, max(3, ww // 6)))
        out = np.empty((hh, ww + 2 * m), np.float32)
        out[:, m : m + ww] = a
        t = np.arange(1, m + 1, dtype=np.float32)
        sig_along = max(2.0, 0.6 * k)
        for side in (0, 1):
            strip = (a if side == 0 else a[:, ::-1])[:, : 2 * k].T  # rows: distance from the border
            fa, fb, conf = _robust_border_fit(strip, k)
            # Fitted intercept is at t=0. Averaging an inward strip changes
            # that intercept on slopes and creates a step at the corner.
            base = fa
            slope = -fb  # d/dt into the margin = - d/dt into the image
            wgt = conf ** 2 + 1e-3
            slope = (gaussian((slope * wgt).astype(np.float32)[:, None], sig_along)[:, 0] / gaussian(wgt.astype(np.float32)[:, None], sig_along)[:, 0])
            c_s = gaussian(conf.astype(np.float32)[:, None], sig_along)[:, 0]
            slope = slope * np.clip((c_s - 0.35) / 0.4, 0.0, 1.0)  # poor fits: constant continuation
            slope = np.clip(slope, -0.012, 0.012)
            ext = base[:, None] + slope[:, None] * t[None, :]
            # receding surfaces must not pass beyond the far distance (or turn around)
            ext = np.maximum(ext, np.minimum(base[:, None], 0.02))
            # blur along the border, more with distance from it
            # Every continuation level is smoothed along the border, including
            # the first margin pixel. An unsmoothed level here caused a corner
            # seam when the second axis extended the first axis's boundary.
            levels = [cv2.GaussianBlur(ext, (1, 0), sigmaX=0.01, sigmaY=max(3.0, sig_along), borderType=cv2.BORDER_REPLICATE)]
            sigs = [0.0, 0.08, 0.2, 0.45]
            for sg in sigs[1:]:
                levels.append(cv2.GaussianBlur(ext, (1, 0), sigmaX=0.01, sigmaY=max(3.0, sig_along, sg * m), borderType=cv2.BORDER_REPLICATE))
            pos = np.clip(t / m * (len(sigs) - 1), 0, len(sigs) - 1)
            res = np.zeros_like(ext)
            for i, lv in enumerate(levels):
                res += lv * np.clip(1.0 - np.abs(pos - i), 0, 1)[None, :]
            if side == 0:
                out[:, :m] = res[:, ::-1]
            else:
                out[:, m + ww :] = res
        return out

    big = extend_lr(d.astype(np.float32))
    big = extend_lr(big.T.copy()).T.copy()
    return np.clip(big, SKY_DISP, 2.5).astype(np.float32)

def _border_objects(d: np.ndarray, thr: float) -> np.ndarray:
    """Near objects cut by the frame border that no segmenter picked up (a car, an animal).

    A grey-scale opening with a large disk removes everything narrower than
    the disk but reproduces planes exactly (floors, walls), so pixels clearly
    nearer than the opened map are objects standing in front of their
    surroundings. Only components touching the border matter here.
    """
    h, w = d.shape
    f = 4
    small = cv2.resize(d.astype(np.float32), (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    r = max(3, int(round(0.12 * max(h, w) / f)))
    # the disk is cut off at the image border, where an opening no longer reproduces a
    # sloped plane: open a canvas on which the slopes continue beyond the border
    ext = _extend_depth(small, r + 1)
    opened = cv2.morphologyEx(ext, cv2.MORPH_OPEN, _disk(r), borderType=cv2.BORDER_REPLICATE)[r + 1 : -(r + 1), r + 1 : -(r + 1)]
    opened = cv2.resize(opened, (w, h), interpolation=cv2.INTER_LINEAR)
    near = disp_to_z(np.maximum(opened, SKY_DISP)) / disp_to_z(np.maximum(d, SKY_DISP)) - 1 > 3.0 * thr
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, _disk(2)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(near.astype(np.uint8), connectivity=8)
    if n <= 1:
        return near
    # only objects with a substantial contact with the border continue beyond it: a thin pole
    # or lamp simply ends at the frame (its continuation would float as a sliver)
    min_contact = max(4, int(0.04 * max(h, w)))
    keep = np.zeros(n, bool)
    for side in (lab[:2, :], lab[-2:, :], lab[:, :2].T, lab[:, -2:].T):
        ids, counts = np.unique(side.max(axis=0), return_counts=True)
        keep[ids[counts >= min_contact]] = True
    keep[0] = False
    return keep[lab]

def _extend_mask(mask: np.ndarray, m: int, rates: tuple[float, float, float, float] = (1.0, 0.35, 0.6, 0.6)) -> np.ndarray:
    """Continue a mask beyond the frame as a rounded cap: border rows/columns are
    replicated and eroded a little more with every pixel of distance (rates in
    px per px for top, bottom, left, right: heads end quickly at the top, bodies
    continue below)."""
    h, w = mask.shape
    big = np.zeros((h + 2 * m, w + 2 * m), bool)
    big[m : m + h, m : m + w] = mask
    if m <= 0 or not mask.any():
        return big

    def run(row: np.ndarray, rate: float) -> np.ndarray:
        out = np.zeros((m, row.size), bool)
        cur = row.astype(np.uint8)[None, :]
        for t in range(1, m + 1):
            e = int(rate * t)
            v = cv2.erode(cur, np.ones((1, 2 * e + 1), np.uint8)) if e > 0 else cur
            out[t - 1] = v[0] > 0
            if not out[t - 1].any():
                break
        return out

    top, bottom = run(mask[0], rates[0]), run(mask[-1], rates[1])
    big[:m, m : m + w] = top[::-1]
    big[m + h :, m : m + w] = bottom
    left, right = run(big[:, m], rates[2]), run(big[:, m + w - 1], rates[3])
    big[:, :m] |= left[::-1].T
    big[:, m + w :] |= right.T
    return big

def _embed(a: np.ndarray, m: int, mode=cv2.BORDER_REPLICATE) -> np.ndarray:
    if m <= 0:
        return a.copy()
    if a.dtype == bool:
        return cv2.copyMakeBorder(a.astype(np.uint8), m, m, m, m, mode) > 0
    return cv2.copyMakeBorder(a, m, m, m, m, mode)

_PM_SIDE = float(__import__("os").environ.get("LUMEN_OUTPAINT_SIDE", 560))  # long side of the PatchMatch outpainting canvas


def _margin_sky(sky: np.ndarray, m: int) -> np.ndarray:
    """Sky mask continued into the margins: replicated sideways, the whole top margin above a
    border that is mostly sky, never in the bottom margin."""
    h, w = sky.shape
    big = _embed(sky.astype(bool), m)
    if m <= 0:
        return big
    top = sky[0]
    if top.mean() > 0.5:
        big[:m, :] = True
    big[m + h :, :] = False
    return big


def _stretch_border(img: np.ndarray, m: int, region: np.ndarray) -> np.ndarray:
    """Continue the photo into ``region`` (margin pixels) by stretching its border pixels outwards
    (clothing continues; nothing is copied from elsewhere in the photo), softened along the border."""
    rep = _embed(img.astype(np.float32), m, cv2.BORDER_REPLICATE)
    H, W = rep.shape[:2]
    if not region.any():
        return rep
    # blur only along the border direction (rows of the top/bottom margins, columns of the sides)
    sm_h = cv2.GaussianBlur(rep, (0, 1), sigmaX=max(1.0, 0.04 * m), sigmaY=0.01)
    sm_v = cv2.GaussianBlur(rep, (1, 0), sigmaX=0.01, sigmaY=max(1.0, 0.04 * m))
    yy, xx = np.mgrid[0:H, 0:W]
    dy = np.maximum(m - yy, yy - (H - 1 - m))
    dx = np.maximum(m - xx, xx - (W - 1 - m))
    out = np.where((dy >= dx)[..., None], sm_h, sm_v)
    return out


_LEARNED_SIDE = float(__import__("os").environ.get("LUMEN_OUTPAINT_LEARNED_SIDE", 640))  # long side of the learned outpainting canvas


def _learned_outpaint(big: np.ndarray, margin: np.ndarray, m: int, mid: str, prog, avoid: np.ndarray | None = None) -> np.ndarray:
    """Learned outpainting (LaMa / MI-GAN) of the out-of-frame ``margin`` of canvas ``big``.

    Runs on a reduced canvas (long side ~640 px: the periphery needs plausible structure more than full
    detail, and every model run costs seconds on a CPU). Wide margins grow outwards in steps - learned
    inpainting degrades on very wide strips - and every step is one run per side on a strip window: the
    ring of that step along the side plus about twice its width of photo / earlier rings as context
    (the top and bottom strips also fill the corners). ``avoid`` (canvas-size, optional) is hidden from
    the model (people: no ghost copies of a face). The fill is upsampled and gets the photo's grain."""
    H, W = margin.shape
    sc = min(1.0, _LEARNED_SIDE / max(H, W))
    hs, ws = max(8, int(round(H * sc))), max(8, int(round(W * sc)))
    small = cv2.resize(big, (ws, hs), interpolation=cv2.INTER_AREA) if sc < 1 else big.copy()
    ms = max(1, int(round(m * sc)))
    yy, xx = np.mgrid[0:hs, 0:ws]
    # Chebyshev distance outside the frame rectangle (0 inside)
    dist = np.maximum(np.maximum(ms - yy, yy - (hs - 1 - ms)), np.maximum(ms - xx, xx - (ws - 1 - ms)))
    hole_s = dist > 0
    hide = None
    if avoid is not None and avoid.any():
        hide = cv2.resize(avoid.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST) > 0
        hide = (cv2.dilate(hide.astype(np.uint8), _disk(2)) > 0) & ~hole_s
    nsteps = 1 if ms <= 90 else 2
    bounds = [int(round(ms * (k + 1) / nsteps)) for k in range(nsteps)]
    runs = 4 * nsteps
    done = 0
    lo = 0
    for hi in bounds:
        ring = (dist > lo) & (dist <= hi)
        width = hi - lo
        ctx = int(max(2.2 * width, 0.25 * min(hs, ws), 48))
        # strip windows: (y0, y1, x0, x1) per side; the ring part that lies in the strip is filled
        o_lo = ms - hi  # outer edge of this ring, measured from the canvas border
        sides = [
            (o_lo, min(hs, o_lo + width + ctx), 0, ws, ring & (yy < ms)),  # top (+ corners)
            (max(0, hs - o_lo - width - ctx), hs - o_lo, 0, ws, ring & (yy > hs - 1 - ms)),  # bottom (+ corners)
            (0, hs, o_lo, min(ws, o_lo + width + ctx), ring & (xx < ms)),  # left
            (0, hs, max(0, ws - o_lo - width - ctx), ws - o_lo, ring & (xx > ws - 1 - ms)),  # right
        ]
        for (y0, y1, x0, x1, part) in sides:
            sub = part[y0:y1, x0:x1]
            done += 1
            if not sub.any():
                continue
            crop = small[y0:y1, x0:x1]
            mh = None if hide is None else (hide[y0:y1, x0:x1] | sub)
            res = learned_inpaint(crop, sub, mid, max_pixels=int(crop.shape[0] * crop.shape[1] * 1.05) + 64, model_hole=mh)
            crop[sub] = res[sub]
            part[...] = False  # filled (avoid double processing in later sides)
            prog(done / runs)
        lo = hi
    out = big.copy()
    if sc < 1:
        up = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
        from ..imaging.inpaint import _add_grain

        up = _add_grain(np.clip(up, 0, 1), big, margin, m * 4)
        out[margin] = up[margin]
    else:
        out[margin] = small[margin]
    return out


def _outpaint(work: np.ndarray, plate: np.ndarray, guide_d: np.ndarray, m: int, ext_fg: np.ndarray, method: str, prog,
              avoid: np.ndarray | None = None, sky: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Synthesise the out-of-frame margins of the photo and of the background plate.

    ``avoid`` (work-res bool, optional): pixels that must never be copied into the margin (people,
    segmented objects - a face must not appear twice). ``sky`` (work-res bool, optional): the sky
    margin may copy only sky, the ground margin only non-sky. A segmented person cut by the frame
    (``ext_fg``, canvas-size) continues by stretching its own border pixels in the photo canvas and
    by background in the plate canvas."""
    h, w = work.shape[:2]
    big = _embed(work.astype(np.float32), m, cv2.BORDER_REFLECT)
    pbig = _embed(plate.astype(np.float32), m, cv2.BORDER_REFLECT)
    H, W = big.shape[:2]
    margin = np.ones((H, W), bool)
    margin[m : H - m, m : W - m] = False
    ext_fg = ext_fg.astype(bool) & margin if ext_fg is not None else np.zeros((H, W), bool)
    if method == "auto":
        av_big = None
        if avoid is not None and avoid.any():
            av_big = _embed(avoid.astype(bool), m) & ~margin
        for mid in available_learned():
            try:
                b2 = _learned_outpaint(big, margin, m, mid, lambda p: prog(0.8 * p), avoid=av_big)
                p2 = pbig.copy()
                p2[margin] = b2[margin]
                if ext_fg.any():
                    # background behind a person cut by the frame: fill that part of the plate's margin
                    # again with the person hidden (the plate has no person; its margin context is b2)
                    hole_p = _dil(ext_fg, max(2, m // 20)) & margin
                    p2 = learned_inpaint(p2, hole_p, mid, max_pixels=int(min(H * W, 900 * 900)))
                    st = _stretch_border(work, m, ext_fg)
                    wgt = cv2.GaussianBlur(ext_fg.astype(np.float32), (0, 0), 1.5)[..., None] * margin[..., None]
                    b2 = b2 * (1 - wgt) + st * wgt
                return b2, p2, mid
            except Exception:  # noqa: BLE001 - fall back to PatchMatch
                continue
    # PatchMatch on a reduced canvas (long side <= ~900 px: the periphery needs plausibility more than
    # full detail). Photo and plate are completed jointly: one nearest-neighbour field, so a person
    # extended into the margin gets the matching background behind it.
    sc = min(1.0, _PM_SIDE / max(H, W))
    hw, hh = max(8, int(round(W * sc))), max(8, int(round(H * sc)))
    joint = np.dstack([big, pbig]).astype(np.float32)
    small = np.concatenate([cv2.resize(joint[..., :3], (hw, hh), interpolation=cv2.INTER_AREA), cv2.resize(joint[..., 3:], (hw, hh), interpolation=cv2.INTER_AREA)], -1)
    hm = cv2.resize(margin.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_NEAREST) > 0
    src = ~hm
    if avoid is not None and avoid.any():
        av = _embed(avoid.astype(bool), m) & ~margin
        av = cv2.resize(av.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_NEAREST) > 0
        av = cv2.dilate(av.astype(np.uint8), _disk(3)) > 0
        if (src & ~av).sum() > 0.25 * src.sum():
            src &= ~av
    gy, gx = np.mgrid[0:hh, 0:hw].astype(np.float32) * (0.5 / hh)  # locality: extend each border from nearby content
    # depth guide (weak, and capped at the frame's background range: a floor continued below the frame
    # must not go looking for the nearest object in the photo, e.g. the bonnet of a car)
    gds = cv2.resize(guide_d, (hw, hh), interpolation=cv2.INTER_AREA)
    cap = float(np.percentile(gds[~hm], 90)) if (~hm).any() else 1.0
    gd = np.clip(gds, SKY_DISP, max(cap, 0.05)) * 0.2
    chans = [gd, gy, gx]
    # sky only from sky, ground only from ground (the canvas depth marks the sky, also in the margin)
    skyc = _margin_sky(sky, m) if sky is not None else (guide_d <= SKY_DISP + 1e-3)
    if sky is None and m > 0 and skyc[m, m : W - m].mean() > 0.5:
        skyc[:m] = True
    if skyc.any():
        skb = cv2.resize(skyc.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        chans.append(skb * 1.5)
    g = np.dstack(chans)
    filled = patchmatch_inpaint(small, hm, source=src, quality="fast", guide=g, progress=prog)
    if sc < 1.0:
        up = np.concatenate([cv2.resize(filled[..., :3], (W, H), interpolation=cv2.INTER_CUBIC), cv2.resize(filled[..., 3:], (W, H), interpolation=cv2.INTER_CUBIC)], -1)
    else:
        up = filled
    from ..imaging.inpaint import _add_grain

    up = _add_grain(np.clip(up, 0, 1), joint, margin, m * 4, channels=6)
    big[margin] = up[..., :3][margin]
    pbig[margin] = up[..., 3:][margin]
    if ext_fg.any():
        st = _stretch_border(work, m, ext_fg)
        wgt = cv2.GaussianBlur(ext_fg.astype(np.float32), (0, 0), 1.5)[..., None] * margin[..., None]
        big = big * (1 - wgt) + st * wgt
    return big, pbig, "patchmatch"
