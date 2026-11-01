"""Monocular depth estimation with locally installed ONNX models.

Models output *relative inverse depth* (disparity, affine-invariant). The
engine adds:

* horizontal-flip test-time augmentation (reduces left/right bias),
* optional multi-crop **detail fusion**: overlapping high-resolution crops are
  aligned to the global estimate (least-squares scale/shift) and their high
  frequencies are merged, recovering thin structures that a single 518 px
  inference misses,
* edge-aware upsampling to the image resolution (guided filter + weighted
  median), so depth discontinuities coincide with object boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..hardware import onnx_session
from ..imaging.filters import gaussian, guided_filter
from ..models import manager

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

MODEL_INPUT = {"depth-anything-v2-small": 518, "depth-anything-v2-base": 518, "midas-v21-small": 256}
# optional models (not downloaded by default) and what to use when they are missing
_FALLBACK = {"depth-anything-v2-base": "depth-anything-v2-small"}


def _infer(sess, rgb: np.ndarray, size: int) -> np.ndarray:
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC if rgb.shape[0] < size else cv2.INTER_AREA)
    x = ((np.clip(x, 0, 1) - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x})[0]
    return np.asarray(out, np.float32).reshape(out.shape[-2], out.shape[-1])


def _normalize(d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _align_relative_depth(pred: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust positive scale/shift alignment; reject inconsistent crop geometry.

    Relative-depth networks have independent scale/offset for every view.
    Negative or near-constant fits must never reverse surface ordering.
    """
    x, y = pred.ravel(), reference.ravel()
    stride = max(1, len(x)//12000)
    x, y = x[::stride].astype(np.float64), y[::stride].astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 16 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return reference.copy(), 0.0
    # Keep a small microphone, clipped highlight or crop hallucination from
    # dominating the covariance before robust fitting even starts.
    xl, xh = np.percentile(x, [5,95])
    yl, yh = np.percentile(y, [2,98])
    central = (x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
    if central.sum() >= 16:
        x, y = x[central], y[central]
    xm, ym = np.median(x), np.median(y)
    xs, ys = x-xm, y-ym
    corr = float(xs @ ys / max(np.linalg.norm(xs)*np.linalg.norm(ys), 1e-12))
    if corr < 0.5:
        return reference.copy(), 0.0
    A = np.stack([xs, np.ones_like(xs)], 1)
    coef = np.linalg.lstsq(A, ys, rcond=None)[0]
    for _ in range(3):
        residual = ys-A@coef
        scale = max(1e-6, 1.4826*np.median(np.abs(residual-np.median(residual))))
        weight = np.minimum(1., 1.5*scale/np.maximum(np.abs(residual), 1e-9))
        coef = np.linalg.lstsq(A*weight[:,None]**0.5, ys*weight**0.5, rcond=None)[0]
    if not np.isfinite(coef).all() or coef[0] <= 0:
        return reference.copy(), 0.0
    aligned = (pred-xm)*coef[0]+coef[1]+ym
    # Poor contextual crops add no reliable detail. Do not amplify their noise.
    error = np.median(np.abs(ys-A@coef))/max(float(np.percentile(y,95)-np.percentile(y,5)),1e-6)
    confidence = float(np.clip((corr-0.5)/0.4,0,1)*np.clip(1-error/0.2,0,1))
    return aligned.astype(np.float32), confidence


def _predict(sess, rgb: np.ndarray, size: int, tta: bool) -> np.ndarray:
    h, w = rgb.shape[:2]
    d = _infer(sess, rgb, size)
    if tta:
        flipped, confidence = _align_relative_depth(_infer(sess, rgb[:, ::-1].copy(), size)[:, ::-1], d)
        d = d + 0.5 * confidence * (flipped-d)
    return cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)


def estimate_depth(rgb: np.ndarray, model_id: str = "depth-anything-v2-small", detail: bool = True, progress=None,
                   info: dict | None = None, detail_side: int | None = None) -> np.ndarray:
    """Return normalised disparity in [0, 1] (1 = nearest) at the image resolution.

    ``info`` (optional dict) receives ``sky_hint``: a float map (image resolution, 0..1) of
    where the network's raw output is clamped at zero - Depth Anything puts the sky (and only
    the sky, in practice: walls, ceilings and far buildings stay clearly positive) exactly
    there. ``detect_sky(..., hint=info["sky_hint"])`` uses it to tell sky from white walls.

    ``detail_side`` caps the resolution (long side) of the detail-fusion pass: detail finer
    than the mesh that will be built from the depth is averaged away anyway, and each crop
    is a full network inference (seconds on CPUs without AVX). ``None``: up to 1600 px.
    """
    prog = progress or (lambda *_: None)
    if model_id in _FALLBACK and not manager.is_installed(model_id):
        model_id = _FALLBACK[model_id]  # optional model not downloaded: use the default one
    if info is not None:
        info["model"] = model_id
    path = manager.require(model_id)
    sess = onnx_session(str(path))
    size = MODEL_INPUT.get(model_id, 518)
    h, w = rgb.shape[:2]
    # global estimate on the whole frame
    prog(0.1)
    work, _ = (rgb, 1.0) if max(h, w) <= 1600 else (cv2.resize(rgb, (int(w * 1600 / max(h, w)), int(h * 1600 / max(h, w))), interpolation=cv2.INTER_AREA), 0)
    wh, ww = work.shape[:2]
    glob = _predict(sess, work, size, tta=True)
    if info is not None:
        hi = float(np.percentile(glob, 99.5))
        hint = (glob <= 0.01 * max(hi, 1e-6)).astype(np.float32)
        info["sky_hint"] = hint if (wh, ww) == (h, w) else cv2.resize(hint, (w, h), interpolation=cv2.INTER_LINEAR)
        info["raw_range"] = (float(np.percentile(glob, 0.5)), hi)
    prog(0.4)
    full_work = None
    if detail and detail_side and max(wh, ww) > detail_side >= size * 1.3:
        sc = detail_side / max(wh, ww)
        full_work = (work, glob)
        work = cv2.resize(work, (max(1, int(round(ww * sc))), max(1, int(round(wh * sc)))), interpolation=cv2.INTER_AREA)
        glob = cv2.resize(glob, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA)
        wh, ww = work.shape[:2]
    if detail and max(wh, ww) >= size * 1.3:
        # overlapping crops along the long axis (squares), aligned to the global estimate
        side = min(wh, ww)
        n = int(np.ceil(max(wh, ww) / (side * 0.75)))
        n = max(2, n)
        acc = np.zeros_like(glob)
        wsum = np.zeros_like(glob)
        starts = np.linspace(0, max(wh, ww) - side, n).astype(int)
        if side > size * 1.5:  # also split the short axis when the image is large
            sub = int(side * 0.62)
            starts2 = np.linspace(0, side - sub, 2).astype(int)
        else:
            sub, starts2 = side, [0]
        for i, st in enumerate(starts):
            for st2 in starts2:
                if ww >= wh:
                    x0, y0, cw, ch = st + (st2 if sub < side else 0), st2, sub, sub
                    x0 = min(x0, ww - cw)
                else:
                    x0, y0, cw, ch = st2, st + (st2 if sub < side else 0), sub, sub
                    y0 = min(y0, wh - ch)
                crop = work[y0 : y0 + ch, x0 : x0 + cw]
                d = _predict(sess, crop, size, tta=False)
                g = glob[y0 : y0 + ch, x0 : x0 + cw]
                d, confidence = _align_relative_depth(d, g)
                # Keep the global surface ordering in ambiguous/flat crops.
                limit = max(1e-5, 0.12*float(np.percentile(glob,99)-np.percentile(glob,1)))
                d = g + confidence*np.clip(d-g, -limit, limit)
                wy = np.hanning(ch + 2)[1:-1]
                wx = np.hanning(cw + 2)[1:-1]
                wgt = np.outer(wy, wx).astype(np.float32) + 1e-3
                acc[y0 : y0 + ch, x0 : x0 + cw] += d * wgt
                wsum[y0 : y0 + ch, x0 : x0 + cw] += wgt
            prog(0.4 + 0.4 * (i + 1) / len(starts))
        tiled = np.where(wsum > 1e-3, acc / np.maximum(wsum, 1e-6), glob)
        sigma = max(wh, ww) / 40
        # low frequencies (global layout) from the full-frame pass, details from the crops
        fused = gaussian(glob, sigma) + (tiled - gaussian(tiled, sigma))
    else:
        fused = glob
    if full_work is not None:  # back to the global pass resolution
        correction = fused-glob
        work, glob = full_work
        fused = glob + cv2.resize(correction, (glob.shape[1], glob.shape[0]), interpolation=cv2.INTER_CUBIC)
        wh, ww = work.shape[:2]
    disp = _normalize(fused)
    if (wh, ww) != (h, w):
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC)
    prog(0.9)
    disp = refine_edges(rgb, disp)
    prog(1.0)
    return np.clip(disp, 0, 1).astype(np.float32)


def refine_edges(rgb: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Snap depth edges to image edges (guided filter + joint weighted median)."""
    h, w = disp.shape
    r = max(2, int(max(h, w) / 300))
    d = guided_filter(rgb.astype(np.float32), disp.astype(np.float32), r, 1e-4)
    if hasattr(cv2, "ximgproc"):
        u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        try:
            d = cv2.ximgproc.weightedMedianFilter(u8, d.astype(np.float32), max(2, r), 25.5)
        except cv2.error:
            pass
    return d
