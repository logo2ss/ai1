"""Learned colour / lighting correction for face replacement.

A regularised linear model (ridge regression, weights in ``data/learned_color.npz``
- plain arrays, no pickle) predicts, per pixel, the CIELAB correction that makes
the warped source face consistent with the target photo. It was fitted on the
self-swap benchmark of ``engine/tools/face_train`` (thousands of augmented
source/target pairs with ground truth: white balance, exposure, gamma, contrast,
saturation, 3D directional relighting, coloured light, blur, noise, JPEG, scale).

The model is a learned *combination of transfer operators* rather than a black
box, so it generalises from the self-swap data to cross-identity swaps:

* local mean differences target-vs-source at several scales (>= 6 % of the face
  size, measured on skin only so the target's eyes/brows/lips - its identity -
  never leak into the new face), per facial region (skin / features / border),
* global Reinhard and histogram-matching corrections on skin,
* multiplicative-lighting terms (shadows scale detail contrast and chroma),
* a local-contrast term.

The fitted weights decide how much of each operator to apply per Lab channel and
region. Inference is a handful of blurs and one matrix product (< 0.1 s for a
typical face ROI).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("lumen.face.learned")
DATA = Path(__file__).parent / "data" / "learned_color.npz"
SCALES = (0.06, 0.12, 0.25)  # local-statistics scales, fraction of the face size
VERSION = 1


# --------------------------------------------------------------------------- helpers
def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large sigmas run on a downsampled copy (same result for smooth fields)."""
    if sigma <= 0.3:
        return x
    h, w = x.shape[:2]
    f = int(max(1, sigma // 4))
    if f > 1 and min(h, w) // f >= 8:
        small = cv2.resize(x, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f, borderType=cv2.BORDER_REFLECT)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def _nblur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = _blur(x * (w[..., None] if x.ndim == 3 else w), sigma)
    den = _blur(w, sigma)
    return num / np.maximum(den[..., None] if x.ndim == 3 else den, 1e-4)


def _wstats(x: np.ndarray, w: np.ndarray):
    sw = float(w.sum()) + 1e-6
    m = (x * w[..., None]).reshape(-1, 3).sum(0) / sw
    v = (((x - m) ** 2) * w[..., None]).reshape(-1, 3).sum(0) / sw
    return m.astype(np.float32), np.sqrt(v + 1e-6).astype(np.float32)


def _hist_delta(src: np.ndarray, ws: np.ndarray, ref: np.ndarray, wr: np.ndarray, bins: int = 64) -> np.ndarray:
    """Per-channel histogram-matching correction (smoothed transfer curve) evaluated at every pixel."""
    out = np.zeros_like(src)
    qs = np.linspace(0.01, 0.99, bins)
    ms, mr = ws > 0.5, wr > 0.5
    if ms.sum() < 50 or mr.sum() < 50:
        return out
    for c in range(3):
        sv = np.quantile(src[..., c][ms], qs)
        rv = np.quantile(ref[..., c][mr], qs)
        sv = np.maximum.accumulate(sv + np.arange(bins) * 1e-4)
        d = np.convolve(np.pad(rv - sv, 3, mode="edge"), np.ones(7) / 7, mode="valid")
        out[..., c] = np.interp(src[..., c], sv, d)
    return out


def regions(shape, pts: np.ndarray, fs: float):
    """Soft region maps (skin, features, border) and the skin-statistics weight, in ROI coordinates."""
    from .geometry import feature_mask, poly_mask
    from .landmarks import topology

    t = topology()
    oval = poly_mask(shape, pts[t["oval"]])
    feat = feature_mask(shape, pts, (0, 0), fs * 0.02)
    feat_s = cv2.GaussianBlur(feat, (0, 0), max(1.0, fs * 0.015))
    dist = cv2.distanceTransform((oval > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    border = 1 - np.clip(dist / max(1.0, fs * 0.1), 0, 1)
    border = np.maximum(border, 1 - oval)
    r_feat = feat_s * (1 - border)
    r_border = border
    r_skin = np.clip(1 - r_feat - r_border, 0, 1)
    k = max(1, int(fs * 0.04))
    stat_w = cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) * (1 - feature_mask(shape, pts, (0, 0), fs * 0.04))
    return r_skin, r_feat, r_border, stat_w.astype(np.float32)


def features(warped: np.ndarray, target: np.ndarray, pts: np.ndarray, fs: float, skin: np.ndarray | None = None):
    """Per-pixel feature stack (H, W, F) and the source Lab image. ``pts``: landmarks in ROI coordinates."""
    shape = warped.shape[:2]
    lw, lt = _lab(warped), _lab(target)
    r_skin, r_feat, r_border, stat_w = regions(shape, pts, fs)
    if skin is not None and skin.shape == shape:
        # soft skin likelihood (both photos): hair, beard, glasses and shadows on the border do not bias the statistics
        sw2 = stat_w * np.clip(skin.astype(np.float32), 0, 1)
        if float(sw2.sum()) > 200:
            stat_w = sw2
    if float(stat_w.sum()) < 50:
        stat_w = np.ones(shape, np.float32)
    F = []
    regs = (r_skin, r_feat, r_border)
    for r in regs:  # region biases
        F.append(r)
    mus = []
    for s in SCALES:
        mw, mt = _nblur(lw, stat_w, fs * s), _nblur(lt, stat_w, fs * s)
        mus.append((mw, mt))
        d = mt - mw
        for r in regs:
            for c in range(3):
                F.append(d[..., c] * r)
    gw, sw = _wstats(lw, stat_w)
    gt, st = _wstats(lt, stat_w)
    dg = gt - gw
    for r in regs:
        for c in range(3):
            F.append(np.full(shape, dg[c], np.float32) * r)
    ratio = np.clip(st / np.maximum(sw, 1e-3), 0.5, 2.0)
    for c in range(3):  # Reinhard contrast term
        F.append((lw[..., c] - gw[c]) * (ratio[c] - 1))
    hd = _hist_delta(lw, stat_w, lt, stat_w)
    for c in range(3):
        F.append(hd[..., c])
    for mw, mt in mus[:2]:  # multiplicative lighting: detail and chroma scale with the shading change
        rho = (mt[..., 0] - mw[..., 0]) / (mw[..., 0] + 16.0)
        F.append((lw[..., 0] - mw[..., 0]) * rho)
        F.append(lw[..., 1] * rho)
        F.append(lw[..., 2] * rho)
    mw, _ = mus[1]
    dl = lw[..., 0] - mw[..., 0]
    vw = np.sqrt(_nblur(dl * dl, stat_w, fs * SCALES[1]) + 1e-3)
    dlt = lt[..., 0] - mus[1][1][..., 0]
    vt = np.sqrt(_nblur(dlt * dlt, stat_w, fs * SCALES[1]) + 1e-3)
    F.append(dl * (np.clip(vt / vw, 0.5, 2.0) - 1))
    return np.stack(F, -1).astype(np.float32), lw


N_FEATURES = 3 + len(SCALES) * 9 + 9 + 3 + 3 + 6 + 1


@lru_cache(maxsize=1)
def model() -> dict | None:
    if not DATA.is_file():
        return None
    try:
        d = np.load(DATA, allow_pickle=False)
        m = {k: d[k] for k in d.files}
        if int(m.get("version", 0)) != VERSION or m["weights"].shape != (N_FEATURES, 3):
            log.warning("learned colour model has an unexpected format; ignoring it")
            return None
        return m
    except Exception:  # noqa: BLE001
        log.exception("could not load %s", DATA)
        return None


def available() -> bool:
    return model() is not None


def predict_delta(feats: np.ndarray, m: dict) -> np.ndarray:
    X = (feats - m["mean"]) / m["std"]
    return (X.reshape(-1, X.shape[-1]) @ m["weights"] + m["bias"]).reshape(feats.shape[:2] + (3,))


def color_transfer(
    warped: np.ndarray,
    target: np.ndarray,
    dst_pts: np.ndarray,
    roi_off=(0, 0),
    fs: float | None = None,
    skin: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Colour/lighting-correct the warped source face so it is consistent with ``target``.

    ``warped`` and ``target`` are aligned float32 RGB crops (the target ROI);
    ``dst_pts`` are the destination landmarks in full-image coordinates and
    ``roi_off`` = (x0, y0) of the crop. ``skin`` (optional) restricts the
    statistics to pixels that are skin in both images. Returns the corrected
    image (same shape); returns ``warped`` unchanged when the model is missing.
    """
    m = model()
    if m is None or strength <= 0:
        return warped
    pts = np.asarray(dst_pts, np.float64)[:468, :2] - np.asarray(roi_off, np.float64)
    if fs is None:
        from .landmarks import topology

        fs = float(np.sqrt(abs(cv2.contourArea(pts[topology()["oval"]].astype(np.float32)))))
    feats, lw = features(warped.astype(np.float32), target.astype(np.float32), pts, float(fs), skin)
    delta = predict_delta(feats, m) * float(strength)
    clip = m.get("clip")
    if clip is not None:
        delta = np.clip(delta, -clip, clip)
    lab = lw + delta.astype(np.float32)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
