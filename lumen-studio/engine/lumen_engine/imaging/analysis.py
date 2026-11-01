"""Image measurements used to make inserted content statistically consistent
with its destination: noise level, sharpness/blur, local brightness/contrast
and low-frequency illumination."""

from __future__ import annotations

import cv2
import numpy as np

from .color import luminance
from .filters import gaussian

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise_sigma(rgb: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Estimate additive noise std (in [0,1] units) - Immerkær (1996).

    Strong edges are excluded so texture is not mistaken for noise.
    """
    g = luminance(rgb) if rgb.ndim == 3 else rgb
    resp = np.abs(cv2.filter2D(g, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    w = np.ones_like(g) if weight is None else (weight > 0.5).astype(np.float32)
    w[[0, -1], :] = 0
    w[:, [0, -1]] = 0
    if w.sum() < 100:
        return 0.0
    thr = np.quantile(grad[w > 0], 0.6)  # flattest 60 % of the region
    flat = (grad <= thr) & (w > 0)
    clip = (g > 0.01) & (g < 0.99)
    flat &= clip
    if flat.sum() < 50:
        return 0.0
    return float(np.sqrt(np.pi / 2) / 6.0 * resp[flat].mean())


def sharpness(rgb: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Contrast-invariant sharpness: fine-scale / coarse-scale gradient energy."""
    g = luminance(rgb) if rgb.ndim == 3 else rgb

    def gm(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    fine = gm(gaussian(g, 0.6))
    coarse = gm(gaussian(g, 2.5))
    w = np.ones_like(g) if weight is None else weight
    return float((fine * w).sum() / ((coarse * w).sum() + 1e-6))


def match_sharpness(src: np.ndarray, ref: np.ndarray, weight: np.ndarray, max_sigma: float = 4.0) -> tuple[np.ndarray, dict]:
    """Blur or sharpen ``src`` so its sharpness matches ``ref`` inside ``weight``.

    Both images must be aligned in the same coordinate frame (e.g. the warped
    source face and the target face). Returns (adjusted, report).
    """
    s0, r0 = sharpness(src, weight), sharpness(ref, weight)
    report = {"source_sharpness": round(s0, 3), "target_sharpness": round(r0, 3), "action": "none", "amount": 0.0}
    if abs(s0 - r0) / max(r0, 1e-6) < 0.04:
        return src, report
    if s0 > r0:  # source too sharp -> gaussian blur, bisection on sigma
        lo, hi = 0.0, max_sigma
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if sharpness(gaussian(src, mid), weight) > r0:
                lo = mid
            else:
                hi = mid
        sig = 0.5 * (lo + hi)
        report.update(action="blur", amount=round(sig, 2))
        return gaussian(src, sig), report
    # source too soft -> unsharp mask (bounded, to avoid halos)
    lo, hi = 0.0, 1.5
    base = gaussian(src, 1.2)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if sharpness(np.clip(src + (src - base) * mid, 0, 1), weight) < r0:
            lo = mid
        else:
            hi = mid
    amt = 0.5 * (lo + hi)
    report.update(action="sharpen", amount=round(amt, 2))
    return np.clip(src + (src - base) * amt, 0, 1), report


def synth_grain(shape: tuple[int, int], sigma: float, rng: np.random.Generator, correlation: float = 0.55) -> np.ndarray:
    """Camera-like noise: spatially correlated luminance grain + weaker chroma noise."""
    h, w = shape
    if sigma <= 0:
        return np.zeros((h, w, 3), np.float32)
    luma = gaussian(rng.standard_normal((h, w)).astype(np.float32), correlation)
    luma *= sigma / (luma.std() + 1e-8)
    chroma = gaussian(rng.standard_normal((h, w, 2)).astype(np.float32), correlation * 2)
    chroma *= 0.35 * sigma / (chroma.std() + 1e-8)
    # luma along (1,1,1); chroma along two orthogonal axes
    n = luma[..., None] * np.array([1, 1, 1], np.float32)
    n += chroma[..., :1] * np.array([0.7, -0.35, -0.35], np.float32)
    n += chroma[..., 1:] * np.array([0.0, 0.6, -0.6], np.float32)
    return n


def match_noise(src: np.ndarray, ref: np.ndarray, src_w: np.ndarray, ref_w: np.ndarray, seed: int = 7) -> tuple[np.ndarray, dict]:
    """Add grain (or gently denoise) so the noise level of ``src`` matches ``ref``."""
    ns, nr = noise_sigma(src, src_w), noise_sigma(ref, ref_w)
    report = {"source_noise": round(ns * 255, 2), "target_noise": round(nr * 255, 2), "action": "none"}
    if nr > ns * 1.1 and nr > 0.6 / 255:
        add = np.sqrt(max(nr * nr - ns * ns, 0.0))
        rng = np.random.default_rng(seed)
        grain = synth_grain(src.shape[:2], 1.0, rng)
        # calibrate with the same estimator used on the target (grain is spatially correlated)
        measured = noise_sigma(np.clip(0.5 + grain * 0.02, 0, 1)) / 0.02
        out = np.clip(src + grain * (add / max(measured, 1e-3)), 0, 1)
        report.update(action="add_grain", amount=round(add * 255, 2))
        return out, report
    if ns > nr * 1.5 and ns > 2.0 / 255:
        u8 = (np.clip(src, 0, 1) * 255).astype(np.uint8)
        h = float(np.clip((ns - nr) * 255 * 1.2, 2, 15))
        den = cv2.fastNlMeansDenoisingColored(u8, None, h, h * 0.6, 5, 15).astype(np.float32) / 255
        report.update(action="denoise", amount=round(h, 2))
        return den, report
    return src, report


def fit_luminance_plane(log_l: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Least-squares plane a*x + b*y + c fitted to (log) luminance where weight>0.

    Coordinates are normalised to [-1, 1]. Returns (a, b, c).
    """
    h, w = log_l.shape
    ys, xs = np.nonzero(weight > 0.05)
    if len(ys) < 30:
        return np.zeros(3, np.float32)
    ww = weight[ys, xs]
    X = np.stack([xs / max(w - 1, 1) * 2 - 1, ys / max(h - 1, 1) * 2 - 1, np.ones_like(xs, dtype=np.float64)], 1)
    A = X * ww[:, None]
    coef, *_ = np.linalg.lstsq(A, log_l[ys, xs] * ww, rcond=None)
    return coef.astype(np.float32)


def plane_image(coef: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return coef[0] * (xx / max(w - 1, 1) * 2 - 1) + coef[1] * (yy / max(h - 1, 1) * 2 - 1) + coef[2]
