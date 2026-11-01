"""Edge-aware and masked filtering primitives."""

from __future__ import annotations

import cv2
import numpy as np


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def normalized_blur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: blur ``x`` using only pixels where ``w`` > 0.

    Used to estimate local statistics (colour, brightness, background colour
    behind an object edge) from a restricted set of pixels.
    """
    w = w.astype(np.float32)
    if x.ndim == 3:
        num = gaussian(x * w[..., None], sigma)
        den = gaussian(w, sigma)[..., None]
    else:
        num = gaussian(x * w, sigma)
        den = gaussian(w, sigma)
    return num / np.maximum(den, 1e-6)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving guided filter (He et al. 2010).

    Uses OpenCV-contrib's implementation when available, otherwise a numpy
    implementation with a grayscale guide.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    radius = max(1, int(radius))
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.guidedFilter(guide, src, radius, eps)
    g = guide if guide.ndim == 2 else cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda a: cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    mi, mp = box(g), box(src)
    cov = box(g * src) - mi * mp
    var = box(g * g) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box(a) * g + box(b)


def bbox_of(mask: np.ndarray, pad: int = 0, shape: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) of mask > 0 expanded by ``pad`` and clipped."""
    ys, xs = np.nonzero(mask > 0)
    h, w = shape or mask.shape[:2]
    if len(ys) == 0:
        return 0, h, 0, w
    return max(0, ys.min() - pad), min(h, ys.max() + 1 + pad), max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, k)


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask, k)


def signed_distance(binary: np.ndarray) -> np.ndarray:
    """Signed Euclidean distance (positive inside)."""
    b = (binary > 0).astype(np.uint8)
    inside = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
    return (inside - outside).astype(np.float32)
