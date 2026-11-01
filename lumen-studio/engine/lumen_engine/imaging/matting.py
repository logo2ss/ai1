"""Mask refinement and alpha matting.

User selections are rough. These functions turn them into soft, edge-aware
alpha mattes that follow real object boundaries and estimate the true
foreground colour in semi-transparent edge pixels (so the source background
does not "halo" into the destination).
"""

from __future__ import annotations

import cv2
import numpy as np

from .filters import bbox_of, dilate, erode, gaussian, guided_filter, normalized_blur


def trimap(mask: np.ndarray, band: int) -> np.ndarray:
    """0 = background, 0.5 = unknown, 1 = foreground."""
    m = (mask > 0.5).astype(np.uint8)
    fg = erode(m, band)
    bg = 1 - dilate(m, band)
    t = np.full(m.shape, 0.5, np.float32)
    t[fg > 0] = 1
    t[bg > 0] = 0
    return t


def snap_to_edges(image: np.ndarray, mask: np.ndarray, band: int = 12, iterations: int = 4) -> np.ndarray:
    """Snap a rough selection to object edges with GrabCut inside a band.

    Pixels deep inside the selection stay foreground and pixels far outside
    stay background; only the uncertain band is re-labelled by the
    colour-model graph cut. Works on the mask's bounding box only.
    """
    m = (mask > 0.5).astype(np.uint8)
    if m.sum() < 30:
        return mask.astype(np.float32)
    y0, y1, x0, x1 = bbox_of(m, pad=band * 3 + 8)
    roi = (np.clip(image[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    mm = m[y0:y1, x0:x1]
    gc = np.full(mm.shape, cv2.GC_PR_BGD, np.uint8)
    gc[dilate(mm, band) > 0] = cv2.GC_PR_BGD
    gc[mm > 0] = cv2.GC_PR_FGD
    gc[erode(mm, band) > 0] = cv2.GC_FGD
    gc[dilate(mm, band) == 0] = cv2.GC_BGD
    if not (gc == cv2.GC_BGD).any():
        gc[[0, -1], :] = cv2.GC_BGD
        gc[:, [0, -1]] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask.astype(np.float32)
    res = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    # keep only components that touch the user's core selection (no stray islands)
    n, lab = cv2.connectedComponents(res)
    core = erode(mm, max(1, band // 2))
    keep = np.zeros_like(res)
    for i in range(1, n):
        comp = lab == i
        if (core[comp] > 0).any():
            keep[comp] = 1
    if keep.sum() < 0.3 * mm.sum():  # graph cut collapsed - trust the user
        keep = mm
    out = np.zeros(mask.shape, np.float32)
    out[y0:y1, x0:x1] = keep
    return out


def refine_alpha(image: np.ndarray, mask: np.ndarray, radius: int = 8, eps: float = 1e-4, feather: float = 0.0) -> np.ndarray:
    """Edge-aware soft alpha via a colour-guided filter.

    ``radius`` controls how far the alpha may adapt to image edges; ``feather``
    adds an extra Gaussian softening (pixels) for very soft transitions.
    """
    m = mask.astype(np.float32)
    if m.max() <= 0:
        return m
    pad = int(radius * 3 + feather * 3 + 4)
    y0, y1, x0, x1 = bbox_of(m > 0.01, pad=pad)
    a = guided_filter(image[y0:y1, x0:x1].astype(np.float32), m[y0:y1, x0:x1], radius, eps)
    if feather > 0:
        a = gaussian(a, feather)
    a = np.clip(a, 0, 1)
    # remove faint fog far from the selection
    a[a < 0.02] = 0
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = a
    return out


def estimate_foreground(image: np.ndarray, alpha: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Approximate foreground colours F in I = aF + (1-a)B.

    B is estimated by normalised convolution of confidently-background pixels
    and propagated under the edge, then F is solved per pixel. Interior
    pixels (a≈1) are untouched.
    """
    a = alpha.astype(np.float32)
    edge = (a > 0.02) & (a < 0.98)
    if not edge.any():
        return image.copy()
    y0, y1, x0, x1 = bbox_of(edge, pad=int(sigma * 4))
    I = image[y0:y1, x0:x1]
    A = a[y0:y1, x0:x1]
    bgw = (A < 0.05).astype(np.float32)
    fgw = (A > 0.95).astype(np.float32)
    B = normalized_blur(I, bgw, sigma)
    Fs = normalized_blur(I, fgw, sigma)
    Aa = np.maximum(A, 0.15)[..., None]
    F = (I - (1 - A[..., None]) * B) / Aa
    # where alpha is small the solve is ill-conditioned: lean on the smooth fg estimate
    conf = np.clip((A - 0.15) / 0.5, 0, 1)[..., None]
    F = np.clip(F * conf + Fs * (1 - conf), 0, 1)
    F = np.where((A >= 0.98)[..., None], I, F)
    out = image.copy()
    out[y0:y1, x0:x1] = F
    return out
