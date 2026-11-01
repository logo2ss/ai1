"""Colour science utilities: colour spaces, statistics, colour transfer,
histogram matching and illuminant estimation.

All functions take float32 RGB images in [0, 1] and optional weight maps
(float32 in [0, 1]) that restrict *where* statistics are measured.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------- spaces
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> CIE L*a*b* (L in [0,100])."""
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


# --------------------------------------------------------------------------- stats
def weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel weighted mean and standard deviation."""
    w = w.astype(np.float64)
    sw = w.sum() + EPS
    if x.ndim == 2:
        x = x[..., None]
    mean = (x * w[..., None]).reshape(-1, x.shape[-1]).sum(0) / sw
    var = (((x - mean) ** 2) * w[..., None]).reshape(-1, x.shape[-1]).sum(0) / sw
    return mean.astype(np.float32), np.sqrt(var + EPS).astype(np.float32)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v, w = values[order], weights[order].astype(np.float64)
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.full_like(qs, np.median(values), dtype=np.float32)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, cw, v).astype(np.float32)


# --------------------------------------------------------------------------- transfer
def reinhard_transfer(
    src: np.ndarray,
    src_w: np.ndarray,
    ref: np.ndarray,
    ref_w: np.ndarray,
    strength: float = 1.0,
    std_clamp: tuple[float, float] = (0.6, 1.6),
    channel_strength: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Statistical colour transfer in CIELAB (Reinhard et al. 2001).

    Statistics are computed only where the weights are high (e.g. skin of the
    source face vs. skin of the target face), which avoids the classic failure
    of global transfer where hair or background pollute the statistics.
    The scaling of the standard deviation is clamped so a low-contrast
    reference cannot flatten texture.
    """
    s_lab, r_lab = rgb_to_lab(src), rgb_to_lab(ref)
    ms, ss = weighted_stats(s_lab, src_w)
    mr, sr = weighted_stats(r_lab, ref_w)
    scale = np.clip(sr / np.maximum(ss, EPS), *std_clamp)
    out = (s_lab - ms) * scale + mr
    k = strength * np.asarray(channel_strength, dtype=np.float32)
    out = s_lab + (out - s_lab) * k
    return lab_to_rgb(out)


def histogram_match(
    src: np.ndarray, src_w: np.ndarray, ref: np.ndarray, ref_w: np.ndarray, strength: float = 1.0, bins: int = 256
) -> np.ndarray:
    """Weighted per-channel histogram specification in CIELAB.

    The CDFs are computed with weights so only the relevant areas contribute.
    The mapping is smoothed to avoid banding/posterisation.
    """
    s_lab, r_lab = rgb_to_lab(src), rgb_to_lab(ref)
    out = s_lab.copy()
    qs = np.linspace(0, 1, bins)
    sm, rm = src_w > 0.05, ref_w > 0.05
    if sm.sum() < 50 or rm.sum() < 50:
        return src.copy()
    for c in range(3):
        sv = weighted_quantile(s_lab[..., c][sm], src_w[sm], qs)
        rv = weighted_quantile(r_lab[..., c][rm], ref_w[rm], qs)
        sv = np.maximum.accumulate(sv + np.arange(bins) * 1e-5)
        # smooth the transfer curve (removes jitter from sparse histograms)
        delta = np.convolve(np.pad(rv - sv, 8, mode="edge"), np.ones(17) / 17, mode="valid")
        mapped = s_lab[..., c] + np.interp(s_lab[..., c], sv, delta)
        out[..., c] = s_lab[..., c] + (mapped - s_lab[..., c]) * strength
    return lab_to_rgb(out)


# --------------------------------------------------------------------------- illuminant
def estimate_illuminant(rgb: np.ndarray, weight: np.ndarray | None = None, p: float = 6.0) -> np.ndarray:
    """Shades-of-Grey illuminant estimate (Finlayson & Trezzi 2004) in linear RGB.

    Clipped and very dark pixels are ignored. Returns a unit-norm RGB vector.
    """
    lin = srgb_to_linear(rgb)
    w = np.ones(rgb.shape[:2], np.float32) if weight is None else weight.astype(np.float32)
    valid = (rgb.max(-1) < 0.98) & (rgb.max(-1) > 0.02)
    w = w * valid
    if w.sum() < 50:
        return np.ones(3, np.float32) / np.sqrt(3)
    e = np.power((np.power(lin, p) * w[..., None]).reshape(-1, 3).sum(0) / (w.sum() + EPS), 1.0 / p)
    return (e / (np.linalg.norm(e) + EPS)).astype(np.float32)


def adapt_illuminant(rgb: np.ndarray, src_illum: np.ndarray, dst_illum: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Von Kries chromatic adaptation in linear RGB, luminance preserving."""
    gains = dst_illum / np.maximum(src_illum, EPS)
    gains = gains / (0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2])
    gains = 1.0 + (np.clip(gains, 0.5, 2.0) - 1.0) * strength
    return linear_to_srgb(srgb_to_linear(rgb) * gains[None, None, :])


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.sqrt(((lab1 - lab2) ** 2).sum(-1))
