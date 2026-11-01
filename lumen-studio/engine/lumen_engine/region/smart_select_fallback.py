"""One-click object selection without MediaPipe.

A click is turned into an object mask by running a salient-object segmenter
(U2-Net "silueta", ONNX) on square crops of several sizes centred on the
click. For each crop the connected region under the click is a candidate; the
chosen candidate is the largest one that is confidently segmented and *not cut
off by the crop border* (i.e. the whole object fits in the crop). Edges are
then snapped to the full-resolution photo. When nothing salient lies under the
click, a colour/edge graph cut (GrabCut) seeded around the click is used, so
smart select always works - even without any model installed.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging.filters import guided_filter
from ..models import manager


def _grabcut_click(rgb: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(1.0, 900 / max(h, w))
    small = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    sh, sw = small.shape[:2]
    cx, cy = int(x * s), int(y * s)
    u8 = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    side = int(min(sh, sw) * 0.45)
    rect = (max(0, cx - side // 2), max(0, cy - side // 2), min(side, sw - max(0, cx - side // 2) - 1), min(side, sh - max(0, cy - side // 2) - 1))
    mask = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(u8, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        cv2.circle(mask, (cx, cy), max(3, side // 20), cv2.GC_FGD, -1)
        cv2.grabCut(u8, mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.float32)
    m = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    if lab[min(cy, sh - 1), min(cx, sw - 1)] > 0:
        m = (lab == lab[min(cy, sh - 1), min(cx, sw - 1)]).astype(np.uint8)
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _onnx_click(rgb: np.ndarray, x: int, y: int) -> np.ndarray | None:
    from ..imaging.u2net import predict

    h, w = rgb.shape[:2]
    best, best_score = None, -1.0
    for frac in (0.3, 0.5, 0.75, 1.0):
        side = int(min(h, w) * frac) if frac < 1 else max(h, w)
        x0 = int(np.clip(x - side // 2, 0, max(0, w - side))) if frac < 1 else 0
        y0 = int(np.clip(y - side // 2, 0, max(0, h - side))) if frac < 1 else 0
        x1, y1 = (min(w, x0 + side), min(h, y0 + side)) if frac < 1 else (w, h)
        crop = rgb[y0:y1, x0:x1]
        sal = predict(crop, "segment-silueta")
        binm = (sal > 0.5).astype(np.uint8)
        n, lab = cv2.connectedComponents(binm)
        lx, ly = x - x0, y - y0
        if not (0 <= ly < lab.shape[0] and 0 <= lx < lab.shape[1]):
            continue
        k = lab[ly, lx]
        if k == 0:  # allow a near miss of a few pixels
            r = max(3, side // 60)
            win = lab[max(0, ly - r) : ly + r + 1, max(0, lx - r) : lx + r + 1]
            vals = win[win > 0]
            if not len(vals):
                continue
            k = np.bincount(vals).argmax()
        comp = lab == k
        area = comp.mean()
        if area < 0.002:
            continue
        conf = float(sal[comp].mean())
        touches = frac < 1 and (comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any())
        score = conf * (0.35 if touches else 1.0) + 0.05 * frac
        if score > best_score:
            full = np.zeros((h, w), np.float32)
            full[y0:y1, x0:x1] = np.where(comp, np.maximum(sal, 0.5), sal * 0.0)
            best, best_score = full, score
    return best


def smart_select(rgb: np.ndarray, points: list[tuple[float, float]]) -> tuple[np.ndarray, str]:
    """Return (mask, method) for the object(s) under the clicked points."""
    h, w = rgb.shape[:2]
    acc = np.zeros((h, w), np.float32)
    method = "graphcut"
    for px, py in points:
        x, y = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
        m = None
        if manager.is_installed("segment-silueta"):
            m = _onnx_click(rgb, x, y)
            if m is not None:
                method = "silueta"
        if m is None:
            m = _grabcut_click(rgb, x, y)
        acc = np.maximum(acc, m)
    r = max(2, int(max(h, w) / 400))
    acc = guided_filter(rgb.astype(np.float32), acc, r, 1e-3)
    m = (acc > 0.5).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    keep = np.zeros_like(m)
    for px, py in points:
        k = lab[int(np.clip(py, 0, h - 1)), int(np.clip(px, 0, w - 1))]
        if k > 0:
            keep[lab == k] = 1
    if keep.sum() == 0:
        keep = m
    return keep.astype(np.float32), method
