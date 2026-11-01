"""Snapshot of face/masking.py before the face-parsing masks (for A/B comparisons in sheets.py)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np
from lumen_engine.errors import IncompatibleInputsError, ProcessingError
from lumen_engine.imaging import analysis, blend, color
from lumen_engine.imaging.filters import bbox_of, erode, gaussian, guided_filter, normalized_blur
from lumen_engine.face.geometry import (
    dense_warp_maps,
    destination_points,
    feature_mask,
    folded_triangle_fraction,
    poly_mask,
)
from lumen_engine.face.landmarks import Face, topology
from lumen_engine.face.settings import FaceSwapSettings  # noqa: F401


def _skin_visibility(img: np.ndarray, face: Face) -> float:
    """Fraction of the face oval (excluding features) whose colour matches the cheek skin model."""
    t = topology()
    x0, y0, x1, y1 = face.bbox
    roi = (int(max(0, y0)), int(min(img.shape[0], y1 + 1)), int(max(0, x0)), int(min(img.shape[1], x1 + 1)))
    sub = img[roi[0] : roi[1], roi[2] : roi[3]]
    if sub.size == 0:
        return 1.0
    off = (roi[2], roi[0])
    shape = sub.shape[:2]
    oval = poly_mask(shape, face.xy[t["oval"]], off)
    feat = feature_mask(shape, face.xy, off, face.size * 0.03)
    region = oval * (1 - feat)
    cheeks = np.zeros(shape, np.float32)
    for idx in (50, 280, 4, 205, 425):  # cheeks, nose, mid-cheeks: rarely occluded
        cx, cy = face.xy[idx] - np.asarray(off)
        cv2.circle(cheeks, (int(cx), int(cy)), max(2, int(face.size * 0.06)), 1.0, -1)
    p = _skin_probability(sub, cheeks * region)
    return float((p * region).sum() / max(region.sum(), 1))

def _skin_probability(img: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Per-pixel likelihood of matching the skin colour model learned from ``seed``."""
    lab = color.rgb_to_lab(img)
    w = seed > 0.5
    if w.sum() < 30:
        return np.ones(img.shape[:2], np.float32)
    X = lab[w][:, :]
    mu = X.mean(0)
    # robust covariance (trim outliers such as stubble or specular highlights)
    d0 = X - np.median(X, 0)
    keep = (np.abs(d0) < 3 * (np.median(np.abs(d0), 0) * 1.4826 + 1e-3)).all(1)
    if keep.sum() > 20:
        X = X[keep]
    mu = X.mean(0)
    cov = np.cov(X.T) + np.diag([16.0, 2.0, 2.0])
    inv = np.linalg.inv(cov)
    d = lab - mu
    m2 = np.einsum("...i,ij,...j->...", d, inv, d)
    # chi-square(3): 95 % of skin pixels have m2 < 7.8
    return np.exp(-0.5 * np.maximum(m2 - 7.8, 0) / 2.0).astype(np.float32)

def _face_masks(target: np.ndarray, warped: np.ndarray, dst_face: Face, dst_pts: np.ndarray, src_support: np.ndarray, roi, s: FaceSwapSettings, src_face: Face | None = None):
    t = topology()
    y0, y1, x0, x1 = roi
    shape = (y1 - y0, x1 - x0)
    off = (x0, y0)
    fs = dst_face.size
    # face oval, optionally scaled around the face centre
    oval = dst_pts[t["oval"]]
    c = dst_pts[[1, 4, 6, 168]].mean(0)
    oval = c + (oval - c) * s.mask_extent
    hull = poly_mask(shape, oval, off)
    hull *= (src_support > 0.5).astype(np.float32)
    feat = feature_mask(shape, dst_pts, off, fs * 0.03)
    core_skin = erode(hull, int(fs * 0.12)) * (1 - feat)

    # skin likelihood in the outer band (hair, glasses rims, hands, beards stop the mask)
    p_t = _skin_probability(target, core_skin)
    p_s = _skin_probability(warped, core_skin)
    p = np.minimum(p_t, p_s)
    p = gaussian(p, max(1.0, fs * 0.01))
    band_depth = fs * 0.14
    dist_in = cv2.distanceTransform((hull > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    inner_w = np.clip(dist_in / band_depth, 0, 1)
    # Above the eyebrows hair/bangs/hats can reach far into the face: apply the
    # skin test across the whole forehead, not only in the outer band.
    brow = dst_pts[[105, 66, 107, 336, 296, 334]]
    ex = dst_pts[263] - dst_pts[33]
    ex = ex / (np.linalg.norm(ex) + 1e-9)
    ey = np.array([-ex[1], ex[0]])
    if ey @ (dst_pts[152] - dst_pts[10]) < 0:
        ey = -ey  # ey points from forehead to chin
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    v = (xx + off[0] - brow[:, 0].mean()) * ey[0] + (yy + off[1] - brow[:, 1].mean()) * ey[1]
    forehead = np.clip(-(v + fs * 0.03) / (fs * 0.05), 0, 1)
    inner_w = inner_w * (1 - forehead)
    mask = hull * (inner_w + (1 - inner_w) * p)
    # never let the skin test punch holes into the inner face (below the brows)
    mask = np.maximum(mask, erode(hull, int(band_depth)) * (1 - forehead))

    mouth_kept = False
    open_t = dst_face.mouth_openness()
    open_s = src_face.mouth_openness() if src_face is not None else open_t
    if s.keep_target_mouth and abs(open_s - open_t) > 0.012:
        # the expressions differ (smiling with teeth vs closed mouth): warping one mouth
        # into the other stretches lips and teeth unnaturally - keep the target's mouth
        lips = poly_mask(shape, dst_face.xy[t["outer_lip"]], off)
        lips = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(fs * 0.05)) | 1,) * 2))
        mask *= 1 - gaussian(lips, max(1.0, fs * 0.015))
        mouth_kept = "expressions differ"
    elif s.keep_target_mouth and open_t > 0.003:
        inner = poly_mask(shape, dst_face.xy[t["inner_lip"]], off)
        inner = cv2.dilate(inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask *= 1 - gaussian(inner, max(1.0, fs * 0.006))
        mouth_kept = True

    # keep the transition inside the face region, then edge-aware feathering
    er = s.edge_refinement
    shrink = int(fs * (0.01 + 0.02 * er))
    mask = erode(mask, shrink)
    radius = max(2, int(fs * (0.015 + 0.05 * er)))
    soft = guided_filter(target.astype(np.float32), mask.astype(np.float32), radius, 2e-3)
    soft = gaussian(np.clip(soft, 0, 1), max(0.8, fs * (0.006 + 0.02 * er)))
    soft = np.clip(soft, 0, 1) * (hull > 0)  # do not spill outside the face contour
    soft = gaussian(soft, 0.8)
    return np.clip(soft, 0, 1).astype(np.float32), core_skin, feat, mouth_kept, p_t * p_s
