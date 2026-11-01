"""Geometric face operations: similarity alignment, dense piecewise-affine
warping over the face mesh, region masks and compatibility analysis."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .landmarks import Face, topology


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z-component of the 2D cross product (numpy 2 removed 2D np.cross)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (2x3) mapping src -> dst points."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (s**2).sum(1).mean()
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def destination_points(src: Face, dst: Face, preserve_shape: float) -> np.ndarray:
    """Where each source landmark should land in the target image.

    0 -> conform fully to the target's facial geometry (expression, pose,
    proportions). 1 -> keep the source's proportions (similarity-aligned),
    while the outer face contour always follows the target so the boundary
    stays on the target's jawline/hairline.
    """
    base = _proportion_points(src, dst, preserve_shape)
    if os.environ.get("LUMEN_EYE_ADAPT", "1") == "0":
        return base
    return adapt_eyes(src, dst, base)[0]


def _proportion_points(src: Face, dst: Face, preserve_shape: float) -> np.ndarray:
    t = topology()
    tp = dst.xy[:468].astype(np.float64)
    if preserve_shape <= 0:
        return tp
    M = umeyama(src.xy[t["anchors"]], dst.xy[t["anchors"]])
    sp = apply_affine(M, src.xy[:468])
    w = np.full(468, float(np.clip(preserve_shape, 0, 1)))
    # contour fixed; weight fades in towards the face centre
    oval = t["oval"]
    w[oval] = 0.0
    centre = tp[[1, 4, 6, 168, 197, 195, 5]].mean(0)
    r = np.linalg.norm(tp - centre, axis=1)
    rmax = np.linalg.norm(tp[oval] - centre, axis=1).mean()
    fade = np.clip(1.2 - r / rmax, 0, 1) / 0.6
    w *= np.clip(fade, 0, 1)
    return tp * (1 - w[:, None]) + sp * w[:, None]


EYES = {"right_eye": (33, 133), "left_eye": (362, 263)}  # contour key -> eye corners


def eye_aperture(xy: np.ndarray, key: str) -> float:
    """Eye opening: lid-contour area / eye width^2 (about 0.05-0.1 open, < 0.02 closed)."""
    t = topology()
    a, b = EYES[key]
    width = np.linalg.norm(xy[b] - xy[a]) + 1e-6
    return float(abs(cv2.contourArea(xy[t[key]].astype(np.float32))) / width**2)


def adapt_eyes(src: Face, dst: Face, dst_pts: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep the source's eye shape where the eye opening differs strongly between the faces.

    Warping a (nearly) closed source eye onto an open target eye stretches eyelid and
    lashes into a hazy grey area without iris; squeezing an open eye into a closed
    contour flattens the iris. When the apertures differ by more than ~1.5x, the lid
    contour and its surroundings follow the source's shape instead, aligned to the
    target's eye corners (so the eye stays exactly where the target's eye is), with a
    smooth fall-off back to the target geometry around the eye.
    """
    t = topology()
    out = dst_pts.copy()
    info = {}
    sx, dx = src.xy[:468].astype(np.float64), dst.xy[:468].astype(np.float64)
    for key, (a, b) in EYES.items():
        ap_s, ap_d = eye_aperture(sx, key), eye_aperture(dx, key)
        ratio = (ap_s + 0.004) / (ap_d + 0.004)
        strength = float(np.clip((abs(np.log(ratio)) - np.log(1.5)) / (np.log(2.5) - np.log(1.5)), 0, 1))
        info[key] = {"source_aperture": round(ap_s, 4), "target_aperture": round(ap_d, 4), "keep_source_shape": round(strength, 2)}
        if strength <= 0:
            continue
        M = umeyama(sx[[a, b]], out[[a, b]])  # exact similarity onto the destination eye corners
        src_al = apply_affine(M, sx)
        centre = out[t[key]].mean(0)
        ew = np.linalg.norm(out[b] - out[a]) + 1e-6
        d = np.linalg.norm(out - centre, axis=1) / ew
        wgt = strength * np.clip((1.5 - d) / 0.9, 0, 1)  # 1 inside the eye (d < 0.6), 0 beyond 1.5 eye widths
        wgt[t["oval"]] = 0.0
        out = out * (1 - wgt[:, None]) + src_al * wgt[:, None]
    return out, info


def dense_warp_maps(src_pts: np.ndarray, dst_pts: np.ndarray, shape: tuple[int, int], roi: tuple[int, int, int, int], depth: np.ndarray | None = None):
    """Per-pixel inverse maps (target ROI -> source image) for the face mesh.

    Every target pixel inside the mesh is mapped through the affine transform
    of the triangle that contains it (barycentric interpolation). Pixels
    outside the mesh use the transform of the nearest triangle, giving a
    smooth extrapolation for the feathered transition band. A single
    ``cv2.remap`` with these maps avoids the seams of per-triangle warping.

    The triangles are the faces of MediaPipe's canonical mesh, which also span
    the eye and mouth openings (so iris and teeth follow the lid/lip contours
    continuously instead of being extrapolated from the nearest lid triangle).
    Where the target view folds the mesh (turned heads), the surface that is
    in front must win: back-facing triangles are drawn first, then the
    front-facing ones - far to near when the target landmark ``depth`` (z,
    smaller = closer) is given, otherwise larger before smaller.

    Returns (map_x, map_y, inside_mesh) for the ROI.
    """
    tris = topology()["mesh_triangles"]
    y0, y1, x0, x1 = roi
    h, w = y1 - y0, x1 - x0
    ids = np.zeros((h, w), np.int32)
    affs = np.zeros((len(tris) + 1, 2, 3), np.float64)
    off = np.array([x0, y0], np.float64)
    D = dst_pts[tris]
    signed = cross2(D[:, 1] - D[:, 0], D[:, 2] - D[:, 0])
    areas = np.abs(signed)
    front = np.sign(signed) == np.sign((signed * (areas > np.median(areas))).sum() or 1.0)
    if depth is not None and len(depth) >= 468:
        key = np.asarray(depth, np.float64)[tris].mean(1)  # far (large z) first
    else:
        key = areas
    order = np.concatenate([np.flatnonzero(~front)[np.argsort(-areas[~front])], np.flatnonzero(front)[np.argsort(-key[front])]])
    for k in order:
        a, b, c = tris[k]
        d = dst_pts[[a, b, c]] - off
        s = src_pts[[a, b, c]]
        if abs(cross2(d[1] - d[0], d[2] - d[0])) < 1e-6:
            continue
        affs[k + 1] = cv2.getAffineTransform(d.astype(np.float32), s.astype(np.float32))
        cv2.fillConvexPoly(ids, np.round(d * 16).astype(np.int32), int(k + 1), lineType=cv2.LINE_8, shift=4)
    inside = ids > 0
    if not inside.any():
        raise ValueError("empty mesh")
    # nearest-triangle extrapolation outside the mesh
    if (~inside).any():
        from scipy.ndimage import distance_transform_edt

        _, (iy, ix) = distance_transform_edt(~inside, return_indices=True)
        ids_full = ids[iy, ix]
    else:
        ids_full = ids
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    A = affs[ids_full]
    map_x = A[..., 0, 0] * xx + A[..., 0, 1] * yy + A[..., 0, 2]
    map_y = A[..., 1, 0] * xx + A[..., 1, 1] * yy + A[..., 1, 2]
    return map_x.astype(np.float32), map_y.astype(np.float32), inside


def poly_mask(shape: tuple[int, int], pts: np.ndarray, offset=(0, 0)) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.round((pts - np.asarray(offset)) * 16).astype(np.int32)
    cv2.fillPoly(m, [p], 1, lineType=cv2.LINE_AA, shift=4)
    return m.astype(np.float32)


def feature_mask(shape, xy: np.ndarray, offset, grow: float) -> np.ndarray:
    """Mask of eyes, brows, lips and nostrils (dilated by ``grow`` pixels)."""
    t = topology()
    m = np.zeros(shape, np.uint8)
    for key in ("left_eye", "right_eye", "left_brow", "right_brow", "outer_lip", "nostrils"):
        hull = cv2.convexHull(np.round(xy[t[key]] - np.asarray(offset)).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1)
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * grow + 1), int(2 * grow + 1)))
        m = cv2.dilate(m, k)
    return m.astype(np.float32)


def folded_triangle_fraction(src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    """Fraction of mesh area whose orientation flips between source and target.

    Flipped triangles indicate self-occlusion (parts of the face visible in one
    image are hidden in the other), i.e. an incompatible head pose.
    """
    tris = topology()["triangles"]
    s = src_pts[tris]
    d = dst_pts[tris]
    cs = cross2(s[:, 1] - s[:, 0], s[:, 2] - s[:, 0])
    cd = cross2(d[:, 1] - d[:, 0], d[:, 2] - d[:, 0])
    # global orientation could be mirrored; compare to the dominant sign
    sign = np.sign((cs * cd).sum())
    flipped = (cs * cd * sign) < 0
    return float(np.abs(cd)[flipped].sum() / max(np.abs(cd).sum(), 1e-6))
