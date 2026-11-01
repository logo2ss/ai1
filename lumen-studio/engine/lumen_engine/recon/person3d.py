"""Closed, textured silhouette volume. All inference is local ONNX; depth is optional.

Hidden surfaces are inferred from the outline, not recovered anatomy. The two
height fields share an explicit rim, so oblique views never expose a cardboard
cutout. No background pixels are used to texture the rim.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..depth.estimator import estimate_depth
from ..errors import InvalidInputError
from ..imaging.filters import normalized_blur
from ..imaging.io import encode_image, resize_max
from ..imaging.matting import estimate_foreground, refine_alpha
from ..models import manager


@dataclass
class Person3DSettings:
    detail: int = 220
    thickness: float = 0.65
    model: str = "depth-anything-v2-small"

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        if not isinstance(d, dict):
            raise InvalidInputError("Person settings must be an object.")
        try:
            detail, thickness = float(d.get("detail", 220)), float(d.get("thickness", 0.65))
            if not np.isfinite([detail, thickness]).all():
                raise ValueError
            model = d.get("model", "depth-anything-v2-small")
            if model not in ("depth-anything-v2-small", "midas-v21-small"):
                raise ValueError
            return cls(int(np.clip(detail, 80, 360)), float(np.clip(thickness, 0.2, 1.0)), model)
        except (TypeError, ValueError):
            raise InvalidInputError("Invalid Person 3D settings.") from None


def build_person3d(rgb, mask=None, settings=None, progress=None):
    s = settings or Person3DSettings()
    prog = progress or (lambda *_: None)
    work, _ = resize_max(rgb, 1200)
    warnings = ["Hidden sides are estimated from one photo; use a near-frontal, unobstructed person for best results."]
    prog(0.02, "Separating the person")
    if mask is None:
        from ..imaging.u2net import person_mask, predict
        if manager.is_installed("segment-human"):
            mask = person_mask(work)
        elif manager.is_installed("segment-silueta"):
            mask = predict(work, "segment-silueta")
            warnings.append("Human segmentation missing: using general object segmentation.")
        else:
            manager.require("segment-human")
    mask = cv2.resize(mask.astype(np.float32), (work.shape[1], work.shape[0]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0.5).astype(np.uint8))
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 32:
        raise InvalidInputError("No person was found. Use a clear photo or a transparent person PNG.")
    keep = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # Keep actual gaps between the arms/legs, including enclosed holes.
    alpha = refine_alpha(work, mask * keep, radius=3)
    foreground = estimate_foreground(work, alpha)
    ys, xs = np.nonzero(keep)
    y0, y1 = max(0, ys.min()-3), min(work.shape[0], ys.max()+4)
    x0, x1 = max(0, xs.min()-3), min(work.shape[1], xs.max()+4)
    crop = np.s_[y0:y1, x0:x1]
    a = alpha[crop]
    tex = foreground[crop].copy()
    # Extend confidently foreground colour across uncertain pixels and the rim.
    _, inds = distance_transform_edt(a < 0.9, return_indices=True)
    nearest = tex[tuple(inds)]
    tex = np.where((a > 0.98)[..., None], tex, nearest)
    ph, pw = a.shape
    rows = min(600, max(24, round(s.detail * ph / pw)))
    cols = max(24, round(rows * pw / ph))
    ag = cv2.resize(a, (cols, rows), interpolation=cv2.INTER_AREA)
    inside = ag > 0.5
    aspect = pw / ph
    yy, xx = np.mgrid[:rows, :cols]
    X = (xx / (cols-1) - 0.5) * aspect
    Y = 0.5 - yy / (rows-1)
    # Circular horizontal cross-sections, tapered by distance to the full outline
    # at heads, fingers, feet and the ends of clothing.
    radius = np.zeros_like(ag)
    horizontal = np.zeros_like(ag)
    for y in range(rows):
        edges = np.diff(np.r_[False, inside[y], False].astype(np.int8))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            r = (hi-lo) / 2
            radius[y, lo:hi] = r
            horizontal[y, lo:hi] = np.sqrt(np.maximum(0, r*r - (np.arange(lo, hi)-(lo+hi-1)/2)**2))
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    cap = np.sqrt(np.maximum(0, dist * (2*radius-dist)))
    half = np.minimum(horizontal, cap) * (s.thickness * aspect / (cols-1))
    half = cv2.GaussianBlur(half, (0, 0), 1.2)
    half *= np.clip((dist-1)/2, 0, 1)
    mid = np.zeros_like(ag)
    detail = np.zeros_like(ag)
    prog(0.15, "Estimating person depth")
    if manager.is_installed(s.model):
        depth = estimate_depth(work, s.model, detail=False)
        d = cv2.resize(depth[crop], (cols, rows), interpolation=cv2.INTER_AREA)
        lo, hi = np.percentile(d[inside], [3, 97])
        rel = np.clip((d-lo) / max(hi-lo, 1e-5), 0, 1)
        low = normalized_blur(rel, inside.astype(np.float32), max(2, cols/25))
        mid = (low-0.5) * aspect * 0.1
        detail = np.clip(rel-low, -0.25, 0.25) * aspect * 0.12
    else:
        warnings.append("Depth model missing: using rounded silhouette volume without photo depth detail.")
    # Keep high frequency detail off the silhouette and off the back.
    detail *= np.clip(dist/4, 0, 1)
    front_z = mid+half+detail
    # Optional local 478-point ONNX face relief: retain the actual nose,
    # cheeks and eye sockets instead of rounding the head like clothing.
    if manager.is_installed("face-mesh-onnx") and manager.is_installed("face-detector-yunet"):
        from ..face import fallback
        from ..face.appearance import MeshRaster
        from ..face.landmarks import topology
        try:
            people = fallback.detect_faces(work)
            if people:
                face = max(people, key=lambda f: f.size)
                points = face.xy.copy()
                points -= (x0, y0)
                points *= ((cols-1)/pw, (rows-1)/ph)
                raster = MeshRaster(points, (rows, cols))
                face_depth = raster.interp(face.points[:468, 2:3].astype(np.float32))[..., 0]
                sel = raster.inside & inside
                if sel.sum() > 16:
                    ref = float(np.median(front_z[sel]))
                    face_z = ref + (float(np.median(face.points[topology()["oval"], 2]))-face_depth)/ph
                    face_z = np.maximum(mid + 0.005, face_z)
                    face_z = cv2.GaussianBlur(face_z.astype(np.float32), (0, 0), 1.2)
                    fade = 0.25 * np.clip(cv2.distanceTransform(sel.astype(np.uint8), cv2.DIST_L2, 5)/3, 0, 1)
                    front_z = front_z*(1-fade)+face_z*fade
        except Exception:  # Optional face relief must not prevent silhouette/depth reconstruction.
            warnings.append("Face relief unavailable: using photo depth for the head.")
    front = np.stack([X, Y, front_z], -1).astype(np.float32).reshape(-1, 3)
    back = np.stack([X, Y, mid-half], -1).astype(np.float32).reshape(-1, 3)
    grid = np.arange(rows*cols).reshape(rows, cols)
    valid = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
    tl = grid[:-1, :-1][valid]
    tr, bl, br = tl+1, tl+cols, tl+cols+1
    if not len(tl):
        raise InvalidInputError("The person is too small to reconstruct.")
    ft = np.concatenate([np.stack([tl, bl, tr], 1), np.stack([tr, bl, br], 1)])
    offset = rows*cols
    # Explicit walls on every boundary (outer outline AND holes).
    edges = np.sort(np.concatenate([ft[:, [0, 1]], ft[:, [1, 2]], ft[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    e0, e1 = unique[counts == 1].T
    walls = np.concatenate([np.stack([e0, e1, e0+offset], 1), np.stack([e1, e1+offset, e0+offset], 1)])
    faces = np.concatenate([ft, ft[:, ::-1]+offset, walls]).astype(np.int32)
    vertices = np.concatenate([front, back])
    uv = np.stack([xx/(cols-1), yy/(rows-1)], -1).astype(np.float32).reshape(-1, 2)
    uv = np.concatenate([uv, uv])
    used, inverse = np.unique(faces, return_inverse=True)
    vertices, uv, faces = vertices[used], uv[used], inverse.reshape(-1, 3).astype(np.int32)
    buf = io.BytesIO()
    np.savez_compressed(buf, vertices=vertices, uv=uv, faces=faces)
    prog(1, "Person ready")
    return {"person": buf.getvalue(), "texture": encode_image(tex, "png"),
            "matte": encode_image(np.dstack([a]*3), "png"),
            "header": {"version": 1, "vertices": len(vertices), "triangles": len(faces), "settings": asdict(s)},
            "warnings": warnings}
