"""Detailed 3D model of a person from a single photo.

1. **Silhouette** - person segmentation (MediaPipe when available, otherwise
   U2-Net human segmentation; both run locally).
2. **Relief** - monocular depth (Depth Anything V2) inside the silhouette,
   robustly normalised per person and smoothed edge-aware, so arms in front of
   the torso, the head, folds of clothing etc. stand out.
3. **Volume** - a single depth map is flat-ish across the torso; a
   shape-from-silhouette "inflation" term (distance to the outline, rounded)
   adds the body's natural cross-section. ``roundness`` balances the two.
4. **Mesh** - a dense vertex grid over the person (up to ~200k vertices),
   only where the silhouette is, ready for wireframe / scan-line / point /
   solid rendering in the viewer.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..depth.estimator import estimate_depth
from ..errors import NoFaceFoundError
from ..imaging.filters import gaussian, normalized_blur
from ..imaging.io import encode_image, resize_max


@dataclass
class Body3DSettings:
    detail: int = 280  # vertices across the person's width
    thickness: float = 0.5  # body depth relative to its width
    roundness: float = 0.45  # 0 = depth map only, 1 = silhouette inflation only
    model: str = "depth-anything-v2-small"

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, type(getattr(s, k))(v))
        s.detail = int(np.clip(s.detail, 60, 480))
        return s


def build_body3d(rgb: np.ndarray, mask: np.ndarray, settings: Body3DSettings | None = None, progress=None) -> dict:
    s = settings or Body3DSettings()
    prog = progress or (lambda *_: None)
    work, sc = resize_max(rgb, 1400)
    m = cv2.resize(mask.astype(np.float32), (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA)
    binm = (m > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm)
    if n < 2:
        raise NoFaceFoundError("No person was found in this photo.", hint="Use a photo where a person is clearly visible.")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    binm = (lab == k).astype(np.uint8)
    # Fill only tiny enclosed segmentation specks. Flood-filling every hole
    # turned real gaps between arms/legs into textured sheets of background.
    from scipy import ndimage
    holes, count = ndimage.label(binm == 0)
    areas = np.bincount(holes.ravel())
    boundary = np.unique(np.r_[holes[0], holes[-1], holes[:,0], holes[:,-1]])
    fill = areas <= max(4, int(0.0003*binm.size))
    fill[boundary] = False; fill[0] = False
    binm = np.maximum(binm, fill[holes].astype(np.uint8))
    ys, xs = np.nonzero(binm)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.03 * max(x1 - x0, y1 - y0))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(work.shape[1] - 1, x1 + pad), min(work.shape[0] - 1, y1 + pad)

    prog(0.1, "Estimating depth")
    disp = estimate_depth(work, s.model, detail=True, progress=lambda p: prog(0.1 + 0.6 * p, "Estimating depth"))
    prog(0.75, "Shaping the body")
    crop = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    img_c, m_c, d_c = work[crop], binm[crop].astype(np.float32), disp[crop]
    inside = m_c > 0.5
    lo, hi = np.percentile(d_c[inside], 2), np.percentile(d_c[inside], 98)
    rel = np.clip((d_c - lo) / max(hi - lo, 1e-6), 0, 1)  # 1 = nearest part of the body
    # edge-aware smoothing restricted to the person (background never bleeds in)
    sig = max(1.0, (x1 - x0) / 250)
    rel = normalized_blur(rel, m_c, sig)
    # silhouette inflation: rounded distance to the outline
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    infl = np.sqrt(dist / max(float(dist.max()), 1e-6))
    infl = gaussian(infl, sig)
    z = (1 - s.roundness) * rel + s.roundness * infl
    z = np.where(inside, z, 0.0)
    pw = float(x1 - x0)
    ph = float(y1 - y0)

    # sample the vertex grid
    cols = s.detail
    rows = max(10, int(round(cols * ph / max(pw, 1))))
    if rows > 900:  # keep the vertex count reasonable for very tall crops
        rows = 900
        cols = max(10, int(round(rows * pw / ph)))
    zg = cv2.resize((z * m_c).astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    mg = cv2.resize(m_c, (cols, rows), interpolation=cv2.INTER_AREA)
    zg = np.where(mg > 0.05, zg / np.maximum(mg, 1e-6), 0.0)
    keep = (mg > 0.5).astype(np.uint8)
    prog(0.92, "Packaging")
    header = {
        "version": 1,
        "cols": int(cols),
        "rows": int(rows),
        "aspect": pw / ph,  # model width / height (height = 1 unit)
        "depth_scale": float(s.thickness * pw / ph),  # z in the same units
        "vertices": int(keep.sum()),
        "bbox": [int(x0 / sc), int(y0 / sc), int(x1 / sc), int(y1 / sc)],
        "settings": asdict(s),
    }
    hj = json.dumps(header).encode()
    hj += b" " * ((-len(hj) - 8) % 4)
    buf = io.BytesIO()
    buf.write(b"LBD1")
    buf.write(struct.pack("<I", len(hj)))
    buf.write(hj)
    buf.write(zg.astype("<f4").tobytes())
    buf.write(keep.tobytes())
    tex = np.where(m_c[..., None] > 0.5, img_c, img_c * 0.35)
    depth_vis = cv2.applyColorMap((np.clip(z, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32) / 255
    depth_vis = np.where(inside[..., None], depth_vis, 0.04)
    prog(1.0, "Done")
    return {
        "body": buf.getvalue(),
        "texture": encode_image(np.clip(tex, 0, 1), "jpg", quality=92),
        "depth_preview": encode_image(depth_vis, "jpg", quality=90),
        "header": header,
    }
