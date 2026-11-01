"""Body-proportion editing with a pose-fitted deformation lattice.

1. **Detection** - MediaPipe Pose Landmarker gives 33 body keypoints and a
   person segmentation mask. A lattice is placed over the person whose rows
   sit at anatomical heights (head top, shoulders, waist, hips, knees, feet),
   so sliders such as "leg length" or "waist" move the right rows.
2. **Deformation** - control-point displacements are interpolated with a
   bicubic spline over the (non-uniform) lattice, padded with a ring of
   zero-displacement points so the deformation fades out smoothly. The
   backward map is obtained by fixed-point inversion of the forward field.
3. **Naturalisation** ("person layer" mode, default)
   * only the person is warped; the background keeps straight lines,
   * a narrow band of surroundings (contact shadows, clothing spill, the
     ground under the feet) follows the body with a smooth falloff,
   * background revealed by a slimmer/shorter body is synthesised by
     PatchMatch completion copying only background texture,
   * the person is composited with an edge-aware matte, estimated
     foreground colours (no old-background halo) and light wrap,
   * stretched skin/clothing gets its original sensor grain back.
   The alternative "liquify" mode warps the whole image smoothly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..errors import NoFaceFoundError
from ..imaging import analysis, blend, matting
from ..imaging.filters import dilate, erode, gaussian, normalized_blur
from ..imaging.inpaint import scaled_patchmatch
from ..models import manager

ROW_NAMES = ["top", "shoulders", "waist", "hips", "knees", "feet"]
N_COLS = 5

_lock = threading.Lock()
_pose = None


def _landmarker():
    global _pose
    if _pose is None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(manager.require("pose-landmarker")), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=0.4,
            output_segmentation_masks=True,
        )
        _pose = vision.PoseLandmarker.create_from_options(opts)
    return _pose


@dataclass
class BodySettings:
    mode: str = "person"  # person (background stays straight) | liquify (whole image)
    surroundings: float = 0.35  # how much nearby surroundings follow the body (0..1)
    remove_revealed: bool = True  # synthesise background uncovered by the edit
    edge_softness: float = 0.4
    match_grain: bool = True

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, type(getattr(s, k))(v))
        return s


def detect_body(rgb: np.ndarray) -> dict:
    """Find the main person; return mask, keypoints and a default lattice."""
    import mediapipe as mp

    H, W = rgb.shape[:2]
    s = min(1.0, 1280 / max(H, W))
    small = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    u8 = np.ascontiguousarray((np.clip(small, 0, 1) * 255).astype(np.uint8))
    with _lock:
        res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=u8))
    if not res.pose_landmarks:
        raise NoFaceFoundError("No person could be found in this photo.", hint="The body editor needs a clearly visible person (at least head and shoulders).")
    best, best_area, best_mask = 0, -1, None
    for i in range(len(res.pose_landmarks)):
        m = np.asarray(res.segmentation_masks[i].numpy_view(), np.float32).reshape(small.shape[:2])
        if m.sum() > best_area:
            best, best_area, best_mask = i, m.sum(), m
    mask = cv2.resize(best_mask, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = matting.refine_alpha(rgb, (mask > 0.5).astype(np.float32), radius=max(3, int(max(H, W) / 300)), eps=1e-3)
    kp = np.array([[p.x * W, p.y * H, getattr(p, "visibility", 1.0) or 0.0] for p in res.pose_landmarks[best]], np.float32)
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) < 50:
        raise NoFaceFoundError("The detected person is too small to edit.")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ph = y1 - y0
    sh = kp[[11, 12], 1].mean()
    hip = kp[[23, 24], 1].mean()
    knee = kp[[25, 26], 1].mean()
    feet = max(y1, kp[[27, 28, 29, 30, 31, 32], 1].max() if (kp[27:33, 2] > 0.3).any() else y1)
    top = y0 - 0.06 * ph
    bottom = feet + 0.04 * ph
    rows = np.array([top, sh, (sh + hip) / 2, hip, knee, bottom], np.float64)
    # keep rows ordered and separated even when parts are outside the frame
    rows[1] = max(rows[1], top + 0.08 * ph)
    for i in range(1, 6):
        rows[i] = max(rows[i], rows[i - 1] + 0.05 * ph)
    mw = (x1 - x0) * 0.18
    cols = np.linspace(x0 - mw, x1 + mw, N_COLS)
    lattice = np.stack(np.meshgrid(cols, rows), -1)  # (rows, cols, 2) as (x, y)
    return {
        "mask": mask,
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "keypoints": kp,
        "lattice": lattice,
        "people": len(res.pose_landmarks),
        "full_body": bool(kp[[27, 28], 2].mean() > 0.5 and kp[[27, 28], 1].max() < H),
    }


def detect_body_fallback(rgb: np.ndarray) -> dict:
    """Body detection without MediaPipe (compatibility mode).

    The person mask comes from U2-Net human segmentation; the lattice rows are
    placed with standard figure proportions (an adult is about 7.5-8 heads
    tall), anchored on the face found by the YuNet detector when possible.
    """
    from ..imaging.u2net import person_mask

    H, W = rgb.shape[:2]
    soft = person_mask(rgb)
    binm = (soft > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm)
    if n < 2:
        raise NoFaceFoundError("No person could be found in this photo.", hint="The body editor needs a clearly visible person.")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = soft * (lab == k)
    ys, xs = np.nonzero(mask > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ph = float(y1 - y0)
    head = None
    try:
        from ..face import fallback

        faces = [f for f in fallback.detect_faces(rgb) if x0 <= (f.bbox[0] + f.bbox[2]) / 2 <= x1]
        if faces:
            f = max(faces, key=lambda f: f.size)
            head = (f.bbox[1], f.bbox[3])
    except Exception:  # noqa: BLE001 - face is only a refinement
        head = None
    full = y1 < H - 3
    if head:
        hh = (head[1] - head[0]) * 1.15  # face box -> head height
        top = head[0] - 0.25 * hh
        sh = head[1] + 0.35 * hh
        hip = sh + 2.4 * hh
        knee = hip + 2.1 * hh
        bottom = (y1 + 0.04 * ph) if full else max(y1 + 0.04 * ph, hip + 4.0 * hh)
    else:
        top, sh, hip, knee, bottom = y0 - 0.06 * ph, y0 + 0.18 * ph, y0 + 0.52 * ph, y0 + 0.75 * ph, y1 + 0.04 * ph
    rows = np.array([top, sh, (sh + hip) / 2, hip, knee, bottom], np.float64)
    for i in range(1, 6):
        rows[i] = max(rows[i], rows[i - 1] + 0.05 * ph)
    mw = (x1 - x0) * 0.18
    cols = np.linspace(x0 - mw, x1 + mw, N_COLS)
    return {
        "mask": mask.astype(np.float32),
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "keypoints": np.zeros((33, 3), np.float32),
        "lattice": np.stack(np.meshgrid(cols, rows), -1),
        "people": int(n - 1),
        "full_body": bool(full),
        "method": "segmentation",
    }


# --------------------------------------------------------------------------- deformation
def _forward_field(src: np.ndarray, dst: np.ndarray, shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Dense forward displacement (H x W x 2) from lattice control points.

    The lattice is padded with a ring of zero-displacement points so the
    deformation decays smoothly outside the boundary.
    """
    H, W = shape
    rows = src[:, 0, 1] * scale
    cols = src[0, :, 0] * scale
    disp = (dst - src) * scale
    ph = (rows[-1] - rows[0]) * 0.35 + 8
    pw = (cols[-1] - cols[0]) * 0.35 + 8
    rows_p = np.concatenate([[rows[0] - ph], rows, [rows[-1] + ph]])
    cols_p = np.concatenate([[cols[0] - pw], cols, [cols[-1] + pw]])
    dp = np.pad(disp, ((1, 1), (1, 1), (0, 0)))
    field = np.zeros((H, W, 2), np.float32)
    y0, y1 = int(max(0, np.floor(rows_p[0]))), int(min(H, np.ceil(rows_p[-1]) + 1))
    x0, x1 = int(max(0, np.floor(cols_p[0]))), int(min(W, np.ceil(cols_p[-1]) + 1))
    if y1 <= y0 or x1 <= x0:
        return field
    yy, xx = np.arange(y0, y1, dtype=np.float64), np.arange(x0, x1, dtype=np.float64)
    for c in range(2):
        spl = RectBivariateSpline(rows_p, cols_p, dp[..., c], kx=3, ky=3)
        field[y0:y1, x0:x1, c] = spl(yy, xx)
    return field


def _backward_map(field: np.ndarray, iters: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Invert x_dst = x_src + D(x_src) by fixed-point iteration."""
    H, W = field.shape[:2]
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = gx - field[..., 0], gy - field[..., 1]
    for _ in range(iters):
        d = cv2.remap(field, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sx, sy = gx - d[..., 0], gy - d[..., 1]
    return sx, sy


def reshape_body(rgb: np.ndarray, mask: np.ndarray, src: np.ndarray, dst: np.ndarray, settings: BodySettings | None = None, preview: bool = False, progress=None) -> tuple[np.ndarray, dict]:
    s = settings or BodySettings()
    prog = progress or (lambda *_: None)
    H, W = rgb.shape[:2]
    report: dict = {"mode": s.mode}
    max_disp = float(np.abs(dst - src).max())
    if max_disp < 0.5:
        return rgb.copy(), dict(report, changed=False)
    prog(0.1, "Computing deformation")
    field = _forward_field(np.asarray(src, np.float64), np.asarray(dst, np.float64), (H, W))
    mx, my = _backward_map(field)
    interp = cv2.INTER_LINEAR if preview else cv2.INTER_CUBIC
    if s.mode == "liquify":
        out = cv2.remap(rgb, mx, my, interp, borderMode=cv2.BORDER_REFLECT)
        report["changed"] = True
        return np.clip(out, 0, 1), report

    # ---- person layer mode
    prog(0.2, "Separating the person")
    alpha = mask.astype(np.float32)
    fg = matting.estimate_foreground(rgb, alpha) if not preview else rgb
    wfg = cv2.remap(fg, mx, my, interp, borderMode=cv2.BORDER_REFLECT)
    walpha = np.clip(cv2.remap(alpha, mx, my, cv2.INTER_LINEAR, borderValue=0), 0, 1)
    # surroundings near the body follow it with a smooth falloff (shadows, contact areas)
    person = (alpha > 0.5).astype(np.uint8)
    ph = max(np.ptp(np.nonzero(person)[0]) if person.any() else H, 1)
    band = max(4.0, 0.03 * ph)
    dist = cv2.distanceTransform(1 - person, cv2.DIST_L2, 5)
    fall = (np.exp(-dist / band) * s.surroundings).astype(np.float32)
    fall[dist > 3 * band] = 0.0  # hard cutoff: far background is never resampled
    bfield = field * fall[..., None]
    bmx, bmy = _backward_map(bfield, iters=3)
    bg = cv2.remap(rgb, bmx, bmy, interp, borderMode=cv2.BORDER_REFLECT)
    # where was the person (in background coordinates)? those pixels are not valid background
    old_person = cv2.remap(alpha, bmx, bmy, cv2.INTER_LINEAR, borderValue=0)
    prog(0.35, "Repairing revealed background")
    hole = (dilate((old_person > 0.15).astype(np.uint8), max(2, int(ph * 0.006))) > 0) & ~(erode((walpha > 0.97).astype(np.uint8), 2) > 0)
    report["revealed_px"] = int(hole.sum())
    if hole.any() and s.remove_revealed:
        if preview:
            hm = hole.astype(np.uint8)
            u8 = (np.clip(bg, 0, 1) * 255).astype(np.uint8)
            bg = cv2.inpaint(u8, hm, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
        else:
            src_ok = ~(dilate(person, max(3, int(ph * 0.02))) > 0)
            bg = scaled_patchmatch(bg.astype(np.float32), hole, source=src_ok, max_extent=520, progress=lambda p: prog(0.35 + 0.45 * p, "Repairing revealed background"))
    prog(0.85, "Integrating the person")
    if s.match_grain and not preview:
        wfg_n, nrep = analysis.match_noise(wfg, rgb, (walpha > 0.9).astype(np.float32), (alpha > 0.9).astype(np.float32))
        wfg = wfg_n
        report["grain"] = nrep
    a = walpha
    if s.edge_softness > 0:
        a = np.clip(gaussian(a, 0.5 + s.edge_softness), 0, 1) * (walpha > 0.01) + walpha * 0  # soften the warped matte slightly
        a = np.maximum(a, (walpha > 0.98).astype(np.float32))
    if not preview:
        # light wrap: background light spills over the new silhouette
        edge = np.clip(a * (1 - gaussian(a, 2.0)) * 2.0, 0, 1) * 0.3
        bgb = normalized_blur(bg, 1 - a, 3.0)
        wfg = wfg * (1 - edge[..., None]) + (1 - (1 - wfg) * (1 - bgb * 0.6)) * edge[..., None]
    out = blend.alpha_composite(wfg, bg, a)
    # exact preservation outside the edited zone
    touched = (alpha > 0.002) | (walpha > 0.002) | (np.abs(bfield).max(-1) > 0.02) | hole
    touched = dilate(touched.astype(np.uint8), 2) > 0
    out = np.where(touched[..., None], out, rgb)
    report["changed"] = True
    report["changed_fraction"] = round(float(touched.mean()), 4)
    report["max_displacement_px"] = round(max_disp, 1)
    prog(1.0, "Done")
    return np.clip(out, 0, 1).astype(np.float32), report
