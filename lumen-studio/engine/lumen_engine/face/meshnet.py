"""MediaPipe-quality 478-point face mesh without the MediaPipe runtime.

MediaPipe's own face-mesh network (Face Mesh V2, the model inside
``face_landmarker.task``) is converted once to ONNX (see :mod:`.tflite_onnx`)
and run with ONNX Runtime, which works on processors without AVX. The
pre-/post-processing replicates MediaPipe's landmark graph:

* a square, rotated region of interest (1.5x the face, rotated so the eyes are
  level) is resampled to 256x256 (values 0..1);
* the network returns 478 points (x, y in crop pixels, z relative depth on the
  same scale) plus a face-presence logit;
* points are mapped back to the image; z is scaled like x (pixels).

The first region comes from the YuNet detector (box + eye keypoints); it is
then re-derived from the predicted landmarks (MediaPipe's tracking rule: the
landmarks' bounding box x1.5, rotation from the eye corners 33 -> 263) and the
network is run again, which converges to the framing the model expects.

Head pose uses a port of MediaPipe's face-geometry pipeline (weighted
Procrustes fit of the canonical face model in a perspective camera), so yaw,
pitch and roll follow the same convention as the MediaPipe path.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("lumen.face.meshnet")

INPUT = 256
ONNX_FILENAME = "face_landmarks_detector.onnx"
_DATA = Path(__file__).parent / "data"
_lock = threading.Lock()
_convert_failed: str | None = None


# --------------------------------------------------------------------------- model file
def onnx_path() -> Path:
    from ..config import models_dir

    return models_dir() / ONNX_FILENAME


def ensure_model() -> Path | None:
    """Path of the ONNX face-mesh model, converting it from face_landmarker.task on first use.

    Returns None (the caller then uses the LBF fallback) when the model is
    missing and cannot be derived, or when disabled with LUMEN_FACE_MESH=0.
    """
    global _convert_failed
    if os.environ.get("LUMEN_FACE_MESH", "1") == "0":
        return None
    p = onnx_path()
    if p.is_file() and p.stat().st_size > 1_000_000:
        return p
    with _lock:
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
        if _convert_failed:
            return None
        from ..models import manager

        task = manager.model_path("face-landmarker")
        if not task.is_file():
            return None
        try:
            from .tflite_onnx import convert_face_mesh

            convert_face_mesh(task, p)
            log.info("Converted the MediaPipe face-mesh model to ONNX: %s", p)
            return p
        except Exception as exc:  # noqa: BLE001 - corrupt/different model: keep the LBF fallback
            _convert_failed = str(exc)
            log.warning("Could not convert the face-mesh model to ONNX (%s); using LBF landmarks", exc)
            return None


def available() -> bool:
    return ensure_model() is not None


def _session():
    from ..hardware import onnx_session

    p = ensure_model()
    if p is None:
        raise FileNotFoundError("face-mesh ONNX model is not available")
    return onnx_session(str(p))


# --------------------------------------------------------------------------- ROI / crop
# A region of interest is (cx, cy, side, angle): square of ``side`` pixels centred
# on (cx, cy), rotated by ``angle`` radians (positive = clockwise in the image).


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def normalize_radians(a: float) -> float:
    return float(a - 2 * np.pi * np.floor((a + np.pi) / (2 * np.pi)))


def roi_from_landmarks(pts: np.ndarray, scale: float = 1.5) -> tuple[float, float, float, float]:
    """MediaPipe's landmarks -> ROI rule (bounding box, square long side, x1.5, eye-corner rotation)."""
    xy = pts[:, :2]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    a = normalize_radians(-np.arctan2(-(xy[263, 1] - xy[33, 1]), xy[263, 0] - xy[33, 0]))
    return float((x0 + x1) / 2), float((y0 + y1) / 2), float(max(x1 - x0, y1 - y0) * scale), a


def roi_from_detection(box, kps=None) -> tuple[float, float, float, float]:
    """First ROI from a YuNet detection (box x0,y0,x1,y1 and 5 keypoints: eyes, nose, mouth corners)."""
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    if kps is None:
        return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.5, 0.0
    k = np.asarray(kps, np.float64).reshape(5, 2)
    eye_l, eye_r, mouth = k[0], k[1], (k[3] + k[4]) / 2
    a = normalize_radians(-np.arctan2(-(eye_r[1] - eye_l[1]), eye_r[0] - eye_l[0]))
    R = _rot(a)
    eyes = (eye_l + eye_r) / 2
    # face-aligned frame: u along the eye line, v towards the mouth
    em = float((mouth - eyes) @ R[:, 1])
    iod = float(np.linalg.norm(eye_r - eye_l))
    bw, bh = x1 - x0, y1 - y0
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    # centre: between the box centre and the keypoint-based centre (robust to profile views)
    kc = eyes + R[:, 1] * (0.42 * em)
    centre = 0.5 * c + 0.5 * kc
    side = max(bw, bh, 2.2 * max(em, 0.5 * iod)) * 1.55
    return float(centre[0]), float(centre[1]), float(side), a


def _crop_affine(roi) -> np.ndarray:
    """2x3 affine mapping image pixels -> crop pixels (rect corners -> crop corners, like MediaPipe)."""
    cx, cy, side, a = roi
    R = _rot(a)
    h = side / 2
    local = np.array([[-h, h], [-h, -h], [h, -h]])
    src = (local @ R.T + [cx, cy]).astype(np.float32)
    dst = np.array([[0, INPUT], [0, 0], [INPUT, 0]], np.float32)
    return cv2.getAffineTransform(src, dst)


def crop(rgb_u8: np.ndarray, roi, border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """256x256 float32 RGB crop of the ROI (anti-aliased when the face is large)."""
    cx, cy, side, a = roi
    H, W = rgb_u8.shape[:2]
    M = _crop_affine(roi)
    scale = INPUT / max(side, 1e-6)
    if scale < 0.8:
        # pre-shrink the region with area averaging so bilinear sampling does not alias
        R = _rot(a)
        h = side / 2 * 1.02
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ R.T + [cx, cy]
        bx0, by0 = np.floor(corners.min(0)).astype(int) - 2
        bx1, by1 = np.ceil(corners.max(0)).astype(int) + 2
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
        if bx1 - bx0 > 4 and by1 - by0 > 4:
            f = scale * 1.25
            sub = rgb_u8[by0:by1, bx0:bx1]
            sw, sh = max(1, int(round(sub.shape[1] * f))), max(1, int(round(sub.shape[0] * f)))
            small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)
            fx, fy = sub.shape[1] / sw, sub.shape[0] / sh
            # small pixel (u, v) -> image ((u + 0.5) * fx - 0.5 + bx0, ...)
            S = np.array([[fx, 0, 0.5 * fx - 0.5 + bx0], [0, fy, 0.5 * fy - 0.5 + by0], [0, 0, 1]])
            M2 = np.vstack([M, [0, 0, 1]]) @ S
            out = cv2.warpAffine(small, M2[:2], (INPUT, INPUT), flags=cv2.INTER_LINEAR, borderMode=border)
            return out.astype(np.float32) / 255.0
    out = cv2.warpAffine(rgb_u8, M, (INPUT, INPUT), flags=cv2.INTER_LINEAR, borderMode=border)
    return out.astype(np.float32) / 255.0


def project(lm: np.ndarray, roi) -> np.ndarray:
    """Crop-space landmarks (478, 3) -> image pixels; z scaled like x."""
    cx, cy, side, a = roi
    R = _rot(a)
    uv = lm[:, :2] / INPUT - 0.5
    xy = uv @ R.T * side + [cx, cy]
    return np.column_stack([xy, lm[:, 2] / INPUT * side])


# --------------------------------------------------------------------------- inference
def infer(crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """crops (N, 256, 256, 3) float32 -> (landmarks (N, 478, 3) crop px, presence (N,) 0..1)."""
    sess = _session()
    lms, pres = [], []
    for c in crops:  # the converted graph has a fixed batch of 1
        out = sess.run(None, {"input": np.ascontiguousarray(c[None], np.float32)})
        by_name = dict(zip([o.name for o in sess.get_outputs()], out))
        lms.append(by_name["Identity"].reshape(478, 3))
        pres.append(1.0 / (1.0 + np.exp(-float(np.ravel(by_name["Identity_1"])[0]))))
    return np.array(lms, np.float64), np.array(pres)


# MediaPipe's IMAGE mode frames the landmark crop from its face detector; measured
# against its results, that crop is 1.36x the landmarks' bounding box (not the 1.5x of
# its video-tracking rule), so the refinement passes use 1.36.
REFINE_SCALE = 1.36
ITERATIONS = 3


def landmarks_for_roi(rgb_u8: np.ndarray, roi, iterations: int = ITERATIONS, scale: float = REFINE_SCALE) -> tuple[np.ndarray, float, tuple]:
    """Run the mesh on ``roi``, then re-centre on the landmarks ``iterations - 1`` times.

    Returns (points (478, 3) image px, presence score, final roi).
    """
    pts, score = None, 0.0
    for i in range(max(1, iterations)):
        lm, pres = infer(crop(rgb_u8, roi)[None])
        pts, score = project(lm[0], roi), float(pres[0])
        if i + 1 < iterations:
            roi = roi_from_landmarks(pts, scale)
    return pts, score, roi


# --------------------------------------------------------------------------- head pose (MediaPipe face geometry)
@lru_cache(maxsize=1)
def canonical() -> dict:
    """Canonical face model + Procrustes weights (from MediaPipe's geometry metadata, Apache-2.0)."""
    d = np.load(_DATA / "face_geometry.npz")
    return {k: d[k] for k in d.files}


def _weop(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted extended orthogonal Procrustes: 4x4 similarity mapping src (3,N) onto dst (3,N)."""
    sw = np.sqrt(w)
    ws, wt = src * sw, dst * sw
    total = float((sw * sw).sum())
    c = (ws * sw).sum(1) / total
    cws = ws - c[:, None] * sw[None]
    U, _, Vt = np.linalg.svd(wt @ cws.T)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, 2] *= -1
    R = U @ Vt
    scale = float((R @ cws * wt).sum() / max((cws * ws).sum(), 1e-12))
    RS = scale * R
    t = ((wt - RS @ ws) * sw).sum(1) / total
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = RS, t
    return M


def transformation_matrix(pts: np.ndarray, frame_w: float, frame_h: float, fov_deg: float = 63.0, near: float = 1.0) -> np.ndarray:
    """MediaPipe's facial transformation matrix for 468+ landmarks in pixels of a (frame_w x frame_h) frame."""
    cm = canonical()
    can = cm["vertices"][:, :3].T.astype(np.float64)  # (3, 468)
    w = np.zeros(can.shape[1])
    w[cm["basis_ids"]] = cm["basis_w"]
    lm = np.asarray(pts[:468], np.float64)
    s = np.vstack([lm[:, 0] / frame_w, lm[:, 1] / frame_h, lm[:, 2] / frame_w])
    h_near = 2 * near * np.tan(np.radians(fov_deg) / 2)
    w_near = frame_w * h_near / frame_h
    left, bottom = -w_near / 2, -h_near / 2
    # ProjectXY (origin top-left)
    s[1] = 1.0 - s[1]
    s = s * np.array([[w_near], [h_near], [w_near]]) + np.array([[left], [bottom], [0.0]])
    depth_offset = float(s[2].mean())

    def scale_of(L):
        return float(np.linalg.norm(_weop(can, L, w)[:3, 0]))

    def move_rescale_unproject(L, scale):
        L = L.copy()
        L[2] = (L[2] - depth_offset + near) / scale
        L[0] = L[0] * L[2] / near
        L[1] = L[1] * L[2] / near
        L[2] *= -1
        return L

    inter = s.copy()
    inter[2] *= -1
    s1 = scale_of(inter)
    s2 = scale_of(move_rescale_unproject(s, s1))
    metric = move_rescale_unproject(s, s1 * s2)
    return _weop(can, metric, w)


def pose_matrix_for(pts: np.ndarray) -> np.ndarray:
    """Transformation matrix in a virtual square crop around the face (like the MediaPipe path's refine crop)."""
    xy = pts[:, :2]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    side = max(x1 - x0, y1 - y0) * 1.9
    local = np.column_stack([xy - [(x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2], pts[:, 2]])
    return transformation_matrix(local, side, side)
