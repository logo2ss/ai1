"""MediaPipe-free face detection and landmarks.

Some computers cannot run MediaPipe's prebuilt binaries - most often virtual
machines whose virtual CPU does not expose AVX/AVX2, where MediaPipe dies with
"illegal instruction". OpenCV, NumPy and ONNX Runtime select CPU features at
runtime and keep working there, so this module provides an OpenCV-only path:

1. **Detection** - YuNet (ONNX, run by OpenCV's FaceDetectorYN) on the full
   frame and on overlapping tiles, with non-maximum suppression.
2. **Landmarks** - OpenCV's LBF facemark regressor (68 iBUG points), run twice:
   the second pass re-centres the box on the first-pass landmarks.
3. **Dense mesh** - the 68 points are mapped to the 478-point MediaPipe face
   mesh: a mean face template (built from MediaPipe results, see
   ``scripts/build_face_template.py``) is aligned with a similarity transform
   and bent with a thin-plate spline through the 68 correspondences, so the
   rest of the face pipeline (warp, masks, relighting...) is unchanged.
4. **Head pose** - ``solvePnP`` of the 68 template points in 3D.

Preferred landmark step (when ``face_landmarks_detector.onnx`` exists or can be
converted from ``face_landmarker.task``): MediaPipe's own face-mesh network run
with ONNX Runtime (see ``meshnet.py``) - real 478-point landmarks with depth and
MediaPipe-convention head pose. Steps 2-4 (LBF + template) are the last resort
when that model is unavailable (or ``LUMEN_FACE_BACKEND=lbf``).
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..errors import NoFaceFoundError
from ..models import manager
from .geometry import umeyama
from .landmarks import Face, _iou, _roll_from_eyes, pose_from_matrix

MIN_PRESENCE = 0.5  # face-mesh presence score below which a detection is treated as not a face

_DATA = Path(__file__).parent / "data"
_lock = threading.RLock()
_ssd = None
_lbf = None

# iBUG-68 anchor points and their MediaPipe counterparts (eye corners, mouth corners, nose tip, chin)
ANCH_68 = [36, 39, 42, 45, 48, 54, 30, 8]
ANCH_MP = [33, 133, 362, 263, 61, 291, 1, 152]


@lru_cache(maxsize=1)
def template() -> dict:
    d = np.load(_DATA / "face_template.npz")
    return {k: d[k] for k in d.files}


def _models(lbf_needed: bool = True):
    """(YuNet detector, LBF facemark or None); the 56 MB LBF model is only loaded when needed."""
    global _ssd, _lbf
    with _lock:
        if _ssd is None:
            _ssd = cv2.FaceDetectorYN.create(str(manager.require("face-detector-yunet")), "", (640, 640), 0.6, 0.3, 200)
        if _lbf is None and lbf_needed:
            lbf = cv2.face.createFacemarkLBF()
            lbf.loadModel(str(manager.require("face-landmarks-lbf")))
            _lbf = lbf
    return _ssd, _lbf


def _detect_once(det, bgr: np.ndarray, off=(0, 0), conf: float = 0.6) -> list:
    """YuNet on a 640x640 letterboxed copy; boxes mapped back to image pixels."""
    h, w = bgr.shape[:2]
    s = 640.0 / max(h, w)
    small = cv2.resize(bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small
    with _lock:
        _, faces = det.detect(canvas)
    out = []
    for f in faces if faces is not None else []:
        if float(f[14]) < conf:
            continue
        x, y, bw, bh = (f[:4] / s).tolist()
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(w - 1.0, x + bw), min(h - 1.0, y + bh)
        kps = f[4:14].reshape(5, 2) / s + np.asarray(off, np.float64)  # eyes, nose tip, mouth corners
        if x1 - x0 > 8 and y1 - y0 > 8:
            out.append((x0 + off[0], y0 + off[1], x1 + off[0], y1 + off[1], float(f[14]), kps))
    return out


def detect_boxes(bgr: np.ndarray, conf: float = 0.6) -> list:
    """Face boxes (x0, y0, x1, y1, score) from the full frame and overlapping tiles."""
    det, _ = _models(lbf_needed=False)
    h, w = bgr.shape[:2]
    boxes = _detect_once(det, bgr, conf=conf)
    for grid in (2, 3):
        if max(h, w) / grid < 400:
            break
        th, tw = h / grid, w / grid
        side = int(max(th, tw) * 1.4)
        for gy in range(grid):
            for gx in range(grid):
                cx, cy = (gx + 0.5) * tw, (gy + 0.5) * th
                x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
                crop = bgr[y0 : y0 + side, x0 : x0 + side]
                if crop.size:
                    boxes += _detect_once(det, crop, (x0, y0), conf=max(conf, 0.7))
    if not boxes:
        return []
    rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    keep = cv2.dnn.NMSBoxes(rects, [b[4] for b in boxes], conf, 0.35)
    return [boxes[i] for i in np.array(keep).ravel()]


def _lbf_box(box, scale: float, dy: float):
    x0, y0, x1, y1 = box[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + dy * (y1 - y0)
    s = max(x1 - x0, y1 - y0) * scale
    return np.array([[cx - s / 2, cy - s / 2, s, s]], np.float64)


def _kp_error(p: np.ndarray, kps: np.ndarray) -> float:
    """Disagreement between LBF landmarks and the detector's 5 keypoints (fraction of eye distance)."""
    eyes = sorted([p[36:42].mean(0), p[42:48].mean(0)], key=lambda q: q[0])
    mouth = sorted([p[48], p[54]], key=lambda q: q[0])
    ours = np.array([eyes[0], eyes[1], p[30], mouth[0], mouth[1]])
    k_eyes = sorted([kps[0], kps[1]], key=lambda q: q[0])
    k_mouth = sorted([kps[3], kps[4]], key=lambda q: q[0])
    theirs = np.array([k_eyes[0], k_eyes[1], kps[2], k_mouth[0], k_mouth[1]])
    iod = np.linalg.norm(theirs[1] - theirs[0]) + 1e-6
    return float(np.linalg.norm(ours - theirs, axis=1).mean() / iod)


def fit_landmarks(gray: np.ndarray, box, refine: bool = True) -> np.ndarray:
    """68 iBUG landmarks for a detected face box.

    LBF regression can converge to a wrong shape from an unlucky start, so it
    is started from several boxes and the fit that best agrees with the
    detector's own 5 keypoints (eyes, nose, mouth corners) is kept; a second
    pass re-centres the box on the chosen landmarks.
    """
    _, lbf = _models()
    t = template()
    scale, dy = float(t["box_scale"]), float(t["box_dy"])
    kps = box[5] if len(box) > 5 else None
    alts = [(scale * k, dy + d) for k in (0.92, 1.0, 1.1) for d in (-0.06, 0.0, 0.06) if (k, d) != (1.0, 0.0)]
    best, best_err = None, np.inf
    for i, (sc_, dy_) in enumerate([(scale, dy)] + (alts if kps is not None else [])):
        if i == 1 and best is not None and best_err < 0.08:
            break  # the calibrated start already agrees with the detector: keep it
        with _lock:
            ok, lm = lbf.fit(gray, _lbf_box(box, sc_, dy_))
        if not ok or not len(lm):
            continue
        cand = np.asarray(lm[0], np.float64).reshape(68, 2)
        err = _kp_error(cand, kps) if kps is not None else 0.0
        if err < best_err:
            best, best_err = cand, err
    if best is None:
        raise NoFaceFoundError("Facial landmarks could not be fitted.")
    p = best
    if refine:
        r = t["lm_to_box"]
        lx0, ly0 = p.min(0)
        lx1, ly1 = p.max(0)
        lw, lh = lx1 - lx0, ly1 - ly0
        b2 = (lx0 + r[0] * lw, ly0 + r[1] * lh, lx0 + r[2] * lw, ly0 + r[3] * lh)
        with _lock:
            ok2, lm2 = lbf.fit(gray, _lbf_box(b2, scale, dy))
        if ok2 and len(lm2):
            p2 = np.asarray(lm2[0], np.float64).reshape(68, 2)
            if kps is None or _kp_error(p2, kps) <= best_err + 0.05:
                p = p2
    if kps is not None and _kp_error(p, kps) > 0.15:
        # still disagreeing: pull the fit onto the detector keypoints with a similarity transform
        eyes = sorted([kps[0], kps[1]], key=lambda q: q[0])
        mouth = sorted([kps[3], kps[4]], key=lambda q: q[0])
        theirs = np.array([eyes[0], eyes[1], kps[2], mouth[0], mouth[1]])
        ours = np.array([p[36:42].mean(0), p[42:48].mean(0), p[30], p[48], p[54]])
        M = umeyama(ours, theirs)
        p = p @ M[:, :2].T + M[:, 2]
    return p


def densify(p68: np.ndarray, image_shape: tuple[int, int]) -> Face:
    """Map 68 landmarks onto the 478-point mesh topology and estimate head pose."""
    from scipy.interpolate import RBFInterpolator

    t = template()
    tpl68, tpl478 = t["template68"].astype(np.float64), t["template478"].astype(np.float64)
    S = umeyama(tpl68, p68)
    sc = float(np.sqrt(abs(np.linalg.det(S[:, :2]))))
    base68 = tpl68 @ S[:, :2].T + S[:, 2]
    base478 = tpl478[:, :2] @ S[:, :2].T + S[:, 2]
    res = p68 - base68
    iod = np.linalg.norm(p68[42:48].mean(0) - p68[36:42].mean(0)) + 1e-6
    rbf = RBFInterpolator(base68, res, kernel="thin_plate_spline", smoothing=0.02 * iod)
    pts2 = base478 + rbf(base478)
    pts = np.column_stack([pts2, tpl478[:, 2] * sc])
    h, w = image_shape
    f = float(max(h, w))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    obj = t["model68_3d"].astype(np.float64)
    rv0 = np.zeros((3, 1))
    tv0 = np.array([[0.0], [0.0], [f / max(sc, 1e-6)]])
    ok, rvec, _ = cv2.solvePnP(obj, p68.astype(np.float64), K, None, rv0, tv0, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    R = cv2.Rodrigues(rvec)[0] if ok else np.eye(3)
    yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2]))) * float(t["yaw_sign"])
    pitch = float(np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1)))) * float(t["pitch_sign"])
    if "pose_coef" in t:  # pose model calibrated against MediaPipe (more reliable than PnP on a mean face)
        yaw, pitch = (pose_features(p68) @ t["pose_coef"]).tolist()
    M = np.eye(4)
    M[:3, :3] = R
    face = Face(points=pts, matrix=M, source="opencv", yaw=yaw, pitch=pitch, roll=_roll_from_eyes(pts2))
    face.extra["landmarks68"] = p68
    return face


def pose_features(p: np.ndarray) -> np.ndarray:
    """Geometric head-pose features of the 68 landmarks (see scripts/build_face_template.py)."""
    E = p[36:48].mean(0)
    Mo = p[48:68].mean(0)
    N = p[30]
    u = p[42:48].mean(0) - p[36:42].mean(0)
    u /= np.linalg.norm(u) + 1e-9
    v = np.array([-u[1], u[0]])
    if v @ (Mo - E) < 0:
        v = -v
    jw = np.linalg.norm(p[16] - p[0]) + 1e-9
    em = (Mo - E) @ v + 1e-9
    dl, dr = np.linalg.norm(N - p[0]), np.linalg.norm(N - p[16])
    return np.array([((N - (p[0] + p[16]) / 2) @ u) / jw, (dl - dr) / (dl + dr), ((N - E) @ v) / em, ((p[8] - Mo) @ v) / em, ((E - (p[0] + p[16]) / 2) @ v) / jw, 1.0])


def mesh_available() -> bool:
    """True when the ONNX face mesh (MediaPipe's landmark network) can be used."""
    if os.environ.get("LUMEN_FACE_BACKEND", "").lower() == "lbf":
        return False
    from . import meshnet

    return meshnet.available()


def landmark_method() -> str:
    return "mesh-onnx" if mesh_available() else "lbf"


def mesh_face(u8: np.ndarray, box, source: str = "opencv", roi=None) -> Face:
    """478 landmarks (with depth) and head pose from MediaPipe's face-mesh network via ONNX Runtime."""
    from . import meshnet

    if roi is None:
        roi = meshnet.roi_from_detection(box[:4], box[5] if len(box) > 5 else None)
    pts, score, roi = meshnet.landmarks_for_roi(u8, roi)
    M = meshnet.pose_matrix_for(pts)
    yaw, pitch, _ = pose_from_matrix(M)
    face = Face(points=pts, matrix=M, source=source, yaw=yaw, pitch=pitch, roll=_roll_from_eyes(pts))
    face.extra.update(presence=round(score, 3), landmarks="mesh-onnx", roi=tuple(float(v) for v in roi))
    return face


def detect_faces(rgb: np.ndarray, max_faces: int = 30) -> list[Face]:
    u8 = rgb if rgb.dtype == np.uint8 else cv2.convertScaleAbs(rgb, alpha=255.0)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    use_mesh = mesh_available()
    faces: list[Face] = []
    for b in sorted(detect_boxes(bgr), key=lambda b: -b[4])[:max_faces]:
        try:
            if use_mesh:
                f = mesh_face(u8, b)
                if f.extra["presence"] < MIN_PRESENCE:
                    continue
            else:
                f = densify(fit_landmarks(gray, b), u8.shape[:2])
        except NoFaceFoundError:
            continue
        if all(_iou(f.bbox, g.bbox) < 0.3 for g in faces):
            faces.append(f)
    faces.sort(key=lambda f: (f.bbox[0], f.bbox[1]))
    return faces


def _rotated_candidates(bgr: np.ndarray) -> list:
    """YuNet on the crop turned by 90/180/270 degrees (it only finds roughly upright faces);
    boxes and keypoints are mapped back to the crop, so the keypoints carry the true orientation."""
    h, w = bgr.shape[:2]
    inv = {
        cv2.ROTATE_90_CLOCKWISE: lambda p: np.column_stack([p[:, 1], h - 1 - p[:, 0]]),
        cv2.ROTATE_180: lambda p: np.column_stack([w - 1 - p[:, 0], h - 1 - p[:, 1]]),
        cv2.ROTATE_90_COUNTERCLOCKWISE: lambda p: np.column_stack([w - 1 - p[:, 1], p[:, 0]]),
    }
    out = []
    for code, back in inv.items():
        for b in detect_boxes(cv2.rotate(bgr, code), conf=0.4):
            corners = back(np.array([[b[0], b[1]], [b[2], b[3]]], np.float64))
            (x0, y0), (x1, y1) = corners.min(0), corners.max(0)
            out.append((x0, y0, x1, y1, b[4], back(np.asarray(b[5], np.float64))))
    return sorted(out, key=lambda b: -b[4])[:1]


def detect_in_box(rgb: np.ndarray, box) -> Face:
    u8 = rgb if rgb.dtype == np.uint8 else cv2.convertScaleAbs(rgb, alpha=255.0)
    x0, y0, x1, y1 = [float(v) for v in box]
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = u8.shape[:2]
    ax0, ay0 = int(max(0, cx - side)), int(max(0, cy - side))
    ax1, ay1 = int(min(W, cx + side)), int(min(H, cy + side))
    crop = cv2.cvtColor(u8[ay0:ay1, ax0:ax1], cv2.COLOR_RGB2BGR)
    cand = detect_boxes(crop, conf=0.4)
    if mesh_available():  # YuNet may also fire on a sideways face, with wrong keypoint orientation
        rc = _rotated_candidates(crop)
        if rc and (not cand or rc[0][4] > max(c[4] for c in cand) + 0.05):
            cand = rc
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    if cand:
        b = max(cand, key=lambda b: b[4])
        box_full = (b[0] + ax0, b[1] + ay0, b[2] + ax0, b[3] + ay0, b[4], b[5])
    else:
        box_full = (x0, y0, x1, y1)  # trust the user's box
    if mesh_available():
        if cand:
            return mesh_face(u8, box_full, source="manual")
        # no detection: try the user's box at several in-plane rotations, keep the most face-like
        best = None
        for deg in (0, -25, 25, -50, 50, 90, -90, 180):
            f = mesh_face(u8, box_full, source="manual", roi=(cx, cy, side * 1.5, np.radians(deg)))
            if best is None or f.extra["presence"] > best.extra["presence"]:
                best = f
            if best.extra["presence"] > 0.9:
                break
        if best.extra["presence"] >= 0.2:
            return best
        raise NoFaceFoundError(
            "No face could be located inside the selected area.",
            hint="Draw the box tightly around one face (forehead to chin). Faces that are very small, "
            "heavily occluded or seen in full profile cannot be landmarked reliably.",
        )
    f = densify(fit_landmarks(gray, box_full), u8.shape[:2])
    f.source = "manual"
    return f
