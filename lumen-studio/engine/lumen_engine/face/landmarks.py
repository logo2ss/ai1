"""Face detection, dense landmarks and head pose (MediaPipe Face Landmarker).

The landmarker returns 478 3D landmarks (468 mesh vertices + 10 iris points)
and a 4x4 facial transformation matrix from which head pose is derived.

To find small faces in group photos, detection runs on the full image and on
overlapping tiles; every detection is then refined by re-running the model on
a tight crop around the face, which yields the most accurate landmarks.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from ..errors import NoFaceFoundError, ProcessingError
from ..models import manager

log = logging.getLogger(__name__)
os.environ.setdefault("GLOG_minloglevel", "2")


# --------------------------------------------------------------------------- topology
@lru_cache(maxsize=1)
def topology() -> dict:
    """Index sets of the MediaPipe face mesh, derived from the model's own connection tables.

    Loaded from ``data/face_topology.npz`` (generated from MediaPipe's tables) so the
    compatibility path never imports MediaPipe; ``mesh_triangles`` are the faces of
    MediaPipe's canonical mesh (``data/face_geometry.npz``), which - unlike
    ``triangles`` - also cover the eye and mouth openings.
    """
    from pathlib import Path

    data = Path(__file__).parent / "data"
    if (data / "face_topology.npz").is_file():
        d = np.load(data / "face_topology.npz")
        out = {k: (d[k] if k == "triangles" else d[k].tolist()) for k in d.files}
        out["mesh_triangles"] = np.load(data / "face_geometry.npz")["triangles"].astype(np.int32)
        return out
    from mediapipe.tasks.python import vision

    C = vision.FaceLandmarksConnections

    def ordered_loop(conns) -> list[int]:
        nxt = {c.start: c.end for c in conns}
        start = conns[0].start
        loop, cur = [start], nxt[start]
        while cur != start and len(loop) < 500:
            loop.append(cur)
            cur = nxt.get(cur, start)
        return loop

    def idx(conns) -> list[int]:
        return sorted({c.start for c in conns} | {c.end for c in conns})

    edges = {(min(c.start, c.end), max(c.start, c.end)) for c in C.FACE_LANDMARKS_TESSELATION}
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    tris = set()
    for a, b in edges:
        for c in adj[a] & adj[b]:
            tris.add(tuple(sorted((a, b, c))))
    lips = idx(C.FACE_LANDMARKS_LIPS)
    inner_lip = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    outer_lip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    return {
        "oval": ordered_loop(C.FACE_LANDMARKS_FACE_OVAL),
        "left_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
        "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "left_brow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        "right_brow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        "lips": lips,
        "outer_lip": outer_lip,
        "inner_lip": inner_lip,
        "nostrils": [79, 166, 75, 60, 20, 238, 309, 392, 305, 290, 250, 458],
        "triangles": np.array(sorted(tris), dtype=np.int32),
        # stable points for similarity alignment (eye corners, nose, mouth corners, chin, temples)
        "anchors": [33, 133, 362, 263, 1, 4, 61, 291, 152, 234, 454, 10],
        "mesh_triangles": np.load(Path(__file__).parent / "data" / "face_geometry.npz")["triangles"].astype(np.int32),
    }


# --------------------------------------------------------------------------- data
@dataclass
class Face:
    points: np.ndarray  # (478, 3) pixel coordinates; z in pixels (relative depth)
    matrix: np.ndarray  # 4x4 facial transformation matrix
    source: str = "auto"
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        oval = self.xy[topology()["oval"]]
        x0, y0 = oval.min(0)
        x1, y1 = oval.max(0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def size(self) -> float:
        """Inter-ocular-independent face size: sqrt of the face-oval area."""
        oval = self.xy[topology()["oval"]].astype(np.float32)
        return float(np.sqrt(abs(cv2.contourArea(oval))))

    def mouth_openness(self) -> float:
        t = topology()
        inner = self.xy[t["inner_lip"]].astype(np.float32)
        return float(abs(cv2.contourArea(inner)) / max(self.size**2, 1e-6))

    def eye_openness(self) -> float:
        t = topology()
        a = [abs(cv2.contourArea(self.xy[t[k]].astype(np.float32))) for k in ("left_eye", "right_eye")]
        return float(np.mean(a) / max(self.size**2, 1e-6))

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            "bbox": [x0, y0, x1, y1],
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "size": round(self.size, 1),
            "source": self.source,
            "mouth_open": self.mouth_openness() > 0.004,
            "landmarks": self.xy[:468:1].round(1).tolist(),
        }


def pose_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from the MediaPipe facial transformation matrix.

    Sign convention: positive yaw = face turned towards image right,
    positive pitch = looking up, positive roll = tilted clockwise in the image.
    """
    R = np.asarray(m, np.float64)[:3, :3]
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    yaw = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1)))
    roll = np.degrees(np.arctan2(-R[1, 0], R[0, 0]))
    return float(yaw), float(pitch), float(roll)


# --------------------------------------------------------------------------- detector
class _Landmarker:
    def __init__(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        path = manager.require("face-landmarker")
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=12,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._lm = vision.FaceLandmarker.create_from_options(opts)
        except Exception as exc:  # GPU/EGL-less VMs, missing libGL, corrupt model...
            log.exception("MediaPipe FaceLandmarker failed to start")
            raise ProcessingError(
                f"MediaPipe could not start: {exc}",
                hint="MediaPipe could not start. On Linux install libglvnd / mesa "
                "(Arch: sudo pacman -S libglvnd mesa), then restart Lumen Studio. "
                "If the face model file may be damaged, reinstall it from Models.",
            ) from exc
        self._lock = threading.Lock()

    def run(self, rgb_u8: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        import mediapipe as mp

        h, w = rgb_u8.shape[:2]
        with self._lock:
            res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8)))
        out = []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], np.float64)
            mat = np.array(res.facial_transformation_matrixes[i]) if res.facial_transformation_matrixes else np.eye(4)
            out.append((pts, mat))
        return out


_instance: _Landmarker | None = None
_instance_lock = threading.Lock()


def landmarker() -> _Landmarker:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = _Landmarker()
        return _instance


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _make_face(pts: np.ndarray, mat: np.ndarray, source: str) -> Face:
    y, p, r = pose_from_matrix(mat)
    return Face(points=pts, matrix=mat, source=source, yaw=y, pitch=p, roll=r)


def _run_crop(rgb_u8: np.ndarray, box: tuple[float, float, float, float], target: int = 512, rotate: float = 0.0):
    """Run the landmarker on a (possibly rotated) square crop; map points back."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0)
    s = target / max(side, 1)
    M = cv2.getRotationMatrix2D((cx, cy), rotate, s)
    M[0, 2] += target / 2 - cx
    M[1, 2] += target / 2 - cy
    crop = cv2.warpAffine(rgb_u8, M, (target, target), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    Minv = cv2.invertAffineTransform(M)
    out = []
    for pts, mat in landmarker().run(crop):
        xy = pts[:, :2] @ Minv[:, :2].T + Minv[:, 2]
        z = pts[:, 2] / s
        out.append((np.column_stack([xy, z]), mat, rotate))
    return out


def _refine(rgb_u8: np.ndarray, face: Face) -> Face:
    x0, y0, x1, y1 = face.bbox
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.9
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    res = _run_crop(rgb_u8, box, rotate=-face.roll if abs(face.roll) > 20 else 0.0)
    best, best_iou = None, 0.0
    for pts, mat, rot in res:
        f = _make_face(pts, mat, face.source)
        iou = _iou(f.bbox, face.bbox)
        if iou > best_iou:
            best, best_iou = f, iou
    if best is not None and best_iou > 0.4:
        if best.roll is not None and abs(best.roll) < 1e-9:
            pass
        # roll from the matrix is relative to the rotated crop: recompute from eyes
        best.roll = _roll_from_eyes(best.xy)
        return best
    return face


def _roll_from_eyes(xy: np.ndarray) -> float:
    l, r = xy[[33, 133]].mean(0), xy[[362, 263]].mean(0)
    d = r - l
    return float(np.degrees(np.arctan2(d[1], d[0])))


def detect_faces(rgb: np.ndarray, max_faces: int = 30) -> list[Face]:
    """Detect all faces in an RGB float image. Returns faces sorted left-to-right."""
    from . import backend
    if backend.mediapipe_status() is not None or os.environ.get("LUMEN_FACE_BACKEND", "").lower() in ("opencv", "compat", "lbf"):
        from .fallback import detect_faces as compat_detect
        return compat_detect(rgb)[:max_faces]
    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    h, w = u8.shape[:2]
    cands: list[Face] = []

    def add(pts, mat, src="auto"):
        f = _make_face(pts, mat, src)
        x0, y0, x1, y1 = f.bbox
        if x1 - x0 < 12 or y1 - y0 < 12:
            return
        for g in cands:
            if _iou(g.bbox, f.bbox) > 0.3:
                return
        cands.append(f)

    # 1) full frame (downscaled for speed; landmarks are refined later)
    scale = min(1.0, 1280 / max(h, w))
    small = cv2.resize(u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else u8
    for pts, mat in landmarker().run(small):
        pts = pts / scale
        add(pts, mat)
    # 2) overlapping tiles at two scales for small faces
    for grid in (2, 3, 4):
        if max(h, w) / grid < 160:
            break
        th, tw = h / grid, w / grid
        for gy in range(grid):
            for gx in range(grid):
                side = max(th, tw) * 1.5
                cx, cy = (gx + 0.5) * tw, (gy + 0.5) * th
                for pts, mat, _ in _run_crop(u8, (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), target=640):
                    add(pts, mat)
        if len(cands) >= max_faces:
            break
    faces = [_refine(u8, f) for f in cands[:max_faces]]
    faces.sort(key=lambda f: (f.bbox[0], f.bbox[1]))
    return faces


def detect_in_box(rgb: np.ndarray, box: tuple[float, float, float, float]) -> Face:
    """Manual selection: find the face inside a user-drawn box.

    Tries several crop margins and in-plane rotations, which recovers faces the
    automatic pass misses (small, tilted, partially cropped, low contrast).
    """
    from . import backend
    if backend.mediapipe_status() is not None or os.environ.get("LUMEN_FACE_BACKEND", "").lower() in ("opencv", "compat", "lbf"):
        from .fallback import detect_in_box as compat_box
        return compat_box(rgb, box)
    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0)
    best = None
    for margin in (1.6, 1.25, 2.2):
        for rot in (0, -25, 25, -50, 50, 90, -90, 180):
            s = side * margin
            for pts, mat, r in _run_crop(u8, (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2), rotate=rot):
                f = _make_face(pts, mat, "manual")
                fx0, fy0, fx1, fy1 = f.bbox
                fcx, fcy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
                if x0 - side * 0.2 <= fcx <= x1 + side * 0.2 and y0 - side * 0.2 <= fcy <= y1 + side * 0.2:
                    f.roll = _roll_from_eyes(f.xy)
                    if best is None or f.size > best.size:
                        best = f
            if best is not None:
                return best
    raise NoFaceFoundError(
        "No face could be located inside the selected area.",
        hint="Draw the box tightly around one face (forehead to chin). Faces that are very small, "
        "heavily occluded or seen in full profile cannot be landmarked reliably.",
    )
