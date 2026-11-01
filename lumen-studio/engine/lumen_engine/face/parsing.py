"""Semantic face parsing: pixel-accurate face / hair / glasses / ear / neck masks.

A BiSeNet (ResNet backbone) trained on CelebAMask-HQ labels every pixel of an
aligned face crop with one of 19 classes (skin, brows, eyes, glasses, ears,
nose, inner mouth, lips, neck, cloth, hair, hat, ...). The face swap uses it to
decide *exactly* which pixels belong to the replaceable face in both photos
(hair strands, glasses, hats and hands stay on top, the source's fringe or
glasses are not transplanted), which is far more reliable than landmark
polygons plus colour models - especially with the less precise compatibility
landmarks.

* The crop is aligned like CelebA-HQ / FFHQ (eyes horizontal, side =
  max(4 x eye distance, 3.6 x eye-to-mouth distance)), so the network sees
  faces at the scale it was trained on, whatever their size in the photo.
* Inference runs once per (image, face); optional horizontal-flip augmentation
  remains available for evaluation. Class probabilities are cached and mapped back
  to any image region (or through a warp field) on demand.
* Everything is optional: :func:`available` is False when the model file is
  missing or cannot be loaded and callers fall back to their heuristics.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from ..models import manager

log = logging.getLogger("lumen.face.parsing")

MODEL_ID = "face-parsing"
SIZE = 512
CLASSES = (
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear", "ear_r",
    "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat",
)  # fmt: skip
IDX = {n: i for i, n in enumerate(CLASSES)}
GROUPS: dict[str, tuple[int, ...]] = {
    "background": (0,),
    "skin": (1,),
    "brows": (2, 3),
    "eyes": (4, 5),
    "glasses": (6,),
    "ears": (7, 8, 9),
    "nose": (10,),
    "mouth": (11,),  # inner mouth: teeth, tongue
    "lips": (12, 13),
    "neck": (14, 15),
    "cloth": (16,),
    "hair": (17,),
    "hat": (18,),
}
# the replaceable face: skin + inner features
FACE = GROUPS["skin"] + GROUPS["brows"] + GROUPS["eyes"] + GROUPS["nose"] + GROUPS["mouth"] + GROUPS["lips"]
# left/right classes swap under a horizontal flip
_FLIP = list(range(19))
for a, b in ((2, 3), (4, 5), (7, 8)):
    _FLIP[a], _FLIP[b] = b, a
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_lock = threading.Lock()  # session + inference
_cache_lock = threading.Lock()
_session = None
_failed: str | None = None
_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 8


def available() -> bool:
    """True if the parsing model is installed and loads."""
    if _failed is not None or not manager.is_installed(MODEL_ID):
        return False
    return _get_session() is not None


def _get_session():
    global _session, _failed
    with _lock:
        if _session is None and _failed is None:
            try:
                from ..hardware import onnx_session

                sess = onnx_session(str(manager.require(MODEL_ID)))
                shp = sess.get_inputs()[0].shape
                if len(shp) != 4 or shp[1] != 3:
                    raise ValueError(f"unexpected input shape {shp}")
                _session = sess
            except Exception as exc:  # noqa: BLE001 - corrupt file, old runtime...
                _failed = str(exc)
                log.warning("Face parsing model unavailable (%s); using heuristic face masks.", exc)
        return _session


# --------------------------------------------------------------------------- alignment
def _eye_mouth(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .landmarks import topology

    t = topology()
    e0 = xy[t["right_eye"]].mean(0)  # subject's right eye = image left (unmirrored)
    e1 = xy[t["left_eye"]].mean(0)
    return e0, e1, xy[61], xy[291]


def align_matrix(xy: np.ndarray, size: int = SIZE, zoom: float = 1.0) -> np.ndarray:
    """2x3 similarity transform image -> aligned ``size`` x ``size`` crop (CelebA-HQ layout)."""
    xy = np.asarray(xy, np.float64)[:, :2]
    e0, e1, m0, m1 = _eye_mouth(xy)
    eye_avg, mouth_avg = (e0 + e1) / 2, (m0 + m1) / 2
    e2e, e2m = e1 - e0, mouth_avg - eye_avg
    x = e2e - np.array([-e2m[1], e2m[0]])
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        raise ValueError("degenerate landmarks")
    x /= nx
    half = max(np.linalg.norm(e2e) * 2.0, np.linalg.norm(e2m) * 1.8) * zoom
    y = np.array([-x[1], x[0]])
    c = eye_avg + e2m * 0.1
    k = size / (2.0 * half)
    lin = k * np.stack([x, y])
    t = size / 2.0 - lin @ c
    return np.hstack([lin, t[:, None]])


def _fingerprint(img: np.ndarray, xy: np.ndarray, offset=(0, 0)) -> np.ndarray:
    """Coarse thumbnail of the face region (identifies the same photo in a sub-image)."""
    xy = np.asarray(xy)[:, :2] - np.asarray(offset, np.float64)
    x0, y0 = np.floor(xy.min(0)).astype(int)
    x1, y1 = np.ceil(xy.max(0)).astype(int)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((6, 6, 3), np.float32)
    sub = np.asarray(img[y0:y1, x0:x1], np.float32)
    if sub.max() > 1.5:
        sub = sub / 255.0
    return cv2.resize(sub, (6, 6), interpolation=cv2.INTER_AREA).astype(np.float32)


def _key(xy: np.ndarray) -> str:
    return hashlib.blake2b(np.round(np.asarray(xy)[:468, :2] * 4).astype(np.int32).tobytes(), digest_size=12).hexdigest()


@dataclass
class _Entry:
    M: np.ndarray  # image -> crop (in the coordinates of the parsed image)
    probs: np.ndarray  # (SIZE, SIZE, 19) float16, crop space
    fp: np.ndarray


def _infer(crop_u8: np.ndarray, flip_tta: bool = False) -> np.ndarray:
    sess = _get_session()
    if sess is None:
        raise RuntimeError("face parsing model unavailable")
    x = (crop_u8.astype(np.float32) / 255.0 - _MEAN) / _STD
    batch = [x]
    if flip_tta:
        batch.append(x[:, ::-1])
    inp = np.ascontiguousarray(np.stack(batch).transpose(0, 3, 1, 2))
    name = sess.get_inputs()[0].name
    with _lock:  # one inference at a time (memory, CPU slots)
        outs = []
        for i in range(len(batch)):  # batch 1 keeps peak memory low
            outs.append(sess.run([sess.get_outputs()[0].name], {name: inp[i : i + 1]})[0][0])
    logits = outs[0]
    if flip_tta:
        logits = 0.5 * (logits + outs[1][_FLIP, :, ::-1])
    logits = logits - logits.max(0, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(0, keepdims=True)
    return np.ascontiguousarray(p.transpose(1, 2, 0)).astype(np.float16)


def _entry(img: np.ndarray, xy: np.ndarray, offset=(0, 0), flip_tta: bool = False, tag: str = "") -> _Entry | None:
    """Parse the face with landmarks ``xy`` (full-image coords) in ``img`` (whose
    pixel (0,0) is at ``offset`` in full-image coords). Cached."""
    if not available():
        return None
    key = _key(xy) + tag
    fp = _fingerprint(img, xy, offset)
    with _cache_lock:
        e = _cache.get(key)
        if e is not None and np.abs(e.fp - fp).mean() < 0.02:
            _cache.move_to_end(key)
            return e
    try:
        M = align_matrix(xy)
    except ValueError:
        return None
    Ml = M.copy()
    Ml[:, 2] += M[:, :2] @ np.asarray(offset, np.float64)  # crop coords from local pixel coords
    src = np.asarray(img)
    if src.dtype != np.uint8:
        src = (np.clip(src, 0, 1) * 255 + 0.5).astype(np.uint8)
    k = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    if k < 0.5:  # strong downscale: pre-blur against aliasing
        src = cv2.GaussianBlur(src, (0, 0), 0.4 / k)
    crop = cv2.warpAffine(src, Ml, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    try:
        probs = _infer(crop, flip_tta)
    except Exception as exc:  # noqa: BLE001
        log.warning("Face parsing failed: %s", exc)
        return None
    e = _Entry(M=M, probs=probs, fp=fp)
    with _cache_lock:
        _cache[key] = e
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return e


# --------------------------------------------------------------------------- results
@dataclass
class FaceParse:
    """Class probabilities (h, w, 19) for an image region."""

    probs: np.ndarray

    def group(self, *names: str) -> np.ndarray:
        idx: list[int] = []
        for n in names:
            idx.extend(GROUPS[n] if n in GROUPS else (IDX[n],))
        return self.probs[..., idx].astype(np.float32).sum(-1)

    def face(self) -> np.ndarray:
        return self.probs[..., list(FACE)].astype(np.float32).sum(-1)

    def labels(self) -> np.ndarray:
        return self.probs.argmax(-1).astype(np.uint8)

    @property
    def shape(self) -> tuple[int, int]:
        return self.probs.shape[:2]


def _sample(e: _Entry, cx: np.ndarray, cy: np.ndarray) -> FaceParse:
    """Sample crop-space probabilities at crop coordinates (cx, cy)."""
    p = cv2.remap(e.probs.astype(np.float32), cx.astype(np.float32), cy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if p.ndim == 2:
        p = p[..., None]
    valid = cv2.remap(np.ones((SIZE, SIZE), np.float32), cx.astype(np.float32), cy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p[..., 0] += 1.0 - valid  # outside the analysed crop: background
    return FaceParse(p)


def parse_region(img: np.ndarray, xy: np.ndarray, roi: tuple[int, int, int, int] | None = None, offset=(0, 0), tag: str = "") -> FaceParse | None:
    """Parse the face with landmarks ``xy`` and return probabilities over ``roi``.

    ``img`` may be a sub-image whose top-left pixel lies at ``offset`` (x, y) in
    the coordinate frame of ``xy``; ``roi`` = (y0, y1, x0, x1) in that same frame
    (default: the whole of ``img``). Returns None if the model is unavailable.
    """
    e = _entry(img, xy, offset, tag=tag)
    if e is None:
        return None
    if roi is None:
        roi = (offset[1], offset[1] + img.shape[0], offset[0], offset[0] + img.shape[1])
    y0, y1, x0, x1 = roi
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    M = e.M
    cx = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
    cy = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]
    return _sample(e, cx, cy)


def parse_through_warp(img: np.ndarray, xy: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> FaceParse | None:
    """Parse a face in its own photo, then pull the probabilities through a warp
    field (``map_x``/``map_y``: per output pixel, the source-image position)."""
    e = _entry(img, xy)
    if e is None:
        return None
    M = e.M
    cx = M[0, 0] * map_x + M[0, 1] * map_y + M[0, 2]
    cy = M[1, 0] * map_x + M[1, 1] * map_y + M[1, 2]
    return _sample(e, cx, cy)


def cached_through_warp(xy: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> FaceParse | None:
    """Like :func:`parse_through_warp` but only from the cache (no image needed)."""
    if not available():
        return None
    with _cache_lock:
        e = _cache.get(_key(xy))
    if e is None:
        return None
    M = e.M
    cx = M[0, 0] * map_x + M[0, 1] * map_y + M[0, 2]
    cy = M[1, 0] * map_x + M[1, 1] * map_y + M[1, 2]
    return _sample(e, cx, cy)


def colorize(labels: np.ndarray) -> np.ndarray:
    """Debug visualisation of a label map (float RGB)."""
    pal = np.array(
        [[0, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0], [170, 255, 0], [0, 255, 85], [0, 255, 170],
         [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255]],
        np.float32,
    ) / 255.0  # fmt: skip
    return pal[labels]


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
