"""One-click object selection ("smart select") with the MediaPipe
interactive segmenter (MagicTouch model), refined to full resolution with an
edge-aware guided upsampling."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from ..imaging.filters import guided_filter
from ..models import manager

_seg = None
_lock = threading.Lock()


def _segmenter():
    global _seg
    if _seg is None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        path = str(manager.require("magic-touch"))
        if hasattr(vision, "InteractiveSegmenterLegacy"):  # mediapipe >= 1.x
            opts = vision.InteractiveSegmenterLegacyOptions(base_options=BaseOptions(model_asset_path=path, delegate=BaseOptions.Delegate.CPU), output_confidence_masks=True)
            _seg = ("legacy", vision.InteractiveSegmenterLegacy.create_from_options(opts), vision.InteractiveSegmenterLegacyRegionOfInterest)
        else:  # mediapipe 0.10.x
            opts = vision.InteractiveSegmenterOptions(base_options=BaseOptions(model_asset_path=path, delegate=BaseOptions.Delegate.CPU), output_confidence_masks=True)
            _seg = ("v0", vision.InteractiveSegmenter.create_from_options(opts), vision.InteractiveSegmenterRegionOfInterest)
    return _seg


def smart_select(rgb: np.ndarray, points: list[tuple[float, float]], max_side: int = 1024) -> np.ndarray:
    """Return a soft object mask (H x W float) for the object(s) under the given pixel points."""
    import mediapipe as mp
    from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint

    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    u8 = np.ascontiguousarray((np.clip(small, 0, 1) * 255).astype(np.uint8))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=u8)
    acc = np.zeros(small.shape[:2], np.float32)
    with _lock:
        _, seg, ROI = _segmenter()
        for x, y in points:
            roi = ROI(format=ROI.Format.KEYPOINT, keypoint=NormalizedKeypoint(float(x) / w, float(y) / h))
            res = seg.segment(img, roi)
            m = np.asarray(res.confidence_masks[0].numpy_view(), np.float32)
            acc = np.maximum(acc, m.reshape(acc.shape))
    up = cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
    # edge-aware upsampling with the full-resolution image as guide
    r = max(2, int(4 / max(s, 1e-3) * 0.5))
    up = guided_filter(rgb.astype(np.float32), up, r, 1e-3)
    m = (up > 0.5).astype(np.uint8)
    # keep the components under the clicked points
    n, lab = cv2.connectedComponents(m)
    keep = np.zeros_like(m)
    for x, y in points:
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        if lab[yi, xi] > 0:
            keep[lab == lab[yi, xi]] = 1
    if keep.sum() == 0:
        keep = m
    return keep.astype(np.float32)
