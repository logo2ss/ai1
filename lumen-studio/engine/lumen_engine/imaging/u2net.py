"""U2-Net family segmentation (ONNX Runtime) - runs on every CPU, no MediaPipe needed.

Used for one-click object selection (``silueta``, a compact general salient
object model) and person segmentation (``u2net_human_seg``).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..hardware import onnx_session
from ..models import manager

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def predict(rgb: np.ndarray, model_id: str) -> np.ndarray:
    """Soft mask in [0, 1] at the input resolution."""
    sess = onnx_session(str(manager.require(model_id)))
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb.astype(np.float32), (320, 320), interpolation=cv2.INTER_AREA if max(h, w) > 320 else cv2.INTER_LINEAR)
    x = x / max(float(x.max()), 1e-6)
    x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    lo, hi = float(out.min()), float(out.max())
    out = (out - lo) / max(hi - lo, 1e-6)
    return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def person_mask(rgb: np.ndarray) -> np.ndarray:
    """Person segmentation; edges snapped to the image with a guided filter."""
    from .filters import guided_filter

    m = predict(rgb, "segment-human")
    r = max(2, int(max(rgb.shape[:2]) / 250))
    return np.clip(guided_filter(rgb.astype(np.float32), m, r, 1e-3), 0, 1)
