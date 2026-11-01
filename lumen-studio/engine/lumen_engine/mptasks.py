"""Entry points executed inside the isolated MediaPipe helper process.

Images cross the process boundary as uint8 (4x less data than float32) and
are converted back here.
"""

from __future__ import annotations

import os
import signal

import numpy as np


def _maybe_simulate_crash() -> None:
    # testing hook: reproduce what MediaPipe does on a CPU without AVX
    if os.environ.get("LUMEN_SIMULATE_MP_CRASH") == "1":
        os.kill(os.getpid(), signal.SIGILL)


def _f32(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0


def detect_faces(u8: np.ndarray):
    _maybe_simulate_crash()
    from .face.landmarks import detect_faces as _detect

    return _detect(_f32(u8))


def detect_in_box(u8: np.ndarray, box):
    _maybe_simulate_crash()
    from .face.landmarks import detect_in_box as _detect

    return _detect(_f32(u8), box)


def detect_body(u8: np.ndarray):
    _maybe_simulate_crash()
    from .body.reshape import detect_body as _detect

    return _detect(_f32(u8))


def smart_select(u8: np.ndarray, points):
    _maybe_simulate_crash()
    from .region.smart_select import smart_select as _sel

    return _sel(_f32(u8), points)
