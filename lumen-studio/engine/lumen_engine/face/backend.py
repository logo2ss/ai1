"""Chooses how faces are detected: MediaPipe (most precise) or the OpenCV fallback.

MediaPipe's prebuilt x86-64 binaries need the AVX instruction set (verified by
CPU emulation: they die with "illegal instruction" on a Nehalem-class CPU and
work on Sandy Bridge). Many virtual machines hide AVX from the guest. Rather
than crashing, the engine:

* checks the CPU at start-up and skips MediaPipe when AVX is missing,
* remembers if MediaPipe crashed for any other reason (in its isolated helper),
* and then uses the OpenCV path (YuNet + LBF landmarks expanded to the same
  478-point mesh), telling the user why.

``LUMEN_FACE_BACKEND=auto|mediapipe|opencv`` overrides the choice (``compat`` is an
alias of ``opencv``; ``lbf`` forces the old LBF landmarks instead of the ONNX face mesh).

In compatibility mode the landmarks come from MediaPipe's own face-mesh network
converted to ONNX (``meshnet.py``; same 478 points and depth, run by ONNX Runtime,
which needs no AVX). Only if that model is unavailable are LBF's 68 points
densified with a template.
"""

from __future__ import annotations

import logging
import os
import platform
import threading

import cv2
import numpy as np

from .. import mpworker
from ..errors import LumenError, ProcessingError
from ..models import manager

log = logging.getLogger("lumen.face.backend")
_lock = threading.Lock()
_disabled_reason: str | None = None
_cpu_checked = False

VM_HINT = (
    "If this is a virtual machine, set its CPU model to 'host-passthrough' / 'Copy host CPU configuration' "
    "(virt-manager, GNOME Boxes, Proxmox) so the guest sees AVX; then the MediaPipe-based features work too."
)


def _cpu_reason() -> str | None:
    if os.environ.get("LUMEN_ASSUME_NO_AVX") == "1":  # testing hook
        return "This processor (or virtual machine) does not provide the AVX instructions MediaPipe requires."
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    if "avx" not in flags:
                        return "This processor (or virtual machine) does not provide the AVX instructions MediaPipe requires."
                    return None
    except OSError:
        return None  # not Linux: cannot tell, try MediaPipe (crashes are contained anyway)
    return None


def mediapipe_status() -> str | None:
    """None if MediaPipe can be used, otherwise a human-readable reason."""
    global _cpu_checked, _disabled_reason
    with _lock:
        if not _cpu_checked:
            _cpu_checked = True
            r = _cpu_reason()
            if r:
                _disabled_reason = r
                log.warning("MediaPipe disabled: %s", r)
        return _disabled_reason


def disable_mediapipe(reason: str) -> None:
    global _disabled_reason
    with _lock:
        _disabled_reason = reason
    log.warning("MediaPipe disabled for this session: %s", reason)


def run_mediapipe(target: str, *args, what: str):
    """Run a MediaPipe task in the helper process; disable MediaPipe if it crashes."""
    reason = mediapipe_status()
    if reason:
        raise ProcessingError(f"{what} needs MediaPipe, which cannot run on this computer. {reason}", hint=VM_HINT, mediapipe_unavailable=True)
    try:
        return mpworker.call(target, *args, what=what)
    except ProcessingError as e:
        if e.details.get("crashed"):
            disable_mediapipe(e.details.get("reason") or "MediaPipe crashed on this computer.")
        raise


def _fallback_ready() -> bool:
    if not manager.is_installed("face-detector-yunet"):
        return False
    from . import fallback

    return fallback.mesh_available() or manager.is_installed("face-landmarks-lbf")


def compat_info() -> dict:
    """What the compatibility path uses (for /api/system and detection responses)."""
    from . import fallback

    ready = _fallback_ready()
    method = fallback.landmark_method() if ready else None
    return {
        "ready": ready,
        "landmarks": method,
        "description": {
            "mesh-onnx": "MediaPipe face mesh (478 points with depth) via ONNX Runtime",
            "lbf": "OpenCV LBF 68 points densified to the 478-point mesh (less precise)",
        }.get(method or "", "fallback face models are not installed"),
    }


def detect(rgb: np.ndarray, box=None) -> tuple:
    """Detect faces (or the face in ``box``). Returns (faces_or_face, info)."""
    mode = os.environ.get("LUMEN_FACE_BACKEND", "auto").lower()
    if mode in ("compat", "lbf"):
        mode = "opencv"
    reason = None if mode == "opencv" else mediapipe_status()
    if mode != "opencv" and reason is None:
        manager.require("face-landmarker")
        u8 = cv2.convertScaleAbs(rgb, alpha=255.0)
        try:
            if box is None:
                res = mpworker.call("lumen_engine.mptasks.detect_faces", u8, what="Face detection (MediaPipe)")
            else:
                res = mpworker.call("lumen_engine.mptasks.detect_in_box", u8, tuple(box), what="Face detection (MediaPipe)")
            return res, {"backend": "mediapipe"}
        except ProcessingError as e:
            if not e.details.get("crashed"):
                raise
            reason = e.details.get("reason") or "MediaPipe crashed on this computer."
            disable_mediapipe(reason)
            if mode == "mediapipe" or not _fallback_ready():
                raise
    if not _fallback_ready():
        raise LumenError(
            "Face detection cannot run: MediaPipe is unavailable here and the fallback models are not installed."
            if reason
            else "The fallback face models are not installed.",
            hint="Install 'YuNet Face Detector' and 'OpenCV LBF Facial Landmarks' in Models (or run: python scripts/download_models.py).",
        )
    from . import fallback

    res = fallback.detect_faces(rgb) if box is None else fallback.detect_in_box(rgb, box)
    method = fallback.landmark_method()
    if method == "mesh-onnx":
        what = "Using the compatibility face path (YuNet detector + MediaPipe's face-mesh model run by ONNX Runtime; same landmark precision)"
        note = f"{what} because MediaPipe cannot run here: {reason} {VM_HINT}" if reason else f"{what}."
    else:
        note = (
            f"Using the compatibility face detector because MediaPipe cannot run here: {reason} Landmarks are slightly less precise. {VM_HINT}"
            if reason
            else "Using the compatibility face detector (OpenCV)."
        )
    return res, {"backend": "opencv", "landmarks": method, "note": note}
