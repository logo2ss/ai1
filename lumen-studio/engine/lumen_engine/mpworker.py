"""Run MediaPipe-based work in an isolated helper process.

MediaPipe is native code. On some machines - typically virtual machines with
virtual GPUs or unusual graphics drivers - it can crash the whole process
(segmentation fault / abort) instead of raising a Python error. Running it in
a separate, persistent helper process means such a crash can never take the
engine (and the open UI) down: the caller gets a clear error, and the helper
is restarted for the next request.

The helper also hides the system's EGL/GPU drivers from MediaPipe: inference
runs on the CPU anyway, and this avoids crashes inside virtual GPU drivers.
Set ``LUMEN_MEDIAPIPE_GPU=1`` to disable this.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool

from .errors import ProcessingError

log = logging.getLogger("lumen.mpworker")
_pool: ProcessPoolExecutor | None = None
_lock = threading.Lock()

CRASH_HINT = (
    "Nothing was sent to the internet; a native library crashed on this computer. "
    "On Linux make sure the graphics libraries are installed (Arch: sudo pacman -S --needed libglvnd mesa). "
    "The engine kept running, so you can retry; details are printed in the terminal running Lumen Studio."
)


def _init_worker() -> None:
    import faulthandler

    faulthandler.enable()  # a native crash prints a stack trace to the terminal
    os.environ.setdefault("GLOG_minloglevel", "2")
    if os.environ.get("LUMEN_MEDIAPIPE_GPU") != "1":
        from .config import cache_dir

        empty = cache_dir() / "no-egl-vendors"
        empty.mkdir(parents=True, exist_ok=True)
        os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = str(empty)
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"


def _run(target: str, args: tuple, kwargs: dict):
    module, func = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), func)(*args, **kwargs)


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=_init_worker)
    return _pool


def _exit_code() -> int | None:
    try:
        for p in list(getattr(_pool, "_processes", {}).values()):
            p.join(timeout=2)
            if p.exitcode is not None:
                return p.exitcode
    except Exception:  # pragma: no cover
        pass
    return None


def _describe(code: int | None) -> tuple[str, str]:
    import signal

    if code is None:
        return "helper stopped", "MediaPipe stopped unexpectedly on this computer."
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        if name == "SIGILL":
            return name, "MediaPipe stopped with an 'illegal instruction': this processor or virtual machine lacks CPU instructions (AVX) it needs."
        if name == "SIGKILL":
            return name, "MediaPipe was killed by the system (usually: out of memory)."
        return name, f"MediaPipe crashed on this computer ({name})."
    return f"exit code {code}", f"MediaPipe stopped on this computer (exit code {code})."


def _reset() -> None:
    global _pool
    if _pool is not None:
        try:
            for p in list(getattr(_pool, "_processes", {}).values()):
                p.kill()
        except Exception:  # pragma: no cover - best effort
            pass
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = None


def call(target: str, *args, timeout: float = 600, what: str = "MediaPipe", **kwargs):
    """Call ``module.function(*args, **kwargs)`` in the helper process."""
    with _lock:
        try:
            return _get_pool().submit(_run, target, args, kwargs).result(timeout=timeout)
        except BrokenProcessPool:
            code = _exit_code()
            sig, reason = _describe(code)
            log.error("%s helper process crashed (%s) while running %s", what, sig, target)
            _reset()
            raise ProcessingError(f"{what} crashed while processing this image ({sig}).", hint=CRASH_HINT, crashed=True, signal=sig, reason=reason) from None
        except FutureTimeout:
            _reset()
            raise ProcessingError(f"{what} did not finish within {int(timeout)} s.", hint="The computer may be overloaded; try a smaller image.") from None


def shutdown() -> None:
    with _lock:
        _reset()
