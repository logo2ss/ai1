"""Hardware detection and inference-session creation.

Inference uses ONNX Runtime, which provides a single, well-supported runtime
for CPU, NVIDIA CUDA/TensorRT, DirectML (any DX12 GPU on Windows) and CoreML
(Apple Silicon). Execution providers are discovered at runtime; nothing here
contacts the network.
"""

from __future__ import annotations

import os
import platform
import threading
from functools import lru_cache

import onnxruntime as ort

from .config import get_settings

_PROVIDER_FOR_DEVICE = {
    "cuda": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    "directml": ["DmlExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider"],
    "cpu": [],
}


def _total_ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (ValueError, OSError):
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024**3
        except Exception:  # pragma: no cover - best effort only
            return None
    return None


def available_devices() -> list[str]:
    providers = set(ort.get_available_providers())
    devices = ["cpu"]
    for dev, provs in _PROVIDER_FOR_DEVICE.items():
        if dev != "cpu" and any(p in providers for p in provs):
            devices.append(dev)
    return devices


def hardware_info() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(_total_ram_gb() or 0, 1),
        "onnxruntime": ort.__version__,
        "onnx_providers": ort.get_available_providers(),
        "devices": available_devices(),
        "selected_device": resolve_device(),
    }


def resolve_device() -> str:
    wanted = get_settings().device
    devs = available_devices()
    if wanted != "auto":
        return wanted if wanted in devs else "cpu"
    for pref in ("cuda", "directml", "coreml"):
        if pref in devs:
            return pref
    return "cpu"


def providers_for(device: str | None = None) -> list[str]:
    device = device or resolve_device()
    available = set(ort.get_available_providers())
    chosen = [p for p in _PROVIDER_FOR_DEVICE.get(device, []) if p in available]
    return chosen + ["CPUExecutionProvider"]


_session_lock = threading.Lock()


@lru_cache(maxsize=8)
def _cached_session(path: str, providers: tuple[str, ...], threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads
    try:
        return ort.InferenceSession(path, sess_options=opts, providers=list(providers))
    except Exception:
        # A broken GPU driver must never make the app unusable: retry on CPU.
        if providers != ("CPUExecutionProvider",):
            return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        raise


def onnx_session(path: str) -> ort.InferenceSession:
    with _session_lock:
        threads = int(os.environ.get("LUMEN_THREADS") or get_settings().threads)
        return _cached_session(str(path), tuple(providers_for()), threads)


def clear_sessions() -> None:
    _cached_session.cache_clear()
