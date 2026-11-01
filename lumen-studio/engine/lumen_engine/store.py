"""Local storage of uploaded images and results inside the work directory.

Images are stored losslessly (PNG, 16-bit when the source had more than
8 bits) and referenced by opaque ids. Nothing leaves the machine.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .config import work_dir
from .errors import NotFoundError
from .imaging.io import decode_image, encode_image

_lock = threading.Lock()
_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
_cache_order: list[str] = []
_CACHE_MAX = 12


def _dir() -> Path:
    p = work_dir() / "images"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _remember(image_id: str, rgb: np.ndarray, alpha: np.ndarray | None) -> None:
    with _lock:
        _cache[image_id] = (rgb, alpha)
        if image_id in _cache_order:
            _cache_order.remove(image_id)
        _cache_order.append(image_id)
        while len(_cache_order) > _CACHE_MAX:
            _cache.pop(_cache_order.pop(0), None)


def put(rgb: np.ndarray, alpha: np.ndarray | None = None, meta: dict | None = None) -> str:
    image_id = uuid.uuid4().hex[:16]
    d = _dir() / image_id
    d.mkdir()
    deep = bool(meta and meta.get("deep"))
    (d / "image.png").write_bytes(encode_image(rgb, "png", bit_depth=16 if deep else 8))
    if alpha is not None:
        (d / "alpha.png").write_bytes(encode_image(np.dstack([alpha] * 3), "png"))
    m = dict(meta or {})
    m.update(id=image_id, width=int(rgb.shape[1]), height=int(rgb.shape[0]), created=time.time())
    icc = m.pop("icc_profile", None)
    if icc:
        (d / "icc.bin").write_bytes(icc)
    (d / "meta.json").write_text(json.dumps(m))
    _remember(image_id, rgb, alpha)
    return image_id


def put_bytes(data: bytes, name: str = "") -> str:
    rgb, alpha, info = decode_image(data)
    return put(rgb, alpha, {"name": name, "format": info.get("format"), "icc_profile": info.get("icc_profile"), "deep": bool(info.get("deep"))})


def get(image_id: str) -> tuple[np.ndarray, np.ndarray | None]:
    if image_id in _cache:
        rgb, a = _cache[image_id]
        return rgb, a
    d = _dir() / image_id
    if not image_id.isalnum() or not (d / "image.png").is_file():
        raise NotFoundError("The image is no longer available. Please load it again.")
    rgb, _, _ = decode_image((d / "image.png").read_bytes())
    alpha = None
    if (d / "alpha.png").is_file():
        alpha = decode_image((d / "alpha.png").read_bytes())[0][..., 0]
    _remember(image_id, rgb, alpha)
    return rgb, alpha


def meta(image_id: str) -> dict:
    d = _dir() / image_id
    if not image_id.isalnum() or not (d / "meta.json").is_file():
        raise NotFoundError("Unknown image id")
    return json.loads((d / "meta.json").read_text())


def icc(image_id: str) -> bytes | None:
    p = _dir() / image_id / "icc.bin"
    return p.read_bytes() if p.is_file() else None


def put_file(content: bytes, suffix: str) -> str:
    """Store an arbitrary binary result (e.g. a GLB scene)."""
    fid = uuid.uuid4().hex[:16]
    d = work_dir() / "files"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{fid}{suffix}").write_bytes(content)
    return f"{fid}{suffix}"


def file_path(name: str) -> Path:
    p = work_dir() / "files" / name
    if "/" in name or "\\" in name or ".." in name or not p.is_file():
        raise NotFoundError("File not found")
    return p


def usage_bytes() -> int:
    return sum(f.stat().st_size for f in work_dir().rglob("*") if f.is_file())


def clear() -> None:
    with _lock:
        _cache.clear()
        _cache_order.clear()
    for sub in ("images", "files"):
        shutil.rmtree(work_dir() / sub, ignore_errors=True)


def cleanup_older_than(seconds: float = 7 * 86400) -> None:
    now = time.time()
    for sub in ("images", "files"):
        for p in (work_dir() / sub).glob("*") if (work_dir() / sub).exists() else []:
            try:
                if now - p.stat().st_mtime > seconds:
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError:
                pass
