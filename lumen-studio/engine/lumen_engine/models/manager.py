"""Local model management: status, explicit download, import, verification.

This is the *only* module of the engine that is allowed to open a network
connection, and it only does so from :func:`download`, which is triggered by
an explicit user action (the "Install" button or the setup script). Inference
code calls :func:`require`, which never downloads - it raises
:class:`ModelNotInstalledError` so the UI can tell the user what to install.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

from ..config import models_dir
from ..errors import LumenError, ModelNotInstalledError
from .registry import MODELS, ModelSpec, get_spec

_progress: dict[str, dict] = {}
_lock = threading.Lock()


def model_path(model_id: str) -> Path:
    return models_dir() / get_spec(model_id).filename


def is_installed(model_id: str) -> bool:
    p = model_path(model_id)
    return p.is_file() and p.stat().st_size > 0


def require(model_id: str) -> Path:
    """Return the local path of an installed model or raise a clear error."""
    spec = get_spec(model_id)
    p = model_path(model_id)
    if not p.is_file():
        raise ModelNotInstalledError(model_id, spec.name)
    return p


def status() -> list[dict]:
    out = []
    for spec in MODELS.values():
        p = model_path(spec.id)
        d = spec.to_dict()
        d["installed"] = p.is_file()
        d["path"] = str(p)
        d["installed_size"] = p.stat().st_size if p.is_file() else 0
        d["progress"] = _progress.get(spec.id)
        out.append(d)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate(spec: ModelSpec, path: Path) -> None:
    if spec.sha256:
        digest = _sha256(path)
        if digest != spec.sha256:
            raise LumenError(
                f"Checksum mismatch for {spec.name}.",
                hint="The file is corrupted or is a different version. Delete it and install again.",
                expected=spec.sha256,
                actual=digest,
            )
    if path.suffix == ".onnx":
        # Loading proves the file is a complete, readable ONNX graph.
        import onnxruntime as ort

        try:
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover - depends on file
            raise LumenError(f"{spec.name} could not be loaded: {exc}") from exc


def _sources(spec: ModelSpec) -> list[tuple[str, str | None]]:
    """(url, sha256) pairs to try in order: the main URL, then the mirrors in ``spec.extra['mirrors']``
    (a URL string for the same file, or {"url", "sha256"} for an equivalent file with its own checksum)."""
    out = [(spec.url, spec.sha256)]
    for m in (spec.extra or {}).get("mirrors", ()):
        if isinstance(m, str):
            out.append((m, spec.sha256))
        elif isinstance(m, dict) and m.get("url"):
            out.append((m["url"], m.get("sha256", spec.sha256)))
    return out


def _derive(spec: ModelSpec) -> None:
    """Models produced locally from another installed model (e.g. a TFLite -> ONNX conversion); no network."""
    from importlib import import_module

    src = spec.extra["derived_from"]
    if not is_installed(src):
        download(src)
    _progress[spec.id] = {"state": "converting", "received": 0, "total": spec.size_bytes, "error": None}
    dest = model_path(spec.id)
    try:
        mod, fn = spec.extra["converter"].split(":")
        getattr(import_module(mod), fn)(model_path(src), dest)
        _validate(spec, dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        _progress[spec.id] = {"state": "error", "error": f"Conversion failed: {exc}"}
        raise LumenError(f"Preparing {spec.name} failed: {exc}") from exc
    _progress[spec.id] = {"state": "installed", "received": dest.stat().st_size, "total": dest.stat().st_size, "error": None}


def download(model_id: str) -> None:
    """Download a model (blocking), trying its mirrors when the main URL fails. Progress is exposed
    through :func:`status`."""
    spec = get_spec(model_id)
    if spec.extra.get("derived_from"):
        return _derive(spec)
    sources = _sources(spec)
    for i, (url, sha) in enumerate(sources):
        try:
            _download_from(replace(spec, url=url, sha256=sha), i > 0)
            return
        except Exception:
            if i == len(sources) - 1:
                raise


def _download_from(spec: ModelSpec, fresh: bool = False) -> None:
    model_id = spec.id
    dest = model_path(model_id)
    part = dest.with_suffix(dest.suffix + ".part")
    with _lock:
        if _progress.get(model_id, {}).get("state") == "downloading":
            return
        _progress[model_id] = {"state": "downloading", "received": 0, "total": spec.size_bytes, "error": None}
    if fresh:  # a partial file from another source cannot be resumed
        part.unlink(missing_ok=True)
    try:
        start = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(spec.url, headers={"User-Agent": "LumenStudio-ModelManager/1.0"})
        if start:
            req.add_header("Range", f"bytes={start}-")
        with urllib.request.urlopen(req, timeout=60) as resp:
            if start and resp.status != 206:
                start = 0  # server ignored the range request
            total = int(resp.headers.get("Content-Length", 0)) + start or spec.size_bytes
            _progress[model_id].update(total=total, received=start)
            with open(part, "ab" if start else "wb") as f:
                last = time.time()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    _progress[model_id]["received"] += len(chunk)
                    if time.time() - last > 0.2:
                        last = time.time()
        _progress[model_id]["state"] = "verifying"
        _validate(spec, part)
        part.replace(dest)
        _progress[model_id] = {"state": "installed", "received": dest.stat().st_size, "total": dest.stat().st_size, "error": None}
    except LumenError as exc:
        part.unlink(missing_ok=True)
        _progress[model_id] = {"state": "error", "error": exc.message}
        raise
    except Exception as exc:
        _progress[model_id] = {"state": "error", "error": f"Download failed: {exc}"}
        raise LumenError(
            f"Downloading {spec.name} failed: {exc}",
            hint="Check the internet connection, or download the file manually from "
            f"{spec.url} and use 'Import file'.",
        ) from exc


def import_file(model_id: str, source: Path) -> None:
    """Install a model from a file the user obtained separately (air-gapped setups)."""
    spec = get_spec(model_id)
    source = Path(source)
    if not source.is_file():
        raise LumenError(f"File not found: {source}")
    tmp = model_path(model_id).with_suffix(".import")
    shutil.copyfile(source, tmp)
    try:
        _validate(spec, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(model_path(model_id))
    _progress[model_id] = {"state": "installed", "error": None}


def remove(model_id: str) -> None:
    model_path(model_id).unlink(missing_ok=True)
    _progress.pop(model_id, None)
    from ..hardware import clear_sessions

    clear_sessions()
