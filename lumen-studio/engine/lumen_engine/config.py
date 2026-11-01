"""Filesystem locations and runtime configuration.

All data lives under a single local directory (``LUMEN_HOME``), by default
``~/.lumen-studio``:

* ``models/``   - installed model files
* ``work/``     - temporary processing files (uploads, intermediate results)
* ``cache/``    - JIT/numba caches
* ``settings.json``
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


def lumen_home() -> Path:
    p = Path(os.environ.get("LUMEN_HOME", Path.home() / ".lumen-studio")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = Path(os.environ.get("LUMEN_MODELS_DIR", lumen_home() / "models")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def work_dir() -> Path:
    p = lumen_home() / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = lumen_home() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# numba must know where it may write its cache before it is imported.
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir() / "numba"))


@dataclass
class Settings:
    #: "auto", "cpu", "cuda", "directml", "coreml"
    device: str = "auto"
    #: number of CPU threads for inference/processing (0 = automatic)
    threads: int = 0
    #: longest side used for heavy processing (results are always produced at
    #: the original resolution; this limits intermediate solvers)
    max_working_resolution: int = 2400
    #: depth model used for 3D reconstruction
    depth_model: str = "depth-anything-v2-small"
    #: embed an "edited with Lumen Studio" note in exported files
    tag_exports: bool = True
    extra: dict = field(default_factory=dict)


_lock = threading.Lock()
_settings: Settings | None = None


def settings_path() -> Path:
    return lumen_home() / "settings.json"


def get_settings() -> Settings:
    global _settings
    with _lock:
        if _settings is None:
            s = Settings()
            try:
                data = json.loads(settings_path().read_text())
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            _settings = s
        return _settings


def update_settings(**changes) -> Settings:
    s = get_settings()
    with _lock:
        for k, v in changes.items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, v)
        settings_path().write_text(json.dumps(asdict(s), indent=2))
    return s
