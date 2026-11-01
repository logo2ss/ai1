import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# isolate work files from the user's real data directory, but reuse installed models
_home = tempfile.mkdtemp(prefix="lumen-test-")
os.environ["LUMEN_HOME"] = _home
os.environ.setdefault("LUMEN_MODELS_DIR", str(Path.home() / ".lumen-studio" / "models"))

ROOT = Path(__file__).resolve().parents[2]
TESTDATA = ROOT / "testdata"


def load(rel: str) -> np.ndarray:
    from lumen_engine.imaging.io import decode_image

    return decode_image((TESTDATA / rel).read_bytes())[0]


def has_model(mid: str) -> bool:
    from lumen_engine.models import manager

    return manager.is_installed(mid)


needs_face = pytest.mark.skipif(not has_model("face-landmarker"), reason="face-landmarker model not installed")
needs_depth = pytest.mark.skipif(not has_model("depth-anything-v2-small"), reason="depth model not installed")
needs_testdata = pytest.mark.skipif(not (TESTDATA / "faces").is_dir(), reason="testdata not downloaded (scripts/fetch_testdata.py)")
