"""QA regression tests for the user's machine: CPU without AVX -> compat face mode (no MediaPipe process).

Owner: QA. Fast; the fresh-install test runs in a subprocess so it sees a clean models folder.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import has_model, load, needs_testdata

needs_compat = pytest.mark.skipif(
    not (has_model("face-detector-yunet") and (has_model("face-mesh-onnx") or has_model("face-landmarks-lbf"))), reason="compat face models not installed"
)
ENGINE = Path(__file__).resolve().parents[1]


@needs_testdata
@needs_compat
def test_qa_compat_alias_never_uses_mediapipe(monkeypatch):
    """Regression (QA 05:26 -> fixed by F4 06:25): LUMEN_FACE_BACKEND=compat used MediaPipe on CPUs that have AVX."""
    from lumen_engine.face import backend

    monkeypatch.setenv("LUMEN_FACE_BACKEND", "compat")
    monkeypatch.setattr(backend, "mediapipe_status", lambda: None)  # as if MediaPipe could run here

    def no_mediapipe(*_a, **_k):
        raise AssertionError("compat mode must not start MediaPipe")

    monkeypatch.setattr(backend.mpworker, "call", no_mediapipe)
    faces, info = backend.detect(load("faces/obama_small.jpg"))
    assert info["backend"] == "opencv"
    assert len(faces) >= 1 and faces[0].xy.shape[0] >= 468


@needs_testdata
@needs_compat
@pytest.mark.skipif(not has_model("face-landmarker"), reason="face_landmarker.task not installed")
def test_qa_compat_fresh_install_without_converted_face_mesh(tmp_path):
    """A fresh user setup has face_landmarker.task but no converted face_landmarks_detector.onnx yet:
    the first detection on a no-AVX CPU must still work (convert on first use, or fall back to LBF)."""
    src = Path(os.environ["LUMEN_MODELS_DIR"])
    md = tmp_path / "models"
    md.mkdir()
    for f in src.iterdir():
        if f.is_file() and f.name != "face_landmarks_detector.onnx":
            (md / f.name).symlink_to(f)
    code = (
        "import json\n"
        "from lumen_engine.imaging.io import decode_image\n"
        "from lumen_engine.face import backend\n"
        f"rgb = decode_image(open({str(Path(__file__).resolve().parents[2] / 'testdata/faces/obama_small.jpg')!r}, 'rb').read())[0]\n"
        "faces, info = backend.detect(rgb)\n"
        "print(json.dumps({'backend': info.get('backend'), 'n': len(faces), 'landmarks': (faces[0].extra or {}).get('landmarks') if faces else None}))\n"
    )
    env = dict(os.environ, LUMEN_MODELS_DIR=str(md), LUMEN_HOME=str(tmp_path / "home"), LUMEN_ASSUME_NO_AVX="1", LUMEN_FACE_BACKEND="auto")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180, cwd=str(ENGINE))
    assert r.returncode == 0, r.stderr[-2500:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["backend"] == "opencv" and out["n"] >= 1, out
    # the precise MediaPipe mesh (ONNX) must be derived on first use, not the 4x less accurate LBF fallback
    assert out["landmarks"] == "mesh-onnx", out
    assert (md / "face_landmarks_detector.onnx").stat().st_size > 1_000_000
