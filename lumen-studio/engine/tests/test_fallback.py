"""MediaPipe-free face path (for CPUs/VMs without AVX) and automatic switching."""

import numpy as np
import pytest

from conftest import has_model, load, needs_testdata

needs_fallback = pytest.mark.skipif(not (has_model("face-detector-yunet") and has_model("face-landmarks-lbf")), reason="fallback face models not installed")


@pytest.fixture
def fresh_backend(monkeypatch):
    from lumen_engine import mpworker
    from lumen_engine.face import backend

    def reset():
        backend._disabled_reason = None
        backend._cpu_checked = False
        mpworker.shutdown()

    reset()
    yield backend
    reset()


@needs_fallback
@needs_testdata
def test_opencv_path_detects_and_swaps(monkeypatch, fresh_backend):
    from lumen_engine.face.swap import FaceSwapSettings, swap_face

    monkeypatch.setenv("LUMEN_FACE_BACKEND", "opencv")
    S, T = load("faces/obama_small.jpg"), load("faces/lin-manuel-miranda.png")
    fs_list, info = fresh_backend.detect(S)
    ft_list, _ = fresh_backend.detect(T)
    assert info["backend"] == "opencv" and len(fs_list) == 1 and len(ft_list) == 1
    fs, ft = fs_list[0], ft_list[0]
    assert fs.points.shape == (478, 3) and abs(fs.yaw) < 20
    out, rep, dbg = swap_face(S, fs, T, ft, FaceSwapSettings())
    assert rep["analysis"]["compatible"]
    assert rep["quality"]["skin_delta_e_after"] < rep["quality"]["skin_delta_e_before"]
    assert 0.5 < rep["quality"]["seam_visibility"] < 1.4


@needs_fallback
@needs_testdata
def test_opencv_path_group_photo_and_manual_box(monkeypatch, fresh_backend):
    monkeypatch.setenv("LUMEN_FACE_BACKEND", "opencv")
    faces, _ = fresh_backend.detect(load("faces/bald_guys.jpg"))
    assert len(faces) >= 15
    img = load("faces/obama.jpg")
    f, _ = fresh_backend.detect(img, (300, 150, 640, 560))
    assert f.source == "manual" and f.points.shape == (478, 3)


@needs_fallback
@needs_testdata
def test_mediapipe_crash_switches_to_fallback_automatically(monkeypatch, fresh_backend):
    from lumen_engine.errors import ProcessingError

    monkeypatch.setenv("LUMEN_SIMULATE_MP_CRASH", "1")  # helper dies with SIGILL, like on a CPU without AVX
    faces, info = fresh_backend.detect(load("faces/two_people.jpg"))
    assert info["backend"] == "opencv" and len(faces) == 2
    assert "illegal instruction" in info["note"]
    assert fresh_backend.mediapipe_status() is not None
    # MediaPipe-only features now fail fast with a clear message instead of crashing again
    with pytest.raises(ProcessingError) as e:
        fresh_backend.run_mediapipe("lumen_engine.mptasks.detect_body", np.zeros((64, 64, 3), np.uint8), what="The body editor")
    assert e.value.details.get("mediapipe_unavailable") and "host-passthrough" in e.value.hint


def test_cpu_without_avx_disables_mediapipe(monkeypatch, fresh_backend):
    monkeypatch.setenv("LUMEN_ASSUME_NO_AVX", "1")
    assert "AVX" in fresh_backend.mediapipe_status()


@needs_fallback
@needs_testdata
def test_api_face_flow_in_compatibility_mode(monkeypatch, fresh_backend):
    import time

    from fastapi.testclient import TestClient

    from conftest import TESTDATA
    from lumen_engine.server.app import create_app

    monkeypatch.setenv("LUMEN_ASSUME_NO_AVX", "1")
    c = TestClient(create_app(token="fb"))
    H = {"X-Lumen-Token": "fb"}
    up = lambda p: c.post("/api/images", headers=H, files={"file": (p, (TESTDATA / p).read_bytes(), "image/jpeg")}).json()  # noqa: E731
    s, t = up("faces/biden.jpg"), up("faces/obama.jpg")
    d = c.post("/api/face/detect", headers=H, json={"image_id": s["id"]}).json()
    assert d["backend"] == "opencv" and len(d["faces"]) == 1 and "AVX" in d["note"]
    assert c.get("/api/system", headers=H).json()["mediapipe"]["usable"] is False
    j = c.post("/api/face/swap", headers=H, json={"source": {"image_id": s["id"]}, "target": {"image_id": t["id"]}, "settings": {"force": True}}).json()
    for _ in range(300):
        st = c.get(f"/api/jobs/{j['job_id']}", headers=H).json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert st["state"] == "done", st
    r = c.post("/api/region/smart-select", headers=H, json={"image_id": s["id"], "points": [[480, 700]]})
    assert r.status_code == 200 and r.headers["x-lumen-method"] in ("silueta", "graphcut")  # works without MediaPipe


needs_seg = pytest.mark.skipif(not (has_model("segment-silueta") and has_model("segment-human")), reason="segmentation models not installed")


@needs_seg
@needs_testdata
def test_smart_select_without_mediapipe():
    from lumen_engine.region.smart_select_fallback import smart_select

    rgb = load("scenes/demo07.jpg")
    h, w = rgb.shape[:2]
    m, method = smart_select(rgb, [(0.39 * w, 0.33 * h)])  # click on the basketball
    assert method == "silueta"
    assert 0.01 < m.mean() < 0.2  # the ball, not the whole picture
    assert m[int(0.33 * h), int(0.39 * w)] > 0.5


@needs_seg
@needs_testdata
def test_body_editor_and_body3d_in_compatibility_mode(monkeypatch, fresh_backend):
    from lumen_engine.body.reshape import BodySettings, detect_body_fallback, reshape_body
    from lumen_engine.recon.body3d import Body3DSettings, build_body3d

    rgb = load("faces/obama.jpg")
    b = detect_body_fallback(rgb)
    rows = b["lattice"][:, 0, 1]
    assert np.all(np.diff(rows) > 0) and b["method"] == "segmentation"
    dst = b["lattice"].copy()
    dst[2, 1:-1, 0] = dst[2, 2:3, 0] + (dst[2, 1:-1, 0] - dst[2, 2:3, 0]) * 0.9
    out, rep = reshape_body(rgb, b["mask"], b["lattice"], dst, BodySettings(), preview=True)
    assert rep["changed"]
    if has_model("depth-anything-v2-small"):
        res = build_body3d(rgb, b["mask"], Body3DSettings(detail=120))
        hdr = res["header"]
        assert res["body"][:4] == b"LBD1" and hdr["vertices"] > 0.3 * hdr["cols"] * hdr["rows"]
