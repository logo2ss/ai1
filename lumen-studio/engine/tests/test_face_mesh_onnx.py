"""Compatibility face path with MediaPipe's face-mesh network converted to ONNX (no MediaPipe runtime)."""

import hashlib

import numpy as np
import pytest

from conftest import has_model, load, needs_testdata

needs_task = pytest.mark.skipif(not has_model("face-landmarker"), reason="face_landmarker.task not installed")
needs_compat = pytest.mark.skipif(
    not (has_model("face-landmarker") and has_model("face-detector-yunet")), reason="face models not installed"
)


@needs_task
def test_converter_is_deterministic_and_loads():
    import onnxruntime as ort

    from lumen_engine.face import tflite_onnx
    from lumen_engine.models import manager

    data = tflite_onnx.read_task_member(manager.model_path("face-landmarker"), "face_landmarks_detector.tflite")
    a, b = tflite_onnx.convert(data), tflite_onnx.convert(data)
    assert a == b and hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
    sess = ort.InferenceSession(a, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input": np.full((1, 256, 256, 3), 0.5, np.float32)})
    assert [o.shape for o in outs] == [(1, 1, 1, 1434), (1, 1, 1, 1), (1, 1)]
    meta = tflite_onnx.read_geometry_metadata(tflite_onnx.read_task_member(manager.model_path("face-landmarker"), "geometry_pipeline_metadata_landmarks.binarypb"))
    shipped = np.load(__import__("lumen_engine.face.meshnet", fromlist=["_DATA"])._DATA / "face_geometry.npz")
    assert np.array_equal(meta["triangles"], shipped["triangles"]) and np.allclose(meta["vertices"], shipped["vertices"])


@needs_compat
@needs_testdata
def test_compat_mesh_landmarks(monkeypatch):
    from lumen_engine.face import fallback

    monkeypatch.setenv("LUMEN_FACE_BACKEND", "compat")
    assert fallback.mesh_available()
    faces = fallback.detect_faces(load("faces/two_people.jpg"))
    assert len(faces) == 2
    for f in faces:
        assert f.points.shape == (478, 3) and f.extra["landmarks"] == "mesh-onnx" and f.extra["presence"] > 0.8
        z = f.points[:468, 2]
        assert np.ptp(z) > 0.15 * f.size  # real depth: nose tip in front of the face contour
        assert f.points[1, 2] < np.median(z)  # MediaPipe convention: smaller z = closer to the camera
        assert abs(f.yaw) < 40 and abs(f.pitch) < 40


@needs_compat
@needs_testdata
def test_compat_matches_mediapipe_reference(monkeypatch):
    """Landmarks agree with the MediaPipe path to well under 1.5% of the face size."""
    from lumen_engine.face import backend, fallback

    if backend.mediapipe_status() is not None:
        pytest.skip("MediaPipe cannot run on this computer")
    pytest.importorskip("mediapipe")
    from lumen_engine.face.landmarks import detect_faces as mp_detect

    img = load("faces/obama.jpg")
    ref = mp_detect(img)[0]
    monkeypatch.setenv("LUMEN_FACE_BACKEND", "compat")
    f = fallback.detect_faces(img)[0]
    err = np.linalg.norm(f.xy - ref.xy, axis=1).mean() / ref.size
    assert err < 0.015, err
    assert abs(f.yaw - ref.yaw) < 4 and abs(f.pitch - ref.pitch) < 4


@needs_compat
@needs_testdata
def test_manual_box_finds_rotated_face(monkeypatch):
    import cv2

    from lumen_engine.face import fallback

    monkeypatch.setenv("LUMEN_FACE_BACKEND", "compat")
    img = cv2.rotate(load("faces/obama_small.jpg"), cv2.ROTATE_90_CLOCKWISE)
    upright = fallback.detect_faces(load("faces/obama_small.jpg"))[0]
    x0, y0, x1, y1 = upright.bbox
    H = load("faces/obama_small.jpg").shape[0]
    box = (H - 1 - y1, x0, H - 1 - y0, x1)  # the same face box after rotating the image clockwise
    f = fallback.detect_in_box(img, box)
    assert f.source == "manual" and f.extra["presence"] > 0.5
    assert abs(abs(f.roll) - 90) < 20  # eye line is vertical in the rotated image


@needs_compat
@needs_testdata
def test_lbf_mode_still_available(monkeypatch):
    from lumen_engine.face import fallback

    if not has_model("face-landmarks-lbf"):
        pytest.skip("LBF model not installed")
    monkeypatch.setenv("LUMEN_FACE_BACKEND", "lbf")
    assert fallback.landmark_method() == "lbf"
    f = fallback.detect_faces(load("faces/obama_small.jpg"))[0]
    assert "landmarks68" in f.extra and f.points.shape == (478, 3)


def test_pose_of_canonical_face_is_frontal():
    from lumen_engine.face import meshnet
    from lumen_engine.face.landmarks import pose_from_matrix

    v = meshnet.canonical()["vertices"][:, :3].astype(np.float64)
    # frontal canonical face drawn in a 600x600 frame (image y down, MediaPipe z: smaller = closer)
    pts = np.column_stack([300 + v[:, 0] * 12, 300 - v[:, 1] * 12, -v[:, 2] * 12])
    yaw, pitch, roll = pose_from_matrix(meshnet.transformation_matrix(pts, 600, 600))
    assert abs(yaw) < 2 and abs(pitch) < 6 and abs(roll) < 2
