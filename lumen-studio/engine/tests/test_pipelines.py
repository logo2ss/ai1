import cv2
import numpy as np
import pytest

from conftest import load, needs_depth, needs_face, needs_testdata


@needs_face
@needs_testdata
def test_face_detection_group_photo():
    from lumen_engine.face.landmarks import detect_faces

    faces = detect_faces(load("faces/bald_guys.jpg"))
    assert len(faces) >= 20  # multi-scale tiling finds small faces
    assert all(f.points.shape == (478, 3) for f in faces)


@needs_face
@needs_testdata
def test_face_swap_quality_and_locality():
    from lumen_engine.face.landmarks import detect_faces
    from lumen_engine.face.swap import FaceSwapSettings, swap_face

    S, T = load("faces/obama_small.jpg"), load("faces/lin-manuel-miranda.png")
    fs, ft = detect_faces(S)[0], detect_faces(T)[0]
    out, rep, dbg = swap_face(S, fs, T, ft, FaceSwapSettings())
    assert out.shape == T.shape
    assert rep["analysis"]["compatible"]
    q = rep["quality"]
    assert q["skin_delta_e_after"] < q["skin_delta_e_before"]
    assert 0.6 < q["seam_visibility"] < 1.3
    # nothing outside the processing region changes
    y0, y1, x0, x1 = dbg["roi"]
    outside = np.ones(T.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(out[outside], T[outside])


@needs_face
@needs_testdata
def test_face_incompatible_pose_is_refused():
    from lumen_engine.errors import IncompatibleInputsError
    from lumen_engine.face.landmarks import detect_faces
    from lumen_engine.face.swap import FaceSwapSettings, swap_face

    img = load("faces/2008_002079.jpg")
    faces = detect_faces(img)
    a = max(faces, key=lambda f: f.yaw)
    b = min(faces, key=lambda f: f.yaw)
    if a.yaw - b.yaw < 40:
        pytest.skip("test image does not contain strongly different poses")
    with pytest.raises(IncompatibleInputsError):
        swap_face(img, a, img, b, FaceSwapSettings())


@needs_testdata
def test_region_swap_object_and_area_modes():
    from lumen_engine.region.swap import RegionSwapSettings, swap_regions

    A = cv2.resize(load("scenes/fruits.jpg"), (320, 300), interpolation=cv2.INTER_AREA)
    B = cv2.resize(load("scenes/demo07.jpg"), (420, 280), interpolation=cv2.INTER_AREA)
    ma = np.zeros(A.shape[:2], np.float32)
    cv2.ellipse(ma, (130, 150), (60, 90), 20, 0, 360, 1, -1)  # irregular (rotated ellipse)
    mb = np.zeros(B.shape[:2], np.float32)
    cv2.circle(mb, (163, 94), 50, 1, -1)
    for mode in ("object", "area"):
        res = swap_regions(A, ma, B, mb, RegionSwapSettings(mode=mode))
        for key, img, mask in (("a", A, ma), ("b", B, mb)):
            out, rep, _ = res[key]
            assert out.shape == img.shape and np.isfinite(out).all()
            far = cv2.dilate(mask, np.ones((61, 61))) == 0
            assert np.array_equal(out[far], img[far]), f"{mode}/{key}: pixels outside the edit changed"
            assert rep["changed_fraction"] < 0.6


@needs_testdata
def test_region_small_selection():
    from lumen_engine.region.swap import RegionSwapSettings, insert_region

    A = cv2.resize(load("scenes/demo11.jpg"), (400, 260), interpolation=cv2.INTER_AREA)
    B = cv2.resize(load("scenes/demo08.jpg"), (400, 266), interpolation=cv2.INTER_AREA)
    ma = np.zeros(A.shape[:2], np.float32)
    ma[150:162, 200:214] = 1
    mb = np.zeros(B.shape[:2], np.float32)
    mb[200:212, 300:312] = 1
    out, rep, _ = insert_region(A, ma, B, mb, RegionSwapSettings())
    assert np.isfinite(out).all()


@needs_depth
@needs_testdata
def test_reconstruction_scene_container():
    from lumen_engine.recon.scene import ReconSettings, reconstruct

    rgb = cv2.resize(load("scenes/demo11.jpg"), (480, 312), interpolation=cv2.INTER_AREA)
    out = reconstruct(rgb, ReconSettings(quality="fast", detail_fusion=False))
    blob = out["scene"]
    assert blob[:4] == b"LSC1"
    h = out["header"]
    gw, gh = h["grid_w"], h["grid_h"]
    import json
    import struct

    hl = struct.unpack("<I", blob[4:8])[0]
    hdr = json.loads(blob[8 : 8 + hl])
    off, size = hdr["buffers"][0]
    assert size == gw * gh * 4
    rep = out["report"]
    assert 0 < rep["occluded_fraction"] < 0.35
    assert rep["main_triangles"] > 0.5 * 2 * (gw - 1) * (gh - 1)


@needs_face
@needs_testdata
def test_manual_face_box_and_rotated_face():
    """Manual selection recovers a face the automatic pass cannot see (rotated 90°)."""
    from lumen_engine.face.landmarks import detect_faces, detect_in_box

    img = load("faces/obama_small.jpg")
    rot = np.ascontiguousarray(np.rot90(img))  # face lying on its side
    auto = detect_faces(rot)
    f = detect_in_box(rot, (40, 90, 200, 230))
    assert f.source == "manual" and f.points.shape == (478, 3)
    x0, y0, x1, y1 = f.bbox
    assert 40 <= (x0 + x1) / 2 <= 200 and 60 <= (y0 + y1) / 2 <= 260
    assert len(auto) <= 1


@pytest.mark.skipif(not __import__("conftest").has_model("pose-landmarker"), reason="pose model not installed")
@needs_testdata
def test_body_reshape_person_layer_keeps_far_background():
    from lumen_engine.body.reshape import BodySettings, detect_body, reshape_body

    rgb = cv2.resize(load("faces/biden.jpg"), (485, 1102), interpolation=cv2.INTER_AREA)
    b = detect_body(rgb)
    src = b["lattice"]
    dst = src.copy()
    c = src[:, 2:3, 0]
    dst[2, 1:-1, 0] = c[2] + (src[2, 1:-1, 0] - c[2]) * 0.85  # slimmer waist
    out, rep = reshape_body(rgb, b["mask"], src, dst, BodySettings())
    assert rep["changed"] and out.shape == rgb.shape and np.isfinite(out).all()
    far = cv2.dilate((b["mask"] > 0.05).astype(np.uint8), np.ones((221, 221))) == 0
    assert far.sum() > 1000
    assert np.abs(out[far] - rgb[far]).max() < 1e-4  # background away from the body is untouched
