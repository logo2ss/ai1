import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import TESTDATA, needs_depth, needs_face, needs_testdata


@pytest.fixture(scope="module")
def client():
    from lumen_engine.server.app import create_app

    return TestClient(create_app(token="t0k"))


H = {"X-Lumen-Token": "t0k"}


def _png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def _wait(client, job_id):
    import time

    for _ in range(600):
        j = client.get(f"/api/jobs/{job_id}", headers=H).json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise TimeoutError


def test_auth_required(client):
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-Lumen-Token": "wrong"}).status_code == 401
    assert client.get("/api/health", headers=H).json()["offline"] is True


def test_upload_rejects_garbage(client):
    r = client.post("/api/images", headers=H, files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400 and r.json()["code"] == "invalid_input"


def test_upload_preview_export(client):
    img = (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8)
    r = client.post("/api/images", headers=H, files={"file": ("a.png", _png(img), "image/png")}).json()
    assert r["width"] == 96 and r["height"] == 64
    assert client.get(f"/api/images/{r['id']}?max=32&token=t0k").status_code == 200
    e = client.get(f"/api/images/{r['id']}/export?format=tiff", headers=H)
    assert e.status_code == 200 and e.headers["content-type"] == "image/tiff"


def test_models_listing(client):
    models = client.get("/api/models", headers=H).json()
    ids = {m["id"] for m in models}
    assert {"face-landmarker", "depth-anything-v2-small", "lama"} <= ids


@needs_testdata
def test_region_swap_job(client):
    up = lambda p: client.post("/api/images", headers=H, files={"file": (p, (TESTDATA / p).read_bytes(), "image/jpeg")}).json()  # noqa: E731
    a, b = up("scenes/home.jpg"), up("scenes/fruits.jpg")
    ma = np.zeros((a["height"], a["width"]), np.uint8)
    ma[20:120, 30:200] = 255
    mb = np.zeros((b["height"], b["width"]), np.uint8)
    mb[100:300, 100:300] = 255
    payload = json.dumps({"image_a": a["id"], "image_b": b["id"], "settings": {"mode": "area"}, "direction": "both"})
    r = client.post("/api/region/swap", headers=H, data={"payload": payload}, files={"mask_a": ("a.png", _png(ma), "image/png"), "mask_b": ("b.png", _png(mb), "image/png")})
    j = _wait(client, r.json()["job_id"])
    assert j["state"] == "done", j
    assert set(j["result"]) == {"a", "b"}


@needs_face
@needs_testdata
def test_face_api_flow(client):
    up = lambda p: client.post("/api/images", headers=H, files={"file": (p, (TESTDATA / p).read_bytes(), "image/jpeg")}).json()  # noqa: E731
    s, t = up("faces/biden.jpg"), up("faces/obama.jpg")
    assert len(client.post("/api/face/detect", headers=H, json={"image_id": s["id"]}).json()["faces"]) == 1
    an = client.post("/api/face/analyze", headers=H, json={"source": {"image_id": s["id"]}, "target": {"image_id": t["id"]}}).json()
    assert "score" in an
    r = client.post("/api/face/swap", headers=H, json={"source": {"image_id": s["id"]}, "target": {"image_id": t["id"]}, "settings": {"force": True}})
    j = _wait(client, r.json()["job_id"])
    assert j["state"] == "done", j
    assert client.get(f"/api/images/{j['result']['image_id']}/export?format=png", headers=H).status_code == 200


def test_missing_model_gives_clear_error(client, tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_MODELS_DIR", str(tmp_path))
    img = (np.ones((40, 40, 3)) * 128).astype(np.uint8)
    r = client.post("/api/images", headers=H, files={"file": ("a.png", _png(img), "image/png")}).json()
    resp = client.post("/api/recon", headers=H, json={"image_id": r["id"]})
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "model_not_installed" and "offline" in body["hint"]


@needs_depth
@needs_testdata
def test_recon_job(client):
    p = "scenes/home.jpg"
    r = client.post("/api/images", headers=H, files={"file": (p, (TESTDATA / p).read_bytes(), "image/jpeg")}).json()
    j = _wait(client, client.post("/api/recon", headers=H, json={"image_id": r["id"], "settings": {"quality": "fast"}}).json()["job_id"])
    assert j["state"] == "done", j
    assert client.get(f"/api/files/{j['result']['scene']}", headers=H).content[:4] == b"LSC1"
