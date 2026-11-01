"""Shared code of the face-swap benchmark and optimiser.

* **Data** - faces of ``testdata/faces`` detected once per landmark backend
  (cached as ``.npz``, invalidated when the landmark code changes), cropped and
  downscaled so the face size is at most ``MAX_FS`` px.
* **Self-swap cases with ground truth** - the source ``S = augment(T)`` is the
  target photo with known photometric/geometric changes (white balance,
  exposure, gamma, contrast, saturation, a directional *relighting* field
  computed from the 3D face mesh, coloured light, blur/sharpening, noise, JPEG,
  scale 0.4-2x, rotation, translation). ``swap_face(S, face_S, T, face_T)``
  should then return ``T`` itself.
* **Metrics** - low-frequency CIEDE2000 (colour/shading), large-scale
  log-luminance error (lighting), texture-energy mismatch per frequency band
  and noise mismatch (compared as *statistics*, so a blurred or washed-out face
  is penalised instead of rewarded), seam-band error and gradient anomaly,
  coverage, plus no-reference metrics for cross-identity swaps (seam visibility,
  skin-tone/noise/sharpness/lighting consistency, identity retention).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FACES_DIR = ROOT / "testdata" / "faces"
CACHE = Path(os.environ.get("FACE_TRAIN_CACHE") or (Path(tempfile.gettempdir()) / "lumen_face_train"))
MAX_FS = 200.0  # face size (sqrt of oval area) of the benchmark crops
MIN_FS = 55.0
MAX_YAW = 40.0
PERSON = {"obama.jpg": "obama", "obama2.jpg": "obama", "obama_small.jpg": "obama"}
# validation images (never used for fitting); the rest is the training split
VAL_IMAGES = {"2008_001322.jpg", "2009_004587.jpg", "lena.jpg", "obama2.jpg", "two_people.jpg", "alex-lacamoire.png"}


def set_backend_env(backend: str) -> None:
    """Configure the engine like the user's machine (compat = no AVX -> YuNet+LBF path)."""
    os.environ.setdefault("LUMEN_MODELS_DIR", str(Path.home() / ".lumen-studio" / "models"))
    if backend in ("compat", "lbf"):
        os.environ["LUMEN_ASSUME_NO_AVX"] = "1"
        if backend == "lbf":  # old compat landmarks (LBF 68 -> mesh) instead of the ONNX face mesh
            os.environ["LUMEN_FACE_BACKEND"] = "lbf"
        else:
            os.environ.pop("LUMEN_FACE_BACKEND", None)
    else:
        os.environ.pop("LUMEN_ASSUME_NO_AVX", None)
        os.environ["LUMEN_FACE_BACKEND"] = "mediapipe"


# --------------------------------------------------------------------------- images & faces
def load_image(name: str) -> np.ndarray:
    from lumen_engine.imaging.io import decode_image

    return decode_image((FACES_DIR / name).read_bytes())[0].astype(np.float32)


def make_face(points: np.ndarray, like=None, **kw):
    from lumen_engine.face.landmarks import Face, _roll_from_eyes

    f = Face(points=np.asarray(points, np.float64), matrix=np.eye(4) if like is None else like.matrix.copy())
    if like is not None:
        f.source, f.yaw, f.pitch = like.source, like.yaw, like.pitch
        f.extra = dict(like.extra)
    for k, v in kw.items():
        setattr(f, k, v)
    f.roll = _roll_from_eyes(f.xy)
    return f


def transform_face(face, A: np.ndarray, zscale: float):
    """Face whose landmarks are mapped by the 2x3 affine ``A`` (z scaled by ``zscale``)."""
    p = face.points.copy()
    p[:, :2] = face.points[:, :2] @ A[:, :2].T + A[:, 2]
    p[:, 2] = face.points[:, 2] * zscale
    f = make_face(p, like=face)
    if "landmarks68" in face.extra:
        f.extra["landmarks68"] = np.asarray(face.extra["landmarks68"]) @ A[:, :2].T + A[:, 2]
    return f


def _landmark_code_hash(backend: str) -> str:
    face_dir = ROOT / "engine" / "lumen_engine" / "face"
    h = hashlib.md5(backend.encode())
    for f in [face_dir / "fallback.py", face_dir / "landmarks.py", face_dir / "backend.py"] + sorted((face_dir / "data").glob("*")):
        if f.is_file() and not f.name.startswith("learned"):
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def detections(backend: str = "compat", refresh: bool = False) -> dict:
    """{image name: [face dict(points, yaw, pitch, roll, source)]} for testdata/faces (cached)."""
    set_backend_env(backend)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"dets_{backend}_{_landmark_code_hash(backend)}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    from lumen_engine.face import backend as fb

    out = {}
    for p in sorted(FACES_DIR.iterdir()):
        img = load_image(p.name)
        try:
            faces, info = fb.detect(img)
        except Exception as exc:  # noqa: BLE001
            print(f"[detect] {p.name}: {exc}")
            faces = []
        out[p.name] = [
            {"points": f.points.tolist(), "yaw": f.yaw, "pitch": f.pitch, "roll": f.roll, "source": f.source,
             **({"lm68": np.asarray(f.extra["landmarks68"]).tolist()} if "landmarks68" in f.extra else {})}
            for f in faces
        ]
    path.write_text(json.dumps(out))
    return out


def face_from_record(r: dict):
    from lumen_engine.face.landmarks import Face

    f = Face(points=np.asarray(r["points"], np.float64), matrix=np.eye(4), source=r.get("source", "auto"),
             yaw=float(r["yaw"]), pitch=float(r["pitch"]), roll=float(r["roll"]))
    if "lm68" in r:
        f.extra["landmarks68"] = np.asarray(r["lm68"], np.float64)
    return f


def crop_face(img: np.ndarray, face, max_fs: float = MAX_FS, margin: float = 1.0):
    """Square-ish crop around the face (margin x bbox side on each side), downscaled to face size <= max_fs."""
    x0, y0, x1, y1 = face.bbox
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side * (0.5 + margin)
    H, W = img.shape[:2]
    X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half))
    X1, Y1 = int(min(W, cx + half)), int(min(H, cy + half))
    crop = img[Y0:Y1, X0:X1]
    k = min(1.0, max_fs / max(face.size, 1e-6))
    A = np.array([[1.0, 0, -X0], [0, 1.0, -Y0]])
    if k < 1.0:
        nw, nh = max(8, int(round(crop.shape[1] * k))), max(8, int(round(crop.shape[0] * k)))
        kx, ky = nw / crop.shape[1], nh / crop.shape[0]
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        A = np.array([[kx, 0, 0], [0, ky, 0]]) @ np.vstack([A, [0, 0, 1]])
        k = (kx + ky) / 2
    return np.ascontiguousarray(crop, dtype=np.float32), transform_face(face, A, k)


# --------------------------------------------------------------------------- augmentation
def _srgb_to_lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def shading_ratio(shape, face, light: np.ndarray, ambient: float) -> np.ndarray:
    """Smooth multiplicative field = Lambert(light) / Lambert(frontal) over the 3D face mesh, extended outside."""
    from lumen_engine.face.landmarks import topology
    from lumen_engine.imaging.filters import gaussian, normalized_blur

    tris = topology()["triangles"]
    P = face.points[:468].astype(np.float64).copy()
    P[:, 2] = -P[:, 2]  # MediaPipe: smaller z = closer -> +z towards the viewer
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n[n[:, 2] < 0] *= -1
    new = ambient + (1 - ambient) * np.clip(n @ light, 0, 1)
    ref = ambient + (1 - ambient) * np.clip(n[:, 2], 0, 1)
    r = np.log(new / ref)
    acc = np.zeros(shape, np.float32)
    wsum = np.zeros(shape, np.float32)
    for k in np.argsort(np.abs(cross_z(face.xy[tris])))[::-1]:
        tri = np.round(face.xy[tris[k]] * 4).astype(np.int32)
        cv2.fillConvexPoly(acc, tri, float(r[k]), shift=2)
        cv2.fillConvexPoly(wsum, tri, 1.0, shift=2)
    fs = face.size
    inside = (wsum > 0).astype(np.float32)
    acc = normalized_blur(acc, inside, max(1.0, fs * 0.03))
    outside_fill = normalized_blur(acc, inside, max(2.0, fs * 0.35))
    soft_in = gaussian(inside, max(1.0, fs * 0.05))
    field = acc * soft_in + outside_fill * (1 - soft_in)
    return np.exp(gaussian(field, max(1.0, fs * 0.02))).astype(np.float32)


def cross_z(t: np.ndarray) -> np.ndarray:
    d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]


def sample_aug(rng: np.random.Generator, strength: float = 1.0) -> dict:
    s = strength
    p = {
        "scale": float(np.exp(rng.uniform(np.log(0.4), np.log(2.0)))) if rng.random() < 0.85 else 1.0,
        "rot": float(rng.uniform(-12, 12) * s),
        "tx": float(rng.uniform(-0.05, 0.05)),
        "ty": float(rng.uniform(-0.05, 0.05)),
        "light_az": float(rng.uniform(0, 2 * np.pi)),
        "light_el": float(np.radians(rng.uniform(15, 65))),
        "light_amb": float(rng.uniform(0.35, 0.75)),
        "light_amount": float(rng.uniform(0.3, 1.0) * s) if rng.random() < 0.8 else 0.0,
        "light_tint": [float(v) for v in rng.uniform(-0.15, 0.15, 3) * s],
        "grad": float(rng.uniform(-0.35, 0.35) * s),
        "grad_dir": float(rng.uniform(0, 2 * np.pi)),
        "temp": float(rng.uniform(-0.22, 0.22) * s),
        "tint": float(rng.uniform(-0.08, 0.08) * s),
        "ev": float(rng.uniform(-0.7, 0.7) * s),
        "gamma": float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)) * s)),
        "contrast": float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)) * s)),
        "saturation": float(np.exp(rng.uniform(np.log(0.75), np.log(1.3)) * s)),
        "blur": 0.0,
        "sharpen": 0.0,
        "noise": float(rng.uniform(1.0, 7.0) / 255 * s) if rng.random() < 0.6 else 0.0,
        "jpeg": int(rng.uniform(35, 92)) if rng.random() < 0.6 else 0,
    }
    u = rng.random()
    if u < 0.35:
        p["blur"] = float(rng.uniform(0.5, 1.6) * s)
    elif u < 0.6:
        p["sharpen"] = float(rng.uniform(0.3, 1.0) * s)
    return p


def augment(T: np.ndarray, face, p: dict, seed: int = 0):
    """Source image ``S`` and its face from target crop ``T`` with augmentation parameters ``p``."""
    from lumen_engine.imaging.filters import gaussian

    rng = np.random.default_rng(seed)
    H, W = T.shape[:2]
    s = p["scale"]
    c = np.array(face.xy[[1, 4, 6, 168]].mean(0))
    R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), p["rot"], s)
    Ws, Hs = max(16, int(round(W * s))), max(16, int(round(H * s)))
    # place the face centre at (centre of the canvas + translation)
    cc = R[:, :2] @ c + R[:, 2]
    R[:, 2] += np.array([Ws * (0.5 + p["tx"]), Hs * (0.5 + p["ty"])]) - cc
    src = T if s >= 1 else gaussian(T, 0.45 * np.sqrt(max(1 / s**2 - 1, 0)))
    S = cv2.warpAffine(src, R, (Ws, Hs), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    S = np.clip(S, 0, 1)
    fS = transform_face(face, R, s)
    fs = fS.size
    lin = _srgb_to_lin(S)
    # directional relighting over the 3D mesh + soft gradient, optionally coloured
    logf = np.zeros(S.shape[:2], np.float32)
    if p["light_amount"] > 0:
        el, az = p["light_el"], p["light_az"]
        L = np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)])
        logf += np.log(shading_ratio(S.shape[:2], fS, L, p["light_amb"])) * p["light_amount"]
    if abs(p["grad"]) > 0:
        d = np.array([np.cos(p["grad_dir"]), np.sin(p["grad_dir"])])
        yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
        u = ((xx - fS.xy[1, 0]) * d[0] + (yy - fS.xy[1, 1]) * d[1]) / max(fs, 1)
        logf += p["grad"] * np.tanh(u)
    tint = np.asarray(p["light_tint"], np.float32)
    lin = lin * np.exp(logf[..., None] * (1 + tint))
    # white balance + exposure (von Kries gains in linear RGB)
    g = np.array([np.exp(p["temp"]), np.exp(p["tint"]), np.exp(-p["temp"])], np.float32)
    g /= 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2]
    lin = lin * g * (2.0 ** p["ev"])
    S = _lin_to_srgb(lin)
    # tone: gamma, contrast around mid grey (soft), saturation
    S = np.power(np.clip(S, 0, 1), p["gamma"])
    S = np.clip(0.5 + (S - 0.5) * p["contrast"], 0, 1)
    lab = cv2.cvtColor(S.astype(np.float32), cv2.COLOR_RGB2Lab)
    lab[..., 1:] *= p["saturation"]
    S = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    if p["blur"] > 0:
        S = gaussian(S, p["blur"])
    if p["sharpen"] > 0:
        S = np.clip(S + (S - gaussian(S, 1.2)) * p["sharpen"], 0, 1)
    if p["noise"] > 0:
        n = rng.standard_normal(S.shape[:2]).astype(np.float32)[..., None] * p["noise"]
        n = n + rng.standard_normal(S.shape).astype(np.float32) * p["noise"] * 0.4
        S = np.clip(S + n, 0, 1)
    if p["jpeg"]:
        u8 = cv2.cvtColor((S * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(p["jpeg"])])
        S = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return np.ascontiguousarray(S, dtype=np.float32), fS


# --------------------------------------------------------------------------- cases
FACESET = Path(__file__).with_name("faceset.json")


def faceset(backend: str = "compat") -> list[dict]:
    """The benchmark's frozen list of faces (image + bbox centre + size), so cases stay comparable when
    the landmark code changes. Created once from the current detections."""
    if FACESET.exists():
        return json.loads(FACESET.read_text())
    dets = detections(backend)
    out = []
    for name, recs in dets.items():
        cand = []
        for r in recs:
            f = face_from_record(r)
            if f.size < MIN_FS or abs(f.yaw) > MAX_YAW:
                continue
            x0, y0, x1, y1 = f.bbox
            cand.append({"image": name, "cx": round((x0 + x1) / 2, 1), "cy": round((y0 + y1) / 2, 1), "size": round(f.size, 1), "yaw": round(f.yaw, 1)})
        out += sorted(cand, key=lambda c: -c["size"])[:3]
    FACESET.write_text(json.dumps(out, indent=1))
    return out


def find_face(backend: str, ref: dict):
    """Detected face (current landmark code) matching a faceset entry, or None."""
    best, bd = None, 1e9
    for r in detections(backend).get(ref["image"], []):
        f = face_from_record(r)
        x0, y0, x1, y1 = f.bbox
        d = np.hypot((x0 + x1) / 2 - ref["cx"], (y0 + y1) / 2 - ref["cy"])
        if d < bd:
            best, bd = f, d
    return best if best is not None and bd < 0.35 * ref["size"] else None


def _fid(e: dict) -> str:
    return f"{e['image']}@{int(e['cx'])},{int(e['cy'])}"


def build_cases(backend: str = "compat", per_face: int = 2, seed: int = 1234) -> dict:
    """Case specifications (JSON-able) for the self-swap, real-pair and cross-identity sets."""
    faces = faceset(backend)
    rng = np.random.default_rng(seed)
    selfc = []
    for e in faces:
        for k in range(per_face):
            cid = f"{_fid(e)}/a{k}"
            strength = 1.0 if k % 2 == 0 else 0.6
            selfc.append({"kind": "self", "id": cid, "face": e, "aug": sample_aug(rng, strength),
                          "seed": int(zlib.crc32(cid.encode())), "split": "val" if e["image"] in VAL_IMAGES else "train"})
    pairs = []
    obama = [e for e in faces if PERSON.get(e["image"]) == "obama"]
    for a in obama:
        for b in obama:
            if a["image"] != b["image"]:
                pairs.append({"kind": "pair", "id": f"{_fid(a)}>{_fid(b)}", "src": a, "dst": b,
                              "split": "val" if (a["image"] in VAL_IMAGES or b["image"] in VAL_IMAGES) else "train"})
    cross = []
    others = [e for e in faces if PERSON.get(e["image"]) != "obama" or e["image"] == "obama.jpg"]
    for a in others:
        cands = [b for b in others if b["image"] != a["image"] and abs(a["yaw"] - b["yaw"]) < 18]
        rng.shuffle(cands)
        for b in cands[:2]:
            cross.append({"kind": "cross", "id": f"{_fid(a)}>{_fid(b)}", "src": a, "dst": b,
                          "split": "val" if (a["image"] in VAL_IMAGES or b["image"] in VAL_IMAGES) else "train"})
    return {"backend": backend, "self": selfc, "pair": pairs, "cross": cross}


_img_cache: dict = {}


def crop_for(backend: str, ref: dict):
    key = (backend, _fid(ref))
    if key not in _img_cache:
        f = find_face(backend, ref)
        if f is None:
            raise LookupError(f"face {_fid(ref)} not detected with the current landmark code")
        _img_cache[key] = crop_face(load_image(ref["image"]), f)
    return _img_cache[key]


def materialize(case: dict, backend: str):
    """(S, face_S, T, face_T) for a case specification."""
    if case["kind"] == "self":
        T, fT = crop_for(backend, case["face"])
        S, fS = augment(T, fT, case["aug"], case["seed"])
        return S, fS, T, fT
    S, fS = crop_for(backend, case["src"])
    T, fT = crop_for(backend, case["dst"])
    return S, fS, T, fT


# --------------------------------------------------------------------------- metrics
def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma et al. 2005), vectorised."""
    L1, a1, b1 = [lab1[..., i].astype(np.float64) for i in range(3)]
    L2, a2, b2 = [lab2[..., i].astype(np.float64) for i in range(3)]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cm7 / (Cm7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    zero = (C1p * C2p) == 0
    dh = np.where(zero, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2))
    hpm = np.where(zero, hs, hpm)
    Tt = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
          + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Cpm7 = Cpm**7
    Rc = 2 * np.sqrt(Cpm7 / (Cpm7 + 25.0**7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * Tt
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)).astype(np.float32)


def _lab(x):
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def _lum(x):
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)


def _gradmag(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def face_regions(shape, face):
    """(oval, skin) masks of a face in an image of ``shape`` (skin = oval minus dilated features)."""
    from lumen_engine.face.geometry import feature_mask, poly_mask
    from lumen_engine.face.landmarks import topology

    oval = poly_mask(shape, face.xy[topology()["oval"]])
    feat = feature_mask(shape, face.xy[:468], (0, 0), face.size * 0.04)
    return oval, oval * (1 - feat)


def skin_like(img: np.ndarray, face) -> np.ndarray:
    """Binary map of pixels whose colour matches the cheek/nose skin of ``face`` (robust Gaussian in Lab).

    Used as the denominator of the coverage metric so hair, glasses frames or hands that a mask
    (correctly) leaves out are not counted as missed face area."""
    lab = _lab(img)
    seed = np.zeros(img.shape[:2], np.uint8)
    for idx in (50, 280, 4, 205, 425, 101, 330):
        cx, cy = face.xy[idx]
        cv2.circle(seed, (int(cx), int(cy)), max(2, int(face.size * 0.05)), 1, -1)
    X = lab[seed > 0].astype(np.float64)
    if len(X) < 30:
        return np.ones(img.shape[:2], bool)
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0) * 1.4826 + 1e-3
    X = X[(np.abs(X - med) < 3 * mad).all(1)] if ((np.abs(X - med) < 3 * mad).all(1)).sum() > 20 else X
    mu, cov = X.mean(0), np.cov(X.T) + np.diag([25.0, 4.0, 4.0])
    d = lab.reshape(-1, 3) - mu
    m2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d).reshape(img.shape[:2])
    return m2 < 16.0  # generous: shading across the face stays "skin"


def _wmean(x, w):
    s = float(w.sum())
    return float((x * w).sum() / s) if s > 1e-6 else 0.0


def band_energy(img_l: np.ndarray, w: np.ndarray, s1: float, s2: float | None) -> float:
    from lumen_engine.imaging.filters import gaussian

    lo1 = gaussian(img_l, s1) if s1 > 0 else img_l
    b = (img_l - lo1) if s2 is None else (lo1 - gaussian(img_l, s2))
    return float(np.sqrt(_wmean(b * b, w) + 1e-12))


def selfswap_metrics(out: np.ndarray, T: np.ndarray, soft: np.ndarray, face) -> dict:
    """Errors of a self-swap result against its ground truth ``T`` (see module doc)."""
    from lumen_engine.imaging import analysis
    from lumen_engine.imaging.filters import dilate, gaussian, normalized_blur

    fs = face.size
    ys, xs = np.nonzero(soft > 0.01)
    pad = int(fs * 0.15) + 4
    y0, y1 = max(0, ys.min() - pad), min(T.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(T.shape[1], xs.max() + pad + 1)
    O, G, m = out[y0:y1, x0:x1], T[y0:y1, x0:x1], soft[y0:y1, x0:x1]
    oval, skin = face_regions(T.shape[:2], face)
    oval, skin = oval[y0:y1, x0:x1], skin[y0:y1, x0:x1]
    s_lf = max(1.0, fs * 0.012)
    de_lf_map = ciede2000(_lab(gaussian(O, s_lf)), _lab(gaussian(G, s_lf)))
    de_map = ciede2000(_lab(O), _lab(G))
    inner = (m > 0.5).astype(np.float32)
    sk = inner * (skin > 0.5)
    # lighting: large-scale log-luminance on skin
    eps = 0.02
    lo, lg = np.log(_lum(_srgb_to_lin(O)) + eps), np.log(_lum(_srgb_to_lin(G)) + eps)
    s_sh = max(2.0, fs * 0.05)
    sh_err = np.abs(normalized_blur(lo, sk, s_sh) - normalized_blur(lg, sk, s_sh))
    # texture energy per band (luminance), compared as statistics
    lO, lG = _lum(O), _lum(G)
    bands = [(0.8, None), (0.8, 2.0), (2.0, max(3.0, fs * 0.025))]  # fine (noise/pores), mid, low detail
    hf = []
    for s1, s2 in bands:
        eo, eg = band_energy(lO, sk, s1, s2), band_energy(lG, sk, s1, s2)
        hf.append(float(np.log((eo + 1e-4) / (eg + 1e-4))))
    n_o, n_g = analysis.noise_sigma(O, sk), analysis.noise_sigma(G, sk)
    noise = float(np.log((n_o + 0.6 / 255) / (n_g + 0.6 / 255)))
    # transition band
    band = ((m > 0.03) & (m < 0.97)).astype(np.uint8)
    band = dilate(band, 2).astype(np.float32)
    go, gg = _gradmag(gaussian(lO, 0.7)), _gradmag(gaussian(lG, 0.7))
    seam_grad = float(np.log((_wmean(go, band) + 1e-4) / (_wmean(gg, band) + 1e-4)))
    cov_oval = float((m > 0.5).sum() / max(oval.sum(), 1))
    sl = skin_like(T, face)[y0:y1, x0:x1] & (oval > 0.5)
    cov = float(((m > 0.5) & sl).sum() / max(sl.sum(), 1))  # replaced fraction of the visible face skin
    return {
        "de": _wmean(de_map, m),
        "de_lf": _wmean(de_lf_map, m),
        "de_lf_skin": _wmean(de_lf_map, sk),
        "de_lf_feat": _wmean(de_lf_map, inner * (oval > 0.5) * (skin < 0.5)),
        "shade": _wmean(sh_err, sk),
        "hf_fine": hf[0],
        "hf_mid": hf[1],
        "hf_low": hf[2],
        "noise": noise,
        "seam_de": _wmean(de_lf_map, band),
        "seam_grad": seam_grad,
        "cov": cov,
        "cov_oval": cov_oval,
    }


def selfswap_loss(m: dict) -> float:
    """Scalar objective of one self-swap case (ΔE-like units; lower is better)."""
    return (
        m["de_lf"]
        + 0.5 * m["seam_de"]
        + 25.0 * m["shade"]
        + 4.0 * (abs(m["hf_fine"]) + abs(m["hf_mid"]) + 0.5 * abs(m["hf_low"]))
        + 3.0 * abs(m["noise"])
        + 4.0 * abs(m["seam_grad"])
        + 20.0 * max(0.0, 0.85 - m["cov"])
    )


def warp_source(S, fS, T, fT):
    """Plain mesh warp of the source into the target frame (identity reference for no-ref metrics)."""
    from lumen_engine.face.geometry import dense_warp_maps

    H, W = T.shape[:2]
    mx, my, _ = dense_warp_maps(fS.xy[:468], fT.xy[:468].astype(np.float64), (H, W), (0, H, 0, W))
    return np.clip(cv2.remap(S, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), 0, 1)


def noref_metrics(out: np.ndarray, T: np.ndarray, soft: np.ndarray, face, warped: np.ndarray | None = None) -> dict:
    """No-reference naturalness metrics of a (cross-identity) swap in target ``T``."""
    from lumen_engine.imaging import analysis
    from lumen_engine.imaging.filters import dilate, erode, gaussian, normalized_blur

    fs = face.size
    oval, skin = face_regions(T.shape[:2], face)
    inner = (soft > 0.9).astype(np.float32)
    sk = inner * (skin > 0.5)
    lO, lT = _lum(out), _lum(T)
    # seam: gradient energy on the boundary vs. rings on both sides
    g = _gradmag(gaussian(lO, 0.7))
    edge = ((soft > 0.3) & (soft < 0.7)).astype(np.uint8)
    edge_d = dilate(edge, 1)
    b = (soft > 0.5).astype(np.uint8)
    ring_in = (erode(b, max(2, int(fs * 0.04))) == 0) & (b > 0) & (edge_d == 0)
    ring_out = (dilate(b, max(2, int(fs * 0.04))) > 0) & (b == 0) & (edge_d == 0)
    rings = (ring_in | ring_out).astype(np.float32)
    seam = float(np.log((_wmean(g, edge_d.astype(np.float32)) + 1e-4) / (_wmean(g, rings) + 1e-4)))
    # colour step across the boundary relative to the same step in the original target
    s = max(1.5, fs * 0.02)
    lab_o, lab_t = _lab(gaussian(out, s)), _lab(gaussian(T, s))
    ri, ro = ring_in.astype(np.float32), ring_out.astype(np.float32)
    step_o = np.array([_wmean(lab_o[..., c], ri) - _wmean(lab_o[..., c], ro) for c in range(3)])
    step_t = np.array([_wmean(lab_t[..., c], ri) - _wmean(lab_t[..., c], ro) for c in range(3)])
    seam_step = float(np.linalg.norm(step_o - step_t))
    # skin tone of the new face vs. the target's own face skin (= what matches neck/ears/forehead)
    lo, lt = _lab(out), _lab(T)
    mo = np.array([_wmean(lo[..., c], sk) for c in range(3)])
    mt = np.array([_wmean(lt[..., c], sk) for c in range(3)])
    tone = float(ciede2000(mo[None], mt[None])[0])
    n_o, n_t = analysis.noise_sigma(out, sk), analysis.noise_sigma(T, sk)
    noise = float(np.log((n_o + 0.6 / 255) / (n_t + 0.6 / 255)))
    sharp = float(np.log((analysis.sharpness(out, sk) + 1e-4) / (analysis.sharpness(T, sk) + 1e-4)))
    # lighting direction: plane fit of smoothed log-luminance on face skin
    def plane(lum):
        L = normalized_blur(np.log(lum + 0.02), sk, max(2.0, fs * 0.06))
        ys, xs = np.nonzero(sk > 0.5)
        if len(ys) < 50:
            return np.zeros(2)
        A = np.stack([(xs - xs.mean()) / fs, (ys - ys.mean()) / fs, np.ones(len(xs))], 1)
        coef, *_ = np.linalg.lstsq(A, L[ys, xs], rcond=None)
        return coef[:2]

    light = float(np.linalg.norm(plane(_lum(_srgb_to_lin(out))) - plane(_lum(_srgb_to_lin(T)))))
    # blotchiness: mid-frequency chroma / luminance variation on face skin vs. the target's own face skin
    core = erode((sk > 0.5).astype(np.uint8), max(1, int(fs * 0.03))).astype(np.float32)

    def mid(x):
        b = gaussian(x, max(1.0, fs * 0.015)) - gaussian(x, max(3.0, fs * 0.08))
        return float(np.sqrt(_wmean(b * b, core) + 1e-8))

    ch_o = np.hypot(mid(lo[..., 1]), mid(lo[..., 2]))
    ch_t = np.hypot(mid(lt[..., 1]), mid(lt[..., 2]))
    blotch_ab = float(np.log((ch_o + 0.3) / (ch_t + 0.3)))
    blotch_l = float(np.log((mid(lo[..., 0]) + 0.3) / (mid(lt[..., 0]) + 0.3)))
    res = {"x_seam": seam, "x_seam_step": seam_step, "x_tone": tone, "x_noise": noise, "x_sharp": sharp, "x_light": light,
           "x_blotch_ab": blotch_ab, "x_blotch_l": blotch_l}
    if warped is not None:
        # identity retention: mid-frequency facial detail should come from the source, not the target
        def bp(lum):
            return gaussian(lum, max(0.8, fs * 0.008)) - gaussian(lum, max(2.0, fs * 0.04))

        core = (soft > 0.97).astype(np.float32) * (oval > 0.5)
        bo, bt, bw = bp(lO), bp(lT), bp(_lum(warped))

        def corr(a, c):
            w = core > 0
            if w.sum() < 50:
                return 0.0
            a, c = a[w] - a[w].mean(), c[w] - c[w].mean()
            return float((a * c).sum() / (np.sqrt((a * a).sum() * (c * c).sum()) + 1e-9))

        res["x_id_src"] = corr(bo, bw)
        res["x_id_leak"] = corr(bo, bt) - corr(bw, bt)
    return res


def noref_loss(m: dict) -> float:
    return (
        4.0 * abs(m["x_seam"])
        + 0.5 * m["x_seam_step"]
        + 0.5 * m["x_tone"]
        + 2.0 * abs(m["x_noise"])
        + 2.0 * abs(m["x_sharp"])
        + 10.0 * m["x_light"]
        + 6.0 * max(0.0, m.get("x_blotch_ab", 0.0)) + 1.0 * max(0.0, -m.get("x_blotch_ab", 0.0))
        + 2.0 * max(0.0, m.get("x_blotch_l", 0.0))
        + 10.0 * max(0.0, m.get("x_id_leak", 0.0))
        + 5.0 * max(0.0, 0.8 - m.get("x_id_src", 1.0))
    )


def pair_metrics(out: np.ndarray, T: np.ndarray, soft: np.ndarray, face) -> dict:
    """Same-person real pair (weak ground truth): only colour/lighting at a large scale are comparable."""
    from lumen_engine.imaging.filters import normalized_blur

    fs = face.size
    _, skin = face_regions(T.shape[:2], face)
    sk = (soft > 0.9).astype(np.float32) * (skin > 0.5)
    s = max(2.0, fs * 0.06)
    lo, lt = normalized_blur(_lab(out), sk, s), normalized_blur(_lab(T), sk, s)
    return {"p_de_ls": _wmean(ciede2000(lo, lt), sk)}
