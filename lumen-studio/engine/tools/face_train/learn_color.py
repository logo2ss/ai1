"""Fit the learned colour/lighting model (lumen_engine/face/learned.py) on augmented self-swap data.

For every training face many sources ``S = augment(T)`` are generated; the source is
mesh-warped back into the target frame (``W``) and the model learns the CIELAB
correction ``blur(Lab T) - blur(Lab W)`` from the operator features of
``learned.features``. Because the model is linear, blurring the features with the same
kernel gives the exact least-squares fit of the *low-frequency* error (colour and
shading), so noise/texture differences do not bias it. Ridge strength is chosen on the
validation images. Weights are written as plain arrays to
``lumen_engine/face/data/learned_color.npz``.

    /tmp/claude-0/coord/heavy python -m tools.face_train.learn_color --augs 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.face_train import common  # noqa: E402


def case_data(T, fT, aug, seed, n_samples: int, rng: np.random.Generator):
    from lumen_engine.face import learned
    from lumen_engine.imaging.filters import gaussian

    S, fS = common.augment(T, fT, aug, seed)
    scale = fT.size / max(fS.size, 1e-6)
    Sw = gaussian(S, 0.45 / scale) if scale < 0.8 else S
    W = common.warp_source(Sw, fS, T, fT)
    fs = fT.size
    x0, y0, x1, y1 = fT.bbox
    pad = int(fs * 0.3) + 4
    H, Wd = T.shape[:2]
    ry0, ry1, rx0, rx1 = max(0, int(y0) - pad), min(H, int(y1) + pad), max(0, int(x0) - pad), min(Wd, int(x1) + pad)
    Tr, Wr = T[ry0:ry1, rx0:rx1], W[ry0:ry1, rx0:rx1]
    pts = fT.xy[:468] - np.array([rx0, ry0])
    feats, lw = learned.features(Wr, Tr, pts, fs)
    s_lf = max(1.0, fs * 0.012)
    y = gaussian(common._lab(Tr), s_lf) - gaussian(lw, s_lf)
    Xb = np.stack([gaussian(feats[..., i], s_lf) for i in range(feats.shape[-1])], -1)
    oval, skin = common.face_regions(Tr.shape[:2], common.make_face(np.column_stack([pts, fT.points[:468, 2]]), like=fT))
    region = cv2.dilate(oval, np.ones((5, 5), np.uint8)) > 0.5
    ys, xs = np.nonzero(region)
    if len(ys) < 100:
        return None
    wts = np.where(skin[ys, xs] > 0.5, 1.0, 0.6)
    k = rng.choice(len(ys), size=min(n_samples, len(ys)), replace=False, p=wts / wts.sum())
    return Xb[ys[k], xs[k]], y[ys[k], xs[k]], (lw, Tr, feats, region, skin)


def build(split: str, augs: int, n_samples: int, backend: str, seed: int):
    faces = [e for e in common.faceset(backend) if (e["image"] in common.VAL_IMAGES) == (split == "val")]
    rng = np.random.default_rng(seed)
    X, Y, G = [], [], []
    for gi, e in enumerate(faces):
        name, idx = e["image"], int(e["cx"])
        try:
            T, fT = common.crop_for(backend, e)
        except LookupError:
            continue
        for k in range(augs):
            aug = common.sample_aug(rng, 1.0 if k % 3 else 0.6)
            r = case_data(T, fT, aug, int(zlib.crc32(f"{name}{idx}{k}{seed}".encode())), n_samples, rng)
            if r is None:
                continue
            X.append(r[0])
            Y.append(r[1])
            G.append(np.full(len(r[0]), gi))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = Y.mean(0)
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    Wt = np.linalg.solve(A, Z.T @ (Y - ym))
    return {"mean": mu.astype(np.float32), "std": sd.astype(np.float32), "weights": Wt.astype(np.float32), "bias": ym.astype(np.float32)}


def de_after(m, X, Y):
    """Mean Lab error of the low-frequency residual after the correction (≈ ΔE76)."""
    pred = ((X - m["mean"]) / m["std"]) @ m["weights"] + m["bias"]
    return float(np.sqrt(((Y - pred) ** 2).sum(1)).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--augs", type=int, default=10)
    ap.add_argument("--samples", type=int, default=2500)
    ap.add_argument("--backend", default="compat")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    common.set_backend_env(a.backend)
    import cv2 as _cv

    _cv.setNumThreads(2)
    from lumen_engine.face import learned

    t0 = time.time()
    Xtr, Ytr, _ = build("train", a.augs, a.samples, a.backend, 11)
    Xva, Yva, _ = build("val", max(3, a.augs // 2), a.samples, a.backend, 99)
    print(f"data: train {Xtr.shape} val {Xva.shape} in {time.time() - t0:.0f}s", flush=True)
    print(f"no correction: train {np.sqrt((Ytr ** 2).sum(1)).mean():.2f} val {np.sqrt((Yva ** 2).sum(1)).mean():.2f}")
    best = None
    for lam in (1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
        m = fit(Xtr, Ytr, lam)
        e_tr, e_va = de_after(m, Xtr, Ytr), de_after(m, Xva, Yva)
        print(f"lambda {lam:g}: train {e_tr:.3f} val {e_va:.3f}")
        if best is None or e_va < best[0]:
            best = (e_va, lam, m)
    e_va, lam, m = best
    m = fit(np.concatenate([Xtr, Xva]), np.concatenate([Ytr, Yva]), lam) if False else m  # keep val honest
    out = Path(a.out) if a.out else learned.DATA
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, version=np.int32(learned.VERSION), clip=np.float32(25.0), lam=np.float32(lam), val_error=np.float32(e_va), **m)
    print(json.dumps({"saved": str(out), "lambda": lam, "val_lab_error": round(e_va, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
