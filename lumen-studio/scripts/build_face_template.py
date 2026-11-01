#!/usr/bin/env python3
"""Build the mean-face template used by the MediaPipe-free face path.

Runs MediaPipe (on a machine where it works) and the OpenCV YuNet+LBF path on
the test photos (scripts/fetch_testdata.py), then stores in
engine/lumen_engine/face/data/face_template.npz:
  template478  mean 478-point mesh (x, y, z), normalised to unit inter-ocular distance
  template68   mean position of the 68 LBF landmarks in the same frame
  model68_3d   3D points for solvePnP head pose
  box_scale/box_dy  SSD box -> LBF box calibration (chosen by landmark accuracy)
  lm_to_box    landmark bbox -> SSD box ratios (refinement pass)
  yaw_sign/pitch_sign  sign calibration against MediaPipe's head pose
"""

import glob
from pathlib import Path

import cv2
import numpy as np

from lumen_engine.face import fallback as fb
from lumen_engine.face.geometry import umeyama
from lumen_engine.face.landmarks import detect_faces
from lumen_engine.imaging.io import decode_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "engine/lumen_engine/face/data/face_template.npz"

pairs = []  # (mediapipe face, ssd box, gray image, shape)
for path in sorted(glob.glob(str(ROOT / "testdata/faces/*"))):
    rgb = decode_image(open(path, "rb").read())[0]
    u8 = cv2.convertScaleAbs(rgb, alpha=255.0)
    bgr, gray = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    boxes = fb.detect_boxes(bgr)
    for f in detect_faces(rgb):
        best = max(boxes, key=lambda b: fb._iou(b[:4], f.bbox), default=None)
        if best is not None and fb._iou(best[:4], f.bbox) > 0.3 and f.size > 40:
            pairs.append((f, best, gray, u8.shape[:2]))
print("matched faces:", len(pairs))


def anchor_err(p68, f):
    iod = np.linalg.norm(f.xy[263] - f.xy[33])
    return float(np.linalg.norm(p68[fb.ANCH_68] - f.xy[fb.ANCH_MP], axis=1).mean() / iod)


_, lbf = fb._models()
best = None
for scale in (0.9, 1.0, 1.1, 1.2, 1.3):
    for dy in (-0.1, 0.0, 0.1):
        errs = []
        for f, b, gray, _ in pairs:
            ok, lm = lbf.fit(gray, fb._lbf_box(b, scale, dy))
            if ok:
                errs.append(anchor_err(np.asarray(lm[0]).reshape(68, 2), f))
        e = float(np.median(errs))
        if best is None or e < best[0]:
            best = (e, scale, dy)
print("box calibration (median anchor error / IOD, scale, dy):", best)
_, SCALE, DY = best

p68s, ratios = [], []
for f, b, gray, _ in pairs:
    ok, lm = lbf.fit(gray, fb._lbf_box(b, SCALE, DY))
    p = np.asarray(lm[0], np.float64).reshape(68, 2)
    p68s.append(p)
    lx0, ly0 = p.min(0)
    lx1, ly1 = p.max(0)
    lw, lh = lx1 - lx0, ly1 - ly0
    ratios.append([(b[0] - lx0) / lw, (b[1] - ly0) / lh, (b[2] - lx0) / lw, (b[3] - ly0) / lh])
lm_to_box = np.median(np.array(ratios), 0)

front = [i for i, (f, *_r) in enumerate(pairs) if abs(f.yaw) < 14 and abs(f.pitch) < 16]
print("frontal faces for the template:", len(front))


def normalise(xy):
    c = xy[[33, 133, 362, 263]].mean(0)
    return (xy - c) / np.linalg.norm(xy[263] - xy[33])


ref = normalise(pairs[front[0]][0].xy[:478].astype(np.float64))
for _ in range(4):
    al_xy, al_z, al_68 = [], [], []
    for i in front:
        f = pairs[i][0]
        S = umeyama(f.xy[:478].astype(np.float64), ref)
        sc = np.sqrt(abs(np.linalg.det(S[:, :2])))
        al_xy.append(f.xy[:478] @ S[:, :2].T + S[:, 2])
        z = f.points[:478, 2] * sc
        al_z.append(z - z.mean())
        al_68.append(p68s[i] @ S[:, :2].T + S[:, 2])
    ref = normalise(np.mean(al_xy, 0))
tpl478 = np.column_stack([np.mean(al_xy, 0), np.mean(al_z, 0)])
c = tpl478[[33, 133, 362, 263], :2].mean(0)
iod = np.linalg.norm(tpl478[263, :2] - tpl478[33, :2])
tpl478[:, :2] = (tpl478[:, :2] - c) / iod
tpl478[:, 2] /= iod
tpl68 = (np.mean(al_68, 0) - c) / iod
d = np.linalg.norm(tpl478[None, :, :2] - tpl68[:, None, :], axis=2)
model68 = np.column_stack([tpl68, tpl478[d.argmin(1), 2]])

data = dict(template478=tpl478.astype(np.float32), template68=tpl68.astype(np.float32), model68_3d=model68.astype(np.float32),
            box_scale=np.float32(SCALE), box_dy=np.float32(DY), lm_to_box=lm_to_box.astype(np.float32),
            yaw_sign=np.float32(1), pitch_sign=np.float32(1), n_faces=np.int32(len(front)))
np.savez_compressed(OUT, **data)
fb.template.cache_clear()

ym, yf, pm, pf, errs = [], [], [], [], []
for (f, b, gray, shape), p in zip(pairs, p68s):
    g = fb.densify(p, shape)
    ym.append(f.yaw); yf.append(g.yaw); pm.append(f.pitch); pf.append(g.pitch)
    errs.append(np.linalg.norm(g.xy[:468] - f.xy[:468], axis=1).mean() / np.linalg.norm(f.xy[263] - f.xy[33]))
cy, cp = np.corrcoef(ym, yf)[0, 1], np.corrcoef(pm, pf)[0, 1]
data["yaw_sign"], data["pitch_sign"] = np.float32(1 if cy >= 0 else -1), np.float32(1 if cp >= 0 else -1)
np.savez_compressed(OUT, **data)
print(f"yaw correlation {abs(cy):.2f} (sign {int(data['yaw_sign'])}), pitch correlation {abs(cp):.2f} (sign {int(data['pitch_sign'])})")
print(f"dense mesh error vs MediaPipe: median {np.median(errs)*100:.1f}% of inter-ocular distance, 90th pct {np.percentile(errs, 90)*100:.1f}%")
print("saved", OUT)


# ---- data-driven head pose: geometric features of the 68 points -> MediaPipe yaw/pitch
def pose_features(p):
    E = p[36:48].mean(0); Mo = p[48:68].mean(0); N = p[30]
    u = p[42:48].mean(0) - p[36:42].mean(0); u /= np.linalg.norm(u) + 1e-9
    v = np.array([-u[1], u[0]])
    if v @ (Mo - E) < 0:
        v = -v
    jw = np.linalg.norm(p[16] - p[0]) + 1e-9
    em = (Mo - E) @ v + 1e-9
    dl, dr = np.linalg.norm(N - p[0]), np.linalg.norm(N - p[16])
    return np.array([((N - (p[0] + p[16]) / 2) @ u) / jw, (dl - dr) / (dl + dr), ((N - E) @ v) / em, ((p[8] - Mo) @ v) / em, ((E - (p[0] + p[16]) / 2) @ v) / jw, 1.0])

F = np.array([pose_features(p) for p in p68s])
Y = np.array([[f.yaw, f.pitch] for f, *_r in pairs])
coef, *_ = np.linalg.lstsq(F, Y, rcond=None)
pred = np.zeros_like(Y)
for i in range(len(F)):  # leave-one-out estimate of the real accuracy
    m = np.arange(len(F)) != i
    c, *_ = np.linalg.lstsq(F[m], Y[m], rcond=None)
    pred[i] = F[i] @ c
for k, name in enumerate(("yaw", "pitch")):
    print(f"pose regression {name}: leave-one-out correlation {np.corrcoef(pred[:, k], Y[:, k])[0, 1]:.2f}, mean abs error {np.abs(pred[:, k] - Y[:, k]).mean():.1f} deg")
data["pose_coef"] = coef.astype(np.float32)
np.savez_compressed(OUT, **data)
print("saved pose model")
