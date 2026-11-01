"""Face-swap mask evaluation: side-by-side sheets and quality metrics.

For every test pair and face backend it runs ``swap_face`` with the current
masks ("new") and, optionally, with the pre-parsing heuristic masks
("base", tools/face_masks/baseline_masking.py) in the same process, so the
comparison isolates the mask change. Each sheet row shows
source | target | base result | new result | new mask (red, 0.5 contour)
as crops around the target face. ``metrics.json`` collects report["quality"].

    cd engine && python -m tools.face_masks.sheets --out /tmp/sheets --modes compat mediapipe
    python -m tools.face_masks.sheets --out /tmp/sheets --pairs lin obama_biden --variants new
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FACES = ROOT / "testdata" / "faces"

# name: (source file, source pick (rx, ry) or None = largest face, target file, target pick)
PAIRS = {
    "obama_biden": ("obama.jpg", None, "biden.jpg", None),
    "biden_obama2": ("biden.jpg", None, "obama2.jpg", None),  # target: open mouth, microphone
    "obamasmall_lin": ("obama_small.jpg", None, "lin-manuel-miranda.png", None),  # target: microphone over chin
    "lin_obamasmall": ("lin-manuel-miranda.png", None, "obama_small.jpg", None),  # source: microphone
    "two_people": ("two_people.jpg", (0.28, 0.19), "two_people.jpg", (0.77, 0.24)),
    "alex_glasses": ("alex-lacamoire.png", None, "2008_002506.jpg", (0.78, 0.35)),  # target wears glasses
    "glasses_obama": ("2008_002506.jpg", (0.78, 0.35), "obama.jpg", None),  # source wears glasses
    "fringe_alex": ("2009_004587.jpg", (0.74, 0.63), "alex-lacamoire.png", None),  # source fringe
    "alex_fringe": ("alex-lacamoire.png", None, "2009_004587.jpg", (0.74, 0.63)),  # target fringe stays
    "bald": ("bald_guys.jpg", (0.36, 0.22), "bald_guys.jpg", (0.62, 0.69)),
    "obama_lena": ("obama.jpg", None, "lena.jpg", None),  # target: hat, hair over the face
    "alex_lena": ("alex-lacamoire.png", None, "lena.jpg", None),  # beard source, hair strand on target
    "woman_1322": ("2008_001009.jpg", (0.48, 0.52), "2008_001322.jpg", (0.53, 0.67)),
    "obamasmall_messi": ("obama_small.jpg", None, "messi5.jpg", None),  # tiny target face
}
MAX_SIDE = 800


def _load(name: str) -> np.ndarray:
    from lumen_engine.imaging.io import decode_image

    rgb = decode_image((FACES / name).read_bytes())[0]
    h, w = rgb.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


_det_cache: dict = {}


def _faces(name: str, rgb: np.ndarray, mode: str):
    from lumen_engine.face import backend

    key = (name, mode)
    if key not in _det_cache:
        os.environ["LUMEN_FACE_BACKEND"] = "opencv" if mode == "compat" else "mediapipe"
        faces, info = backend.detect(rgb)
        _det_cache[key] = faces
    return _det_cache[key]


def _pick(faces, rgb, pick):
    if not faces:
        return None
    if pick is None:
        return max(faces, key=lambda f: f.size)
    h, w = rgb.shape[:2]
    p = np.array([pick[0] * w, pick[1] * h])

    def d(f):
        x0, y0, x1, y1 = f.bbox
        return np.linalg.norm(np.array([(x0 + x1) / 2, (y0 + y1) / 2]) - p)

    return min(faces, key=d)


def _crop_box(face, shape, k=0.55):
    x0, y0, x1, y1 = face.bbox
    pad = face.size * k
    H, W = shape[:2]
    return int(max(0, y0 - pad)), int(min(H, y1 + pad)), int(max(0, x0 - pad)), int(min(W, x1 + pad))


def _tile(img, box, hgt=260):
    y0, y1, x0, x1 = box
    sub = img[y0:y1, x0:x1]
    s = hgt / max(1, sub.shape[0])
    return cv2.resize(sub, (max(1, int(sub.shape[1] * s)), hgt), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def _overlay(img, mask):
    m = np.clip(mask, 0, 1)[..., None]
    red = np.zeros_like(img)
    red[..., 0] = 1.0
    out = img * (1 - 0.45 * m) + red * 0.45 * m
    cont = ((mask > 0.5).astype(np.uint8))
    edges = cv2.morphologyEx(cont, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edges] = (1.0, 1.0, 0.0)
    return out


def _label(tile, text):
    t = (np.clip(tile, 0, 1) * 255).astype(np.uint8).copy()
    cv2.rectangle(t, (0, 0), (t.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(t, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def run(pairs, modes, variants, out: Path, settings: dict | None = None):
    from lumen_engine.face import swap as swap_mod
    from lumen_engine.face.settings import FaceSwapSettings

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baseline_masking  # noqa: E402

    new_masks = swap_mod._face_masks
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}
    for pname in pairs:
        sfile, spick, tfile, tpick = PAIRS[pname]
        S, T = _load(sfile), _load(tfile)
        rows = []
        for mode in modes:
            fs = _pick(_faces(sfile, S, mode), S, spick)
            ft = _pick(_faces(tfile, T, mode), T, tpick)
            if fs is None or ft is None:
                print(f"{pname} [{mode}]: face not found", flush=True)
                continue
            box = _crop_box(ft, T.shape)
            sbox = _crop_box(fs, S.shape)
            tiles = [_label(_tile(S, sbox), f"source {mode}"), _label(_tile(T, box), "target")]
            for var in variants:
                swap_mod._face_masks = baseline_masking._face_masks if var == "base" else new_masks
                from lumen_engine.face import masking as _mk

                _mk.USE_SEAM = var != "noseam"  # "noseam": parsed masks with the plain distance fade
                st = FaceSwapSettings.from_dict({**(settings or {}), "force": True})
                t0 = time.time()
                try:
                    res, rep, dbg = swap_mod.swap_face(S, fs, T, ft, st)
                except Exception as exc:  # noqa: BLE001
                    print(f"{pname} [{mode}/{var}] failed: {exc}", flush=True)
                    tiles.append(_label(_tile(T, box) * 0.3, f"{var}: FAILED"))
                    continue
                finally:
                    swap_mod._face_masks = new_masks
                    _mk.USE_SEAM = True
                dt = time.time() - t0
                q = dict(rep.get("quality", {}))
                q["seconds"] = round(dt, 2)
                q["compatible"] = rep["analysis"]["compatible"]
                q["mask"] = rep.get("mask")
                metrics.setdefault(pname, {}).setdefault(mode, {})[var] = q
                print(f"{pname:18s} {mode:9s} {var:4s} {json.dumps({k: v for k, v in q.items() if k != 'mask'})}", flush=True)
                tiles.append(_label(_tile(res, box), f"{var} seam={q.get('seam_visibility')} tdE={q.get('transition_delta_e')}"))
                if var == variants[-1]:
                    tiles.append(_label(_tile(_overlay(T, dbg["mask"]), box), f"{var} mask"))
                cv2.imwrite(str(out / f"{pname}_{mode}_{var}.jpg"), (np.clip(res, 0, 1)[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(np.hstack(tiles))
        if rows:
            wmax = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
            sheet = np.vstack(rows)
            if sheet.shape[1] > 1200:  # keep sheets small enough to review
                sheet = cv2.resize(sheet, (1200, int(sheet.shape[0] * 1200 / sheet.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"sheet_{pname}.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def summarize(metrics: dict) -> None:
    keys = ("seam_visibility", "transition_delta_e", "skin_delta_e_after")
    for mode in ("compat", "mediapipe"):
        for var in ("base", "noseam", "new"):
            vals = {k: [] for k in keys}
            for p in metrics.values():
                q = p.get(mode, {}).get(var)
                if q:
                    for k in keys:
                        if k in q:
                            vals[k].append(q[k])
            if vals[keys[0]]:
                print(f"{mode:9s} {var:4s} n={len(vals[keys[0]]):2d} " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in vals.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS))
    ap.add_argument("--modes", nargs="*", default=["compat", "mediapipe"])
    ap.add_argument("--variants", nargs="*", default=["base", "new"])
    a = ap.parse_args()
    m = run(a.pairs, a.modes, a.variants, a.out)
    summarize(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
