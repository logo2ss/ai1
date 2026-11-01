"""Face-swap benchmark: ground-truth self-swaps, same-person pairs and cross-identity no-reference metrics.

Run from ``engine/`` (CPU-heavy: wrap in the shared slot wrapper)::

    /tmp/claude-0/coord/heavy python -m tools.face_train.bench --backend compat --label baseline --post

``--save DIR`` writes visual examples (target | source | result | 5x error) for inspection.
Results go to ``$FACE_TRAIN_CACHE/results/<label>.json``; ``--post`` appends a leaderboard row to
``/tmp/claude-0/coord/face_bench.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.face_train import common  # noqa: E402

LEADERBOARD = Path("/tmp/claude-0/coord/face_bench.md")
SELF_KEYS = ["de_lf", "de_lf_skin", "de_lf_feat", "de", "shade", "hf_fine", "hf_mid", "hf_low", "noise", "seam_de", "seam_grad", "cov", "cov_oval"]
NOREF_KEYS = ["x_seam", "x_seam_step", "x_tone", "x_noise", "x_sharp", "x_light", "x_blotch_ab", "x_blotch_l", "x_id_src", "x_id_leak"]


def _init_worker(backend: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2

    cv2.setNumThreads(1)
    common.set_backend_env(backend)


def run_one(job) -> dict:
    case, backend, settings, overrides, save_dir = job
    import cv2

    from lumen_engine.face import settings as fset
    from lumen_engine.face.swap import FaceSwapSettings, swap_face

    getattr(fset, "set_overrides", lambda d: None)(overrides)  # absent in old pipeline versions
    abl = os.environ.get("FACE_TRAIN_ABLATE", "")
    if "noparse" in abl:  # F1: heuristic masks instead of the face-parsing model
        from lumen_engine.face import parsing

        parsing.available = lambda: False
    if "legacy" in abl:  # F2: legacy appearance transfer instead of the SH/albedo path
        import lumen_engine.face.swap as _sw

        _sw.APPEARANCE_MODE = "legacy"
    if "learned" in abl and not getattr(run_one, "_patched", False):  # F3: learned colour model in place of F2's SH stage
        import lumen_engine.face.swap as _sw
        from lumen_engine.face import learned as _lc

        def _app(src_img, dst_img, warped, target, dst_pts, src_support, feat, roi, fs, s, geo):
            return _lc.color_transfer(warped, target, dst_pts, (roi[2], roi[0]), fs), {"method": "learned"}

        _sw._appearance_sh = _app
        run_one._patched = True
    S, fS, T, fT = common.materialize(case, backend)
    s = FaceSwapSettings.from_dict(settings)
    s.force = True
    base = {"id": case["id"], "kind": case["kind"], "split": case["split"]}
    t0 = time.time()
    try:
        out, rep, dbg = swap_face(S, fS, T, fT, s)
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"{type(exc).__name__}: {exc}"}
    dt = time.time() - t0
    soft = dbg["mask"]
    res = {**base, "time": dt, "compatible": bool(rep["analysis"]["compatible"]), "fs": fT.size}
    if case["kind"] == "self":
        m = common.selfswap_metrics(out, T, soft, fT)
        res.update(m, loss=common.selfswap_loss(m))
    else:
        warped = common.warp_source(S, fS, T, fT)
        m = common.noref_metrics(out, T, soft, fT, warped)
        res.update(m, loss=common.noref_loss(m))
        if case["kind"] == "pair":
            res.update(common.pair_metrics(out, T, soft, fT))
    if save_dir:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        h, w = T.shape[:2]
        Sv = cv2.resize(S, (w, h), interpolation=cv2.INTER_AREA)
        tiles = [T, Sv, out] + ([np.clip(np.abs(out - T) * 5, 0, 1)] if case["kind"] == "self" else [])
        img = np.concatenate(tiles, 1)
        name = case["id"].replace("/", "_").replace("#", "-").replace(">", "_to_")
        cv2.imwrite(str(d / f"{case['kind']}_{name}.jpg"), cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return res


def evaluate(cases: list[dict], backend: str, settings: dict | None = None, overrides: dict | None = None, workers: int = 2, save_dir: str | None = None) -> list[dict]:
    jobs = [(c, backend, settings or {}, overrides or {}, save_dir) for c in cases]
    if workers <= 1:
        _init_worker(backend)
        return [run_one(j) for j in jobs]
    from multiprocessing import get_context

    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(backend,)) as pool:
        return pool.map(run_one, jobs, chunksize=1)


def summarize(results: list[dict]) -> dict:
    out: dict = {}
    for kind, keys in (("self", SELF_KEYS), ("pair", NOREF_KEYS + ["p_de_ls"]), ("cross", NOREF_KEYS)):
        for split in ("train", "val", "all"):
            rs = [r for r in results if r["kind"] == kind and (split == "all" or r["split"] == split) and "error" not in r]
            if not rs:
                continue
            d = {"n": len(rs), "loss": float(np.mean([r["loss"] for r in rs])), "time": float(np.mean([r["time"] for r in rs]))}
            for k in keys:
                vals = [r[k] for r in rs if k in r]
                if vals:
                    # signed log-ratios: report mean |.| (the error) and the signed mean (the bias)
                    signed = k.startswith(("hf_", "noise", "seam_grad", "x_seam", "x_noise", "x_sharp")) and k != "x_seam_step"
                    d[k] = float(np.mean(np.abs(vals))) if signed else float(np.mean(vals))
                    if signed:
                        d[k + "_bias"] = float(np.mean(vals))
            out[f"{kind}/{split}"] = d
    out["errors"] = [f"{r['id']}: {r['error']}" for r in results if "error" in r]
    return out


def post_row(label: str, backend: str, summ: dict, note: str = "") -> str:
    def g(key, k, fmt="{:.2f}"):
        v = summ.get(key, {}).get(k)
        return fmt.format(v) if v is not None else "-"

    row = (
        f"| {time.strftime('%H:%M')} | {label} | {backend} | {g('self/train', 'loss')} | {g('self/val', 'loss')} | {g('self/all', 'de_lf')} "
        f"| {g('self/all', 'shade', '{:.3f}')} | {g('self/all', 'hf_fine')}/{g('self/all', 'hf_mid')} | {g('self/all', 'noise')} "
        f"| {g('self/all', 'seam_de')} | {g('self/all', 'cov')} | {g('pair/all', 'p_de_ls')} | {g('cross/all', 'loss')} "
        f"| {g('cross/all', 'x_seam')} | {g('cross/all', 'x_tone')} | {g('cross/all', 'x_id_leak')} | {g('self/all', 'time')}s | {note} |"
    )
    if not LEADERBOARD.exists():
        LEADERBOARD.write_text(
            "# Face-swap benchmark leaderboard (F3)\n\n"
            "Self-swap = source is the target photo with known changes (colour temperature, exposure, gamma, contrast,\n"
            "3D directional relighting, blur/sharpen, noise, JPEG, scale 0.4-2x, rotation); the ideal result is the target.\n"
            "Lower is better everywhere except cov (coverage of the face oval). self loss = de_lf + 0.5 seam_de + 25 shade\n"
            "+ 4 (|hf_fine|+|hf_mid|+0.5|hf_low|) + 3 |noise| + 4 |seam_grad| + coverage penalty. de_lf = CIEDE2000 after a\n"
            "small blur (colour/shading), shade = large-scale log-luminance error, hf = |log texture-energy ratio| (blur AND\n"
            "over-sharpening are penalised), pair = obama photos swapped into each other (large-scale ΔE),\n"
            "cross = no-reference loss on cross-identity pairs (seam gradient anomaly, tone, noise, sharpness, lighting,\n"
            "identity leak = how much target mid-frequency detail shows through).\n\n"
            "| time | label | backend | self loss train | self loss val | de_lf | shade | hf fine/mid | noise | seam_de | cov | pair dE | cross loss | x_seam | x_tone | x_id_leak | s/swap | note |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
    with LEADERBOARD.open("a") as f:
        f.write(row + "\n")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="compat", choices=["compat", "mediapipe", "lbf"])
    ap.add_argument("--ablate", default="", help="comma list: noparse (F1 parsing off), legacy (F2 legacy appearance)")
    ap.add_argument("--stride", type=int, default=1, help="use every N-th case per kind")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--label", default="run")
    ap.add_argument("--settings", default="{}", help="JSON FaceSwapSettings overrides")
    ap.add_argument("--overrides", default="{}", help="JSON internal-constant overrides (settings.tv names)")
    ap.add_argument("--kinds", default="self,pair,cross")
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="use only the first N cases per kind (quick checks)")
    ap.add_argument("--save", default=None, help="directory for visual examples")
    ap.add_argument("--post", action="store_true", help="append a row to the coord leaderboard")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    common.set_backend_env(a.backend)
    os.environ["FACE_TRAIN_ABLATE"] = a.ablate
    t0 = time.time()
    spec = common.build_cases(a.backend)
    cases = []
    for kind in a.kinds.split(","):
        cs = [c for c in spec[kind] if a.split == "all" or c["split"] == a.split]
        cs = cs[:: a.stride]
        cases += cs[: a.limit] if a.limit else cs
    print(f"{len(cases)} cases ({', '.join(f'{k}={len(spec[k])}' for k in ('self', 'pair', 'cross'))}) backend={a.backend}", flush=True)
    results = evaluate(cases, a.backend, json.loads(a.settings), json.loads(a.overrides), a.workers, a.save)
    summ = summarize(results)
    outdir = common.CACHE / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.label}.json").write_text(json.dumps({"summary": summ, "results": results, "settings": a.settings, "overrides": a.overrides}, indent=1))
    for k, v in summ.items():
        if k != "errors":
            print(k, json.dumps({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}))
    if summ["errors"]:
        print("ERRORS:", *summ["errors"], sep="\n  ")
    print(f"total {time.time() - t0:.0f}s")
    if a.post:
        print(post_row(a.label, a.backend, summ, a.note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
