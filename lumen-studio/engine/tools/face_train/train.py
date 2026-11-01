"""Data-driven tuning of the face-swap pipeline on the self-swap benchmark.

Optimises the user-facing defaults (:class:`FaceSwapSettings`) and the internal
constants that the pipeline reads through ``settings.tv(name, default)``
(discovered automatically after one run) by minimising

    objective = mean self-swap loss (train images) + CROSS_W * mean cross-identity loss (train pairs)

The cross-identity (no-reference) term guards against the self-swap shortcut of
copying the target (identity leak, over-smoothing). Search = random perturbations
around the incumbent followed by coordinate descent with shrinking steps; the
result is reported on the held-out validation images (before vs. after).

    /tmp/claude-0/coord/heavy python -m tools.face_train.train --budget 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.face_train import bench, common  # noqa: E402

CROSS_W = 0.5
# user settings and their search ranges
SETTING_SPACE = {
    "blend_strength": (0.0, 1.0),
    "color_match": (0.0, 1.0),
    "lighting_match": (0.0, 1.0),
}
BOOL_SETTINGS = ("match_grain", "match_sharpness")


def constant_space(seen: dict[str, float], only: list[str] | None = None) -> dict[str, tuple[float, float]]:
    space = {}
    for k, v in seen.items():
        if only and k not in only:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        v = float(v)
        if v == 0:
            space[k] = (0.0, 0.1)
        elif 0 < v <= 1 and any(t in k for t in ("strength", "amount", "weight", "keep", "mix", "frac")):
            space[k] = (0.0, 1.0)
        elif v > 0:
            space[k] = (v * 0.4, v * 2.5)
        else:
            space[k] = (v * 2.5, v * 0.4)
    return space


class Objective:
    def __init__(self, backend: str, cases: list[dict], pool, cross_w: float = CROSS_W):
        self.backend, self.cases, self.pool, self.cross_w = backend, cases, pool, cross_w
        self.history: list[dict] = []

    def __call__(self, settings: dict, overrides: dict) -> tuple[float, dict]:
        jobs = [(c, self.backend, settings, overrides, None) for c in self.cases]
        res = self.pool.map(bench.run_one, jobs, chunksize=1)
        summ = bench.summarize(res)
        ls = summ.get("self/all", {}).get("loss", 99.0)
        lc = summ.get("cross/all", {}).get("loss", 0.0)
        nerr = len(summ["errors"])
        val = ls + self.cross_w * lc + 5.0 * nerr
        self.history.append({"settings": settings, "overrides": overrides, "objective": val, "self": ls, "cross": lc, "errors": nerr})
        return val, summ


def optimise(obj: Objective, settings: dict, overrides: dict, space_s: dict, space_c: dict, budget: int, rng: np.random.Generator, log):
    best_s, best_c = dict(settings), dict(overrides)
    best, _ = obj(best_s, best_c)
    log(f"start objective {best:.4f}")
    names = [("s", k) for k in space_s] + [("c", k) for k in space_c] + [("b", k) for k in BOOL_SETTINGS]

    def get(kind, k):
        return best_s.get(k, getattr(_defaults(), k)) if kind in "sb" else best_c.get(k, _seen().get(k))

    def clipv(kind, k, v):
        lo, hi = (space_s if kind == "s" else space_c)[k]
        return float(np.clip(v, lo, hi))

    evals = 1
    # phase 1: random perturbations of the incumbent (all parameters at once, 15 % of the range)
    n_rand = max(0, budget // 3)
    for _ in range(n_rand):
        cs, cc = dict(best_s), dict(best_c)
        for kind, k in names:
            if kind == "b":
                if rng.random() < 0.1:
                    cs[k] = not get(kind, k)
                continue
            lo, hi = (space_s if kind == "s" else space_c)[k]
            v = get(kind, k) + rng.normal(0, 0.15) * (hi - lo)
            (cs if kind == "s" else cc)[k] = clipv(kind, k, v)
        val, _ = obj(cs, cc)
        evals += 1
        if val < best:
            best, best_s, best_c = val, cs, cc
            log(f"[rand {evals}] improved {best:.4f}")
    # phase 2: coordinate descent with shrinking steps
    step = 0.2
    while evals < budget and step > 0.02:
        improved = False
        for kind, k in names:
            if evals >= budget:
                break
            if kind == "b":
                cs = dict(best_s)
                cs[k] = not get(kind, k)
                val, _ = obj(cs, best_c)
                evals += 1
                if val < best - 1e-4:
                    best, best_s = val, cs
                    improved = True
                    log(f"[cd {evals}] {k}={cs[k]} -> {best:.4f}")
                continue
            lo, hi = (space_s if kind == "s" else space_c)[k]
            cur = get(kind, k)
            for sgn in (+1, -1):
                v = clipv(kind, k, cur + sgn * step * (hi - lo))
                if abs(v - cur) < 1e-9:
                    continue
                cs, cc = dict(best_s), dict(best_c)
                (cs if kind == "s" else cc)[k] = v
                val, _ = obj(cs, cc)
                evals += 1
                if val < best - 1e-4:
                    best, best_s, best_c = val, cs, cc
                    improved = True
                    log(f"[cd {evals}] {k}={v:.4g} -> {best:.4f}")
                    break
        if not improved:
            step *= 0.5
            log(f"step -> {step:.3f}")
    return best, best_s, best_c


_def_cache: dict = {}


def _defaults():
    from lumen_engine.face.settings import FaceSwapSettings

    return FaceSwapSettings()


def _seen():
    from lumen_engine.face import settings as fset

    return fset.seen_constants()


def discover_constants(backend: str, case: dict) -> dict[str, float]:
    """Run one swap in-process to collect the constants read through settings.tv()."""
    bench._init_worker(backend)
    bench.run_one((case, backend, {}, {}, None))
    return _seen()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="compat")
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-settings", action="store_true", help="tune only internal constants")
    ap.add_argument("--constants", default="", help="comma list: tune only these constants ('' = all discovered, 'none' = none)")
    ap.add_argument("--cross-weight", type=float, default=CROSS_W)
    ap.add_argument("--cross-limit", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="train")
    a = ap.parse_args(argv)
    common.set_backend_env(a.backend)
    spec = common.build_cases(a.backend)
    train_cases = [c for c in spec["self"] if c["split"] == "train"] + [c for c in spec["cross"] if c["split"] == "train"][: a.cross_limit]
    val_cases = [c for c in spec["self"] + spec["cross"] + spec["pair"] if c["split"] == "val"] + [c for c in spec["pair"] if c["split"] == "train"]
    seen = discover_constants(a.backend, spec["self"][0])
    only = None if not a.constants else ([] if a.constants == "none" else a.constants.split(","))
    space_c = {} if only == [] else constant_space(seen, only)
    space_s = {} if a.no_settings else dict(SETTING_SPACE)
    logf = common.CACHE / f"train_{a.label}.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with logf.open("a") as f:
            f.write(line + "\n")

    log(f"train cases {len(train_cases)}, val cases {len(val_cases)}; constants: {json.dumps(space_c)}")
    from multiprocessing import get_context

    with get_context("spawn").Pool(a.workers, initializer=bench._init_worker, initargs=(a.backend,)) as pool:
        obj = Objective(a.backend, train_cases, pool, a.cross_weight)
        t0 = time.time()
        best, bs, bc = optimise(obj, {}, {}, space_s, space_c, a.budget, np.random.default_rng(a.seed), log)
        log(f"best objective {best:.4f} after {len(obj.history)} evals in {time.time() - t0:.0f}s")
        log("settings " + json.dumps(bs))
        log("constants " + json.dumps(bc))
        # held-out validation: before vs after
        before = bench.summarize(pool.map(bench.run_one, [(c, a.backend, {}, {}, None) for c in val_cases], chunksize=1))
        after = bench.summarize(pool.map(bench.run_one, [(c, a.backend, bs, bc, None) for c in val_cases], chunksize=1))
    out = {"settings": bs, "constants": bc, "objective": best, "val_before": before, "val_after": after, "history": obj.history}
    (common.CACHE / f"train_{a.label}.json").write_text(json.dumps(out, indent=1))
    for key in ("self/val", "cross/val", "pair/all"):
        b, c = before.get(key, {}), after.get(key, {})
        log(f"{key}: " + ", ".join(f"{k} {b[k]:.3f}->{c.get(k, float('nan')):.3f}" for k in b if isinstance(b[k], float)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
