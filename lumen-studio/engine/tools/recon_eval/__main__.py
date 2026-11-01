"""CLI: ``python -m tools.recon_eval --scenes demo03,demo07 --tag mytag`` (run from ``engine/``)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# make ``lumen_engine`` importable when run from the repository root as well
import importlib.util  # noqa: E402

_ENGINE = Path(__file__).resolve().parents[2]
if importlib.util.find_spec("lumen_engine") is None:  # run from elsewhere: use this checkout's engine
    sys.path.insert(0, str(_ENGINE))
os.environ.setdefault("LUMEN_MODELS_DIR", str(Path.home() / ".lumen-studio" / "models"))


def _settings(spec: str | None) -> dict:
    out: dict = {}
    for kv in (spec or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    from . import STANDARD_SCENES

    ap = argparse.ArgumentParser(prog="python -m tools.recon_eval", description=__doc__)
    ap.add_argument("--scenes", default="std", help="comma list: demo03, home, faces/2008_002079, a path, or 'std' (%s)" % ",".join(STANDARD_SCENES))
    ap.add_argument("--tag", help="output tag (directory under /tmp/claude-0/recon_eval)")
    ap.add_argument("--quality", default="fast", choices=["fast", "balanced", "high"])
    ap.add_argument("--max-side", type=int, default=800, help="photo is resized to this long side before reconstruction")
    ap.add_argument("--settings", help="extra ReconSettings, e.g. 'outpaint_margin=0.15,inpainting=patchmatch'")
    ap.add_argument("--force", action="store_true", help="reconstruct again even when a cached scene exists")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--compare", help="tagA,tagB: side-by-side sheets + metric diff table")
    a = ap.parse_args(argv)
    from . import harness
    import lumen_engine

    print(f"[recon_eval] lumen_engine from {Path(lumen_engine.__file__).parent}")

    if a.compare:
        ta, tb = [t.strip() for t in a.compare.split(",")[:2]]
        harness.compare(ta, tb, harness.scene_list(a.scenes) if a.scenes not in ("std", "standard") else None)
        return 0
    if not a.tag:
        ap.error("--tag is required (or use --compare)")
    res = harness.run(harness.scene_list(a.scenes), a.tag, a.quality, a.max_side, _settings(a.settings), a.force, not a.no_sheets)
    return 0 if res else 1


if __name__ == "__main__":
    raise SystemExit(main())
