#!/usr/bin/env python3
"""Install, list or verify Lumen Studio's local models from the command line.

    python scripts/download_models.py               # install recommended models
    python scripts/download_models.py --all         # install every model (incl. optional)
    python scripts/download_models.py --optional    # try the optional models; failures only warn
    python scripts/download_models.py --optional --only migan lama
    python scripts/download_models.py --list
    python scripts/download_models.py --import lama /path/to/lama_fp32.onnx

This is the only step that needs internet access. Afterwards the application
works fully offline.

With --optional the selected models (the --only ones, otherwise every optional
model) are a best-effort extra: a model that cannot be downloaded, or an id
this version does not know, only produces a warning and the exit code stays 0.
The application works without them and uses its built-in methods instead.
"""

import argparse
import sys

from lumen_engine.models import manager
from lumen_engine.models.registry import MODELS, ModelSpec


def _without(spec: ModelSpec) -> str:
    """What the application does while this optional model is missing."""
    if "inpainting" in spec.name.lower():
        return "the built-in inpainting will be used instead"
    if spec.required:  # forced through --optional --only
        return "the features that need it will ask you to install it"
    return "Lumen Studio works without it and uses its built-in method instead"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also install optional models")
    ap.add_argument(
        "--optional",
        action="store_true",
        help="best effort: try the optional models (or the --only ones); failed downloads and unknown ids "
        "only print a warning (exit code 0)",
    )
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", help="model ids to install")
    ap.add_argument("--import", dest="imp", nargs=2, metavar=("MODEL_ID", "FILE"))
    a = ap.parse_args()
    if a.list:
        for m in manager.status():
            print(f"{'✔' if m['installed'] else '·'} {m['id']:<26} {m['size_bytes'] / 1e6:7.1f} MB  {m['license']:<11} {m['name']}")
        return 0
    if a.imp:
        manager.import_file(a.imp[0], a.imp[1])
        print("imported", a.imp[0])
        return 0
    if a.only:
        ids = a.only
    elif a.optional:
        ids = [m.id for m in MODELS.values() if not m.required]
    else:
        ids = [m.id for m in MODELS.values() if a.all or m.required]
    unknown = [mid for mid in ids if mid not in MODELS]
    if unknown and not a.optional:
        print(f"Unknown model id(s): {', '.join(unknown)}. Known: {', '.join(MODELS)}", file=sys.stderr)
        return 2
    failed = 0
    for mid in ids:
        spec = MODELS.get(mid)
        if spec is None:  # only with --optional: skip it, still try the others
            print(f"⚠ optional model '{mid}' is not known to this version of Lumen Studio - skipped", file=sys.stderr)
            continue
        try:
            if manager.is_installed(mid):
                print(f"✔ {spec.name} already installed")
                continue
            print(f"↓ {spec.name} ({spec.size_bytes / 1e6:.1f} MB, {spec.license}) …", flush=True)
            manager.download(mid)
            print(f"✔ {spec.name} installed")
        except Exception as exc:  # noqa: BLE001
            reason = getattr(exc, "message", None) or str(exc)
            if a.optional:
                print(
                    f"⚠ optional model '{mid}' could not be downloaded: {reason} — {_without(spec)}; you can install "
                    f"it later from the Models page (or with: python scripts/download_models.py --only {mid})",
                    file=sys.stderr,
                )
                continue
            failed += 1
            print(f"✘ {spec.name}: {reason}", file=sys.stderr)
            if getattr(exc, "hint", None):
                print(f"  {exc.hint}", file=sys.stderr)
    # compatibility face path (CPUs without AVX): convert MediaPipe's face mesh to ONNX locally (no download)
    if not a.list and manager.is_installed("face-landmarker") and not manager.is_installed("face-mesh-onnx"):
        try:
            manager.download("face-mesh-onnx")
            print(f"✔ {MODELS['face-mesh-onnx'].name} prepared")
        except Exception as exc:  # noqa: BLE001 - optional: the engine falls back to LBF landmarks
            print(f"⚠ face mesh conversion failed ({getattr(exc, 'message', exc)}); LBF landmarks will be used", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
