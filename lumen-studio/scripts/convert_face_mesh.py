#!/usr/bin/env python3
"""Convert MediaPipe's face-mesh model to ONNX for the compatibility face path.

    python scripts/convert_face_mesh.py                 # ~/.lumen-studio/models/face_landmarks_detector.onnx
    python scripts/convert_face_mesh.py --task path/face_landmarker.task --out model.onnx
    python scripts/convert_face_mesh.py --data          # (developers) rebuild face/data/face_geometry.npz

Computers without AVX cannot run MediaPipe's prebuilt runtime, but ONNX Runtime
works everywhere. The face-mesh network inside the already-downloaded
``face_landmarker.task`` (Apache-2.0) is converted with Lumen Studio's own
converter (no extra packages, no network); the output is deterministic. The
engine also performs this conversion automatically on first use, so running
this script is optional - it is called by setup through download_models.py.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from lumen_engine.face import tflite_onnx
from lumen_engine.face.meshnet import onnx_path
from lumen_engine.models import manager


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=Path, default=None, help="face_landmarker.task (default: the installed model)")
    ap.add_argument("--out", type=Path, default=None, help="output .onnx (default: the models folder)")
    ap.add_argument("--force", action="store_true", help="convert even if the output exists")
    ap.add_argument("--check", action="store_true", help="run the converted model once with ONNX Runtime")
    ap.add_argument("--data", action="store_true", help="also rebuild engine/lumen_engine/face/data/face_geometry.npz")
    a = ap.parse_args()
    task = a.task or manager.model_path("face-landmarker")
    out = a.out or onnx_path()
    if not task.is_file():
        print(f"✘ {task} not found - install 'MediaPipe Face Landmarker' first (python scripts/download_models.py)", file=sys.stderr)
        return 1
    if out.is_file() and not a.force:
        print(f"✔ {out} already exists (use --force to convert again)")
    else:
        tflite_onnx.convert_face_mesh(task, out)
        print(f"✔ face mesh converted to ONNX: {out} ({out.stat().st_size / 1e6:.1f} MB, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()})")
    if a.check:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        res = sess.run(None, {"input": np.full((1, 256, 256, 3), 0.5, np.float32)})
        print("✔ ONNX Runtime check:", ", ".join(f"{o.name}{tuple(r.shape)}" for o, r in zip(sess.get_outputs(), res)))
    if a.data:
        import numpy as np

        meta = tflite_onnx.read_geometry_metadata(tflite_onnx.read_task_member(task, "geometry_pipeline_metadata_landmarks.binarypb"))
        dest = Path(tflite_onnx.__file__).parent / "data" / "face_geometry.npz"
        np.savez_compressed(dest, **meta)
        print(f"✔ wrote {dest} ({len(meta['vertices'])} vertices, {len(meta['triangles'])} triangles, {len(meta['basis_ids'])} pose landmarks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
