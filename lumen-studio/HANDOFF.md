# Lumen Studio v1.4.0 — handoff for the next developer

## What this is
A local, offline desktop photo editor:
- **Engine:** FastAPI + ONNX Runtime, in `engine/lumen_engine`.
- **UI:** React + TypeScript + Vite, in `frontend/src`. The optional Electron shell is in `desktop/`.
- **Running it:** the user runs browser mode with `scripts/run-browser.sh`.
- **Models** live in `~/.lumen-studio/models` and are downloaded by `scripts/download_models.py`. Their sources are listed in `engine/lumen_engine/models/registry.py`.
- **Setup:** `scripts/setup.sh` creates `.venv`, builds the frontend and downloads the models.

There are three tools:
1. **Face replacement:** `face/`.
2. **Region swapping between two images:** `region/` and `frontend/src/components/MaskEditor.tsx`.
3. **Single photo → navigable 3D scene:** `recon/` and `frontend/src/three/`. It also has Body 3D and Body proportions tools.

## The user's machine (always test for this)
- **OS:** Arch Linux VM, using browser mode.
- **No WebGL:** the 3D view uses the server-side CPU renderer. That is `recon/render.py` plus `POST /api/recon/render`, displayed by `CpuSceneViewer.tsx`.
- **CPU without AVX:** MediaPipe crashes on it, so face detection runs in "compat mode". That mode uses YuNet plus the ONNX conversion of the MediaPipe face mesh in `face/meshnet.py` and `face/tflite_onnx.py`.
  - Test the compat path with `LUMEN_ASSUME_NO_AVX=1 LUMEN_FACE_BACKEND=opencv`.
- **Installed models:** the user already has `lama_fp32.onnx` and `migan_pipeline_v2.onnx`.

## Rules
- **ONNX models only.** Never load PyTorch, pickle or TorchScript files.
- **Everything must work offline after setup.**
- **New models must be optional**, with a fallback when the file is missing.
- **Licences:** record the licence and the source URL in `registry.py`.
- **Quality:** realism matters more than anything else.

## How to test
```bash
cd engine && source ../.venv/bin/activate && python -m pytest -q tests/      # 93 tests, ~4 min
cd frontend && npx tsc -b && npx vite build
cd engine && LUMEN_ASSUME_NO_AVX=1 python -m lumen_engine.server.main --port 8820 --static ../frontend/dist --token t
# then open http://127.0.0.1:8820/?token=t ; e2e with Playwright chromium args ['--disable-gpu','--disable-software-rasterizer']
```

Evaluation tools:
- 3D novel-view harness: `python -m tools.recon_eval --scenes std --tag mytag` (run from `engine/`).
- Face benchmark: `python -m tools.face_train.bench` (run from `engine/`).

Test photos are in `testdata/faces` and `testdata/scenes`.

## Still to do (in priority order; the user asked for all of these)
1. **NEW "Person 3D" mode.** This is the user's stated focus.
   - **What the user wants:** "takes the person in the photo and removes all the background and i can have 3d angles and i can move around but its just void and the person". It must look natural at **at least ±50°**.
   - **Plan:** new files `recon/person3d.py` and `recon/person_render.py`, plus new endpoints `POST /api/person3d` and `POST /api/person3d/render`.
     - Person matte from U2-Net human seg (`imaging/u2net.py`), with refined and colour-decontaminated edges.
     - Depth from `depth/estimator.py`.
     - A closed volume from two height fields: front = mid − h, back = mid + h. h comes from silhouette inflation (circular cross-sections, h→0 at the silhouette). mid comes from low-frequency depth. Add face and cloth detail to the front only.
     - Rim texture: the front texture extended outwards.
     - CPU renderer: textured, with subtle Lambert shading on a void background, an orbit camera, and a fast preview mode.
   - **Frontend:** new `three/CpuPersonViewer.tsx` and a "Person 3D" tab in `pages/ReconPage.tsx`.
     - Controls: drag to orbit (±70°), wheel to zoom, right-drag to pan, WASD to move, a reset button, and −50/0/+50 presets.
     - Reuse the patterns in `CpuBodyViewer.tsx` and `cpuFrames.tsx`.
2. **NEW face "Pure swap" option.**
   - **What the user wants:** "take the face exactly as it is and just make the surrounding area natural to it".
   - **Plan:** new file `face/pure.py`, a `mode` field (`"natural"` or `"pure"`) in `face/settings.py`, a dispatch at the top of `swap_face` in `face/swap.py`, and a toggle in `frontend/src/pages/FacePage.tsx`.
     - Align the source face with a similarity transform only. No colour or lighting changes inside the face.
     - Shift the target's neck, ears and skin tone toward the face, with a smooth falloff.
     - Run the gradient-domain solve only on a band outside the face.
     - Inpaint leftover target contours (`imaging.inpaint.inpaint(method="auto")`).
3. **Face swap regression.**
   - On the self-swap benchmark, the new pipeline (`face/appearance.py` and the parsing masks in `face/masking.py`) scored worse than v1.3 (loss 6.0 vs 4.0). It scored better on cross-identity swaps.
   - It is also slow: about 5 s per swap.
   - Ablate parsing masks, the appearance mode (`APPEARANCE_MODE` legacy vs new) and mirror completion. Fix whatever hurts. Profile the swap.
   - Known visual issues to fix: white blotches from source highlights, face tone not matching the neck, a soft or blurry face, and source occluders such as microphones or beards being pasted in.
4. **3D scene mode must hold up at ±50°.**
   - Default drag should orbit around the subject.
   - Confirm that the far environment (`recon/environment.py`) and the wide plane-continued margins (`recon/outpaint.py`) give no black areas and no garbage, using the eval harness with ±35° and ±50° views.
   - One test was loosened: a corner seam in `_extend_depth` (`tests/test_outpaint_env.py`). Fix the seam, then restore the stricter threshold.
5. **Region swap.** Verify the naturalness of swaps (scale/fit, seams, removal of the original content) on real pairs.

After the changes: run the full test suite, build the frontend, run a no-WebGL e2e of every tool, then package with
`git archive --format=tar.gz --prefix=lumen-studio/ -o lumen-studio-1.5.0.tar.gz HEAD`.
