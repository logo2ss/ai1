# Lumen Studio

A desktop application for **realistic, fully local image editing**. It has three independent tools:

| Tool | What it does |
|---|---|
| **Face Replacement** | Puts a face from one photo into another. The face is re-posed, re-lit, colour-matched and blended in the gradient domain. Pairs that can't produce a realistic result are refused with an explanation. |
| **Region Swapping** | Exchanges arbitrary freehand selections between two photos. Selections are cut out with edge-aware mattes, the original content is removed by inpainting, and the inserted region is adapted to its new environment. |
| **3D Reconstruction** | Turns one ordinary photo into a navigable layered-depth 3D scene, with mesh tears at depth discontinuities and synthesised hidden background. People and foreground objects are separated from the background, and the background behind them is completed. It works in any browser: without WebGL (for example in virtual machines) the engine renders the 3D view on the CPU with the same controls. |
| **Body 3D** (in the 3D section) | Builds a detailed 3D model of the person from one photo (up to about 200k vertices). It combines a person mask, monocular depth and silhouette volume, and can be shown as a wireframe, 3D-scanner scan lines, a point cloud or a solid surface, coloured by depth, photo or hologram. Exports to GLB, PLY or PNG. |
| **Body Proportions** (in the 3D section) | Fits a lattice to the person's pose (head, shoulders, waist, hips, knees, feet). Drag the handles or use the sliders to try different proportions. Naturalisation then reshapes only the person, so walls stay straight, lets the nearby surroundings (such as shadows) follow, rebuilds the background the body uncovers, and re-matches edges, light wrap and grain. |

Everything runs on your computer. Images are never uploaded, and there are no accounts, subscriptions, telemetry or external APIs. The only network access ever made is a model download that you start explicitly. After that the app works offline.

---

## Quick start

Requirements: **Python 3.10–3.12**, **Node.js 18+**, about 2 GB of disk space and 8 GB RAM (16 GB recommended).

```bash
./scripts/setup.sh          # macOS / Linux  (add --gpu for NVIDIA CUDA)
./scripts/run.sh            # start the desktop app
```

```powershell
.\scripts\setup.ps1         # Windows  (-Gpu for CUDA, -DirectML for any DX12 GPU)
cd desktop; npx electron .
```

To skip Electron and use the same UI in your own browser instead, run `./scripts/run-browser.sh` and open the `http://127.0.0.1:…/?token=…` link it prints. Keep that terminal open. If the engine ever stops unexpectedly, the script restarts it with the same key and the open page reconnects by itself.

**Compatibility mode (CPUs or VMs without AVX).** MediaPipe's prebuilt binaries need the AVX instruction set, which many virtual machines hide. On such machines MediaPipe dies with "illegal instruction"; this was reproduced with CPU emulation. The engine checks for AVX at start-up. Without it, or if MediaPipe crashes for any other reason, face replacement switches automatically to a MediaPipe-free path: the YuNet face detector plus 68-point LBF landmarks, expanded to the same 478-point mesh. Against MediaPipe its landmarks have a median error of 3.5% of the eye-to-eye distance, and its head pose has an average error of 3–4°. Smart select, the body editor and Body 3D use ONNX segmentation models instead (U2-Net "silueta" for objects, U2-Net human segmentation for people), so every feature works in compatibility mode. To get every feature in a VM, set its CPU model to *host-passthrough*.

MediaPipe (face, body and smart-select detection) runs in an isolated helper process with GPU drivers hidden, because inference is CPU-only. If its native code crashes, which can happen in virtual machines with virtual GPUs, you get a clear error and the engine keeps running. Set `LUMEN_MEDIAPIPE_GPU=1` to let it see the GPU drivers.

On Linux, MediaPipe needs `libEGL`/`libGLES` (`sudo apt install libegl1 libgles2`), which every desktop install already has.

### Models (one-time download, then offline)

`setup.sh` installs the recommended models. You can also manage them in the app (**Models** page) or from the command line:

```bash
python scripts/download_models.py --list
python scripts/download_models.py --all                        # include optional models
python scripts/download_models.py --import lama ./lama_fp32.onnx   # air-gapped install
```

| Model | Size | License | Used for |
|---|---|---|---|
| MediaPipe Face Landmarker | 3.7 MB | Apache-2.0 | Face detection, 478 3D landmarks, head pose |
| Depth Anything V2 Small (ONNX) | 99 MB | Apache-2.0 | Monocular depth (default) |
| MiDaS v2.1 Small (ONNX) | 67 MB | MIT | Fast depth for low-end machines |
| MediaPipe Interactive Segmenter | 6.2 MB | Apache-2.0 | One-click "smart select" |
| MediaPipe Pose Landmarker (Full) | 9.4 MB | Apache-2.0 | Body keypoints and person mask for the body editor |
| YuNet Face Detector (ONNX) | 0.2 MB | MIT | Face detection in compatibility mode (no MediaPipe) |
| Silueta (U2-Net, ONNX) | 44 MB | Apache-2.0 | One-click smart select on every CPU (no MediaPipe) |
| U2-Net Human Segmentation (ONNX) | 176 MB | Apache-2.0 | Person mask for Body 3D and the body editor in compatibility mode |
| OpenCV LBF Facial Landmarks | 56 MB | research use (300-W) | 68-point landmarks in compatibility mode |
| LaMa (ONNX, optional) | 208 MB | Apache-2.0 | Learned inpainting for large removed areas |

Every model is checked against a pinned SHA-256 before it is installed (LaMa is checked by loading the graph instead). If a model is missing, the app says which one to install. It never falls back to an online service. Depth Anything V2 Base/Large produce better depth but are licensed CC-BY-NC-4.0, so they are not included.

### Hardware acceleration

Neural inference uses **ONNX Runtime**, which detects these backends automatically:
- CPU (always available)
- **NVIDIA CUDA/TensorRT** (`onnxruntime-gpu`)
- **DirectML** on any DX12 GPU (`onnxruntime-directml`)
- **CoreML** on Apple Silicon

Choose a device under Settings. If a GPU driver fails, the app falls back to CPU rather than failing. The classical algorithms (Poisson solves, PatchMatch, matting) run multi-threaded on the CPU through NumPy, SciPy, OpenCV, numba and pyamg.

---

## Architecture

```
lumen-studio/
├── engine/                 Python processing engine (local HTTP API on 127.0.0.1)
│   └── lumen_engine/
│       ├── imaging/        colour science, Poisson & multiband blending, matting,
│       │                   noise/sharpness analysis, PatchMatch inpainting
│       ├── face/           landmarks & pose, mesh warping, compatibility analysis, swap pipeline
│       ├── region/         selection refinement, environment analysis, region exchange, smart select
│       ├── depth/          ONNX depth estimation (TTA, multi-crop detail fusion, edge snapping)
│       ├── recon/          layered-depth scene reconstruction
│       ├── models/         model registry + explicit download/import/verify
│       ├── server/         FastAPI app, background job queue
│       ├── hardware.py     device detection, ONNX Runtime sessions
│       └── store.py        local temporary storage of images/results
├── frontend/               React + TypeScript + Tailwind + Three.js UI
│   └── src/  pages/ (Dashboard, Face, Region, Recon, Models, Settings)
│             components/MaskEditor.tsx (lasso/polygon/brush/eraser/smart select, undo, zoom)
│             three/ (scene parser, unprojection shaders, viewer, GLB/PLY export)
├── desktop/                Electron shell (spawns the engine, enforces the offline policy)
├── scripts/                setup, model download, packaging, test data
└── docs/TECHNICAL.md       algorithms, design decisions and known limitations
```

**Why this stack?**
- **Python engine:** the best ecosystem for computer vision (OpenCV, SciPy, ONNX Runtime, MediaPipe, numba).
- **Electron:** ships the same Chromium/WebGL everywhere, which matters for the Three.js viewer. Tauri would need a Rust toolchain and gives no benefit here because the heavy work happens in Python.
- **ONNX Runtime instead of PyTorch:** one 50 MB runtime with CPU/CUDA/DirectML/CoreML backends, instead of a multi-gigabyte framework.

**Offline and security guarantees:**
- The engine binds to **127.0.0.1 only** and requires a random per-session token on every request.
- The UI is served with a strict Content-Security-Policy (`connect-src 'self'`).
- Electron cancels every request that isn't for the local engine.
- Temporary files stay in `~/.lumen-studio/work` and are deleted after 7 days, or on demand from Settings.

### Packaging installers

```bash
./scripts/build_engine.sh          # freezes the engine with PyInstaller → engine/dist/lumen-engine
cd desktop && npm run dist         # electron-builder → dmg / nsis / AppImage
```

---

## Testing

```bash
python scripts/fetch_testdata.py            # public test photos (faces, indoor/outdoor scenes)
cd engine && python -m pytest -q            # 36 tests: algorithms, pipelines, API, crash isolation, compatibility mode
```

The tests check measurable properties, not just "runs without error":
- Poisson boundary conditions hold.
- Pixels outside the edited region stay bit-identical in both face and region editing.
- PatchMatch reconstructs periodic texture.
- Grain matching hits the measured noise level.
- Skin ΔE goes down after face replacement.
- Strongly different head poses are refused.
- The 3D container is consistent.
- A missing model produces the "install this model" error.

---

## Honest limitations

- **Face replacement** handles head-pose differences up to about 20° well. Beyond about 38° yaw or 32° pitch, or with self-occlusion, the pair is refused. Removing hard shadows from a side-lit source isn't reliable either, so that is refused too (adding shading to a flat-lit source works well). If the source face is partly covered by something skin-coloured (a straw hat brim, for example), the cover can't always be detected. Hair and glasses are usually detected.
- **Region swapping** can't infer depth order: if you select a ball behind a net, the net is part of the selection. Exemplar inpainting keeps real texture but can't always continue thin structures (lines, nets, rails) through large holes. The optional LaMa model improves this. Content enlarged more than 2.5× is softer than its surroundings, and the app warns about it rather than inventing detail.
- **3D reconstruction** from one photo can't recover surfaces that were never seen. Areas hidden behind objects and outside the frame are estimated (shown in pink in the *Layers* view). The viewer keeps the camera within a reliable range by default. Gaussian splatting and NeRF-style neural rendering need many views, or large generative models that don't run well on typical desktops offline, so the app uses a layered-depth mesh instead. See `docs/TECHNICAL.md`.

### Verified in this build

- **Tests:** 36/36 engine tests pass, including a run inside a network namespace with **no network access**. That run covers face detection and replacement, depth, 3D reconstruction, region exchange and the API.
- **Before/after:** every tool has a before/after view: face, region results, 3D scene (photo over the live 3D view), Body 3D (photo beside the model) and body proportions.
- **UI:** every workflow was driven end to end in Chromium with Playwright: loading images, detecting faces, compatibility analysis, replacing a face, drawing lasso and smart selections, undo, exchanging regions, reconstructing 3D, orbiting and the layers view.
- **Electron:** the shell starts its own engine and renders the app.
- **Models:** model download with SHA-256 verification was checked against the real download sources.
- **Not exercised:** the LaMa ONNX path is implemented against Carve/LaMa-ONNX's documented input/output, but Hugging Face wasn't reachable from the build environment, so it couldn't be run. If it fails at runtime, the engine falls back to PatchMatch and reports which method it used.
