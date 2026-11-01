# Lumen Studio 1.4.1 — refinement of the original application

This version returns to the supplied **lumen-studio-1.4.0.tar.gz** toolset. The user's subsequent request explicitly supersedes the feature-addition priorities in the original HANDOFF.md. **Person 3D and Pure swap have been removed**, including their UI, endpoints and implementations. The original Face Replacement, Region Swapping, 3D scene, Body 3D and Body proportions tools remain. The original Face page, reconstruction tabs and visual design are retained. README.md is preserved from the original archive; the separately supplied HANDOFF.md is unchanged.

The work changes processing accuracy, edge quality, geometry continuity and stability. No new feature, AI service or model was added. Inference remains local ONNX. No PyTorch, pickle or TorchScript model was introduced.

## 3D map and geometry

- **Consistent relative depth.** Flip and detail-crop predictions have independent scale/offset. They are now aligned with a robust positive affine fit before fusion. Reversed, constant or poorly correlated predictions are rejected. Crop corrections are confidence-weighted and bounded to avoid creating unsupported depth spikes.
- **Fine global structure survives detail fusion.** A capped detail pass now upsamples its correction and adds it to the original high-resolution global estimate. Previously it replaced that estimate with an upsampled low-resolution map, erasing thin depth structures.
- **Sky evidence reaches the reconstruction.** The scene builder now uses the estimator's raw sky confidence instead of only its normalized depth. This prevents a far white wall from being classified as sky merely because it occupies the minimum of normalized depth. Sky also remains connected to image borders after morphological cleanup.
- **People and objects retain relief.** Segmentation now precedes plane regularization. Confident subject pixels are protected from floor/wall flattening, including the uncertainty band at their outlines.
- **Unrelated objects cannot exchange depth.** Silhouette cleanup now works per connected component. A thin object without an eroded interior previously took depth from another thicker object anywhere in the photo, moving it forward incorrectly.
- **Reflective floor pockets.** Larger enclosed far-depth pockets can be regularized on strongly supported floor planes. This does not flatten nearer foreground objects or close equivalent openings in walls. This is a bounded geometric prior, not proof that every enclosed floor opening is a reflection.
- **Outpainting uses its existing safeguards.** The reconstruction now actually passes foreground-avoidance and sky masks into outpainting. The helper already supported those constraints, but the caller omitted them, allowing faces/people and ground texture to be copied into inappropriate margins.
- **CPU preview topology.** Coarse interior geometry is retained for speed, with fine quads restored around cuts. Uniform downsampling previously erased narrow rails/limbs and enlarged cracks during camera motion, even when the full frame contained them.
- **Sparse background depth.** Missing background samples (`-1`) no longer count as sky in environment generation. Main depth supplies sky semantics where the background layer is absent.
- **Existing Body 3D gaps.** Only tiny enclosed segmentation specks are filled. Actual gaps between arms/legs remain open, instead of becoming sheets textured with the original background.

The earlier corner-continuation fix and completed-background environment are retained. The harness retains ±35°/±50° views and now fingerprints processing source files in its cache keys, preventing stale reconstructions from being mistaken for results of a new algorithm.

## Face replacement

The original interface and settings remain. Processing retains the improved semantic/geometric mask coverage, consistent diffuse/specular comparison, and single-pass parsing from the previous quality work. Mirror completion remains limited to meaningful pose changes.

This pass additionally protects source eyes, lips and brows from smoothing driven by **skin** sharpness statistics. Fine-detail attenuation is restricted to skin rather than the entire face. Relighting statistics now use skin visible in both photographs, excluding source/target occluders. A bounded Lab correction matches only the transition band to the surrounding skin while preserving the face core and eyes/lips. Monochrome targets remain monochrome. Multiband corrections no longer tint distant pixels outside the active face mask. The existing B&W QA case now runs as a required compat regression instead of an expected failure.

These corrections preserve source detail and reduce artifacts; they cannot reconstruct missing source facial detail or make arbitrary head poses compatible. The full benchmark includes difficult cases, successful-case counts and errors; see VALIDATION-1.4.1.md.

## Region swapping

Transformed objects are filtered and warped in **premultiplied alpha**, then unpremultiplied. Resampling straight RGB previously mixed background or zero-filled pixels into the foreground and multiplied them by alpha again, producing dark/coloured fringes. A constant-colour cutout now keeps its correct edge colour even with a hostile source background and fractional scaling/translation.

Area-mode sampling stays within the selected source region. Existing selection refinement, target-content removal, lighting adaptation and blending remain. Enlarging a small source still cannot recover detail, and unrelated foreground occluders are not automatically reconstructed.

## Compatibility and stability

The direct face APIs and body editor honor the ONNX compatibility path without requiring MediaPipe Pose. A queued image-size observer callback is now guarded during tool unmount, preventing errors when switching tools quickly. This changes no styling or layout. Frontend dependency versions are unchanged.

## Run on Arch / offline use

From `lumen-studio`:

```bash
# One-time setup with Internet. Python 3.10–3.12, Node/npm; uv can supply Python 3.12.
bash scripts/setup.sh --compat

# Then run offline, including on the no-WebGL/no-AVX machine.
LUMEN_ASSUME_NO_AVX=1 LUMEN_FACE_BACKEND=opencv LUMEN_THREADS=2 NUMBA_NUM_THREADS=2 NUMBA_CPU_NAME=generic bash scripts/run-browser.sh
```

Open the address printed by the script and keep its terminal open. Reuse `~/.lumen-studio/models`, including existing `lama_fp32.onnx` and `migan_pipeline_v2.onnx`. No new weights are needed. Sources/licences and verification remain in `models/registry.py`.

In the prepared cloud machine use `LUMEN_HOME=/workspace/lumen-data` and `LUMEN_MODELS_DIR=/workspace/lumen-data/models`. Optional learned-inpainting downloads were blocked there; real-image validation used the built-in fallback. Learned-model plumbing is tested, but real LaMa/MI-GAN output was not assessed in this instance.

## Reproduce validation

Export the compat variables above and the appropriate models directory first:

```bash
.venv/bin/python -m pytest -q -rs engine/tests/
(cd frontend && npm run build)

# In one terminal:
.venv/bin/python -m lumen_engine.server.main --port 8820 --static frontend/dist --token t
# In another, optional Playwright development dependency and installed Chromium:
.venv/bin/python -m pip install playwright
.venv/bin/python scripts/e2e_cpu.py

# Real-photo review:
.venv/bin/python scripts/validate_visuals.py faces
.venv/bin/python scripts/validate_visuals.py regions

# Quality harnesses:
(cd engine && ../.venv/bin/python -m tools.recon_eval --scenes std --tag quality141)
(cd engine && ../.venv/bin/python -m tools.face_train.bench --backend compat --label quality141)
```

The browser test asserts that only the original reconstruction tabs exist, disables GPU/software rasterization, checks decoded CPU frames and actual editing jobs, and blocks external browser requests. `CHROMIUM` can select another installed Chromium executable. Passing tests or zero uncovered pixels alone do not establish realistic unseen geometry; real-scene ground-truth depth was not available. Numerical geometry regressions use known synthetic truth, and real-photo comparisons are reported separately.
