# Lumen Studio 1.5 — changes, running and validation

This release starts from the supplied **lumen-studio-1.4.0.tar.gz**, with the supplied HANDOFF.md retained unchanged. The enclosing GitHub repository's original files have been preserved. No paid service, remote inference, PyTorch, pickle or TorchScript model was added.

## Run on the user's Arch VM (browser, no WebGL, no AVX)

From the extracted `lumen-studio` folder, run setup once with Internet access:

```bash
bash scripts/setup.sh --compat
```

Python 3.10–3.12 and Node/npm are needed. The setup script can use an installed `uv` to obtain Python 3.12 on a rolling-release distro. `--compat` installs the ONNX workflow and skips Electron and unused MediaPipe/LBF models. The small MediaPipe `.task` archive is read only to convert the face graph into ONNX; it is not executed by a TFLite or MediaPipe interpreter in compat mode. Downloads retain TLS and checksum verification. The converted graph now has its own pinned checksum too. The face archive has a commit-pinned mirror which is checked against the original upstream SHA-256.

Then run, including when disconnected:

```bash
LUMEN_ASSUME_NO_AVX=1 LUMEN_FACE_BACKEND=opencv bash scripts/run-browser.sh
```

The script prints the local browser address. Keep its terminal open. Models are reused from `~/.lumen-studio/models` by default; keep the existing `lama_fp32.onnx` and `migan_pipeline_v2.onnx` there. Nothing needs to be re-downloaded when running offline. In the prepared cloud instance, use `LUMEN_HOME=/workspace/lumen-data` and `LUMEN_MODELS_DIR=/workspace/lumen-data/models` instead. Set `LUMEN_THREADS=4 NUMBA_NUM_THREADS=4` to avoid oversubscribing the VM.

Optional LaMa/MI-GAN downloads were denied by this cloud instance's network policy. This does not disable editing: the tested fallback is built-in PatchMatch. The user's existing local ONNX files will be selected by `inpaint(method="auto")`. Learned-inpainting output was **not validated in this cloud instance**. No new optional model is required by these changes. Sources and licences remain in `engine/lumen_engine/models/registry.py`.

## 1. Person 3D

Open **3D Reconstruction → Person 3D**, load a near-frontal person photo and choose **Build Person 3D**. The mode always uses CPU rendering, including on browsers with no WebGL. Drag to orbit up to ±70°, right-drag to pan, wheel to zoom, WASD/arrows to move, R or Reset to reset, and use −50°/0°/+50° presets. Use **Load another photo** to replace the input and **Save view** to export a frame.

New files: `recon/person3d.py`, `recon/person_render.py`, `three/CpuPersonViewer.tsx`, `pages/Person3D.tsx`; authenticated build/render endpoints are `/api/person3d` and `/api/person3d/render`.

The builder refines the person matte, decontaminates edges, preserves limb gaps, makes rounded cross-sections and joins front and back surfaces with an explicit rim. Texture at the rim comes from foreground pixels. Optional photo depth and conservative ONNX face relief add front detail; absent depth falls back to silhouette volume. Existing human segmentation is preferred, general object segmentation is a fallback, and a transparent PNG can supply the matte directly. Rendering uses a perspective-correct CPU triangle rasterizer, z-buffer, subtle shading and adaptive frame scheduling.

**Quality limit:** this is an inferred closed relief volume, not a recovered human scan. Ears, side texture, hands, folded limbs and cropped bodies can still look stretched or inflated at ±50°. The real-photo contact sheet shows that limitation. Exact unseen anatomy cannot be claimed from the validation performed here.

## 2. Pure swap

Open **Face Replacement**, load source and target and enable **Pure swap**, then Replace face (or Regenerate). Natural-mode controls are hidden while Pure is selected.

`face/pure.py` applies a single similarity transform; the protected face is never colour-matched, relit, reshaped with a dense warp, camera-matched, pyramid-blended or Poisson-solved. Its mouth remains the source mouth. Alignment necessarily resamples pixels; “unchanged” means bit-exact preservation of those aligned pixels through subsequent processing. The report exposes `pure.max_face_change` (tested as zero).

Exposed target contours are inpainted with `auto`, neck/ear/nearby skin is adapted towards source tone, and a Dirichlet solve operates only outside the protected face. Tests cover rotated/scaled alignment, unchanged protected pixels and unchanged distant pixels.

**Quality limit:** preserving the face exactly cannot remove its original hard shadows, missing detail or occluders. Large pose, exposure, jaw or lighting mismatches can still leave a visible join, especially at the forehead and chin. Similar head pose and lighting matter.

## 3. Natural face quality and speed

The benchmark was ablated with parsing off, legacy appearance and mirror completion off. The main self-swap regression came from a doubly inset parsing transition. Disabling parsing helped self-swaps but hurt cross-person results, so parsing remains enabled for occluder protection. The revised mask combines a narrower seam band and geometric coverage with semantic exclusion. The target's hair/glasses and the source's detected non-face regions remain protected.

Appearance transfer now compares diffuse albedo on both images and restores highlights once; a new identity/highlight regression test guards against double highlighting. Parsing uses one aligned inference by default (flip augmentation remains available), and mirror completion is avoided for similar head poses. Direct landmark APIs now also honor no-AVX mode before any MediaPipe import; previously some older call paths bypassed the compatibility gate.

On the initial matched nine-case sample (3 self, 3 same-person, 3 cross-person), before → revised loss was **7.19 → 5.98** for self and **9.76 → 9.61** for cross-person. Lower is better. This is a small sample, not a claim that the full benchmark now beats v1.3 or every difficult swap is fixed. Concurrent-ablation timing is unsuitable for a speedup claim. A separate sequential comparison of the same nine cases measured mean swap time **1.36 → 1.11 s** (18% less time); see VALIDATION-1.5.md for scope.

## 4. Scene navigation and ±50° evaluation

Default drag remains orbit; safe orbit now actually permits ±50°. The old ~17° yaw cap and a second camera-position clamp prevented that. Pan distance is constrained independently. The far environment uses the completed background texture when supplied, rather than repeating foreground subjects. Both CPU and WebGL requests carry that background; the CPU path is the validated one.

Depth continuation now uses the fitted border intercept and smooths even the first continuation level, removing the corner step. The corner test's limit was restored from 0.15 to **0.08**. The novel-view harness now includes both signs of 35° and 50° (25 viewpoints per scene).

The eight-scene evaluation had **zero uncovered/void pixels in 200 frames**. This is coverage evidence, not a realism score: some room/furniture, occlusion and outpainted surfaces still stretch or look estimated at wide angles. **The strict “no garbage at ±50° for arbitrary scenes” quality goal is not fully met.**

## 5. Region swaps and compatibility UI

Area mode now restricts valid samples to the source selection, instead of the entire source image. It warns and retains uncovered destination content if that selection cannot cover the destination. A regression test catches colour leaking from outside an irregular source selection. Existing fit, alpha refinement, edge decontamination, original-content removal and environment harmonisation remain in use.

Real-photo sports-object and outdoor-sky pairs were rendered for review with `scripts/validate_visuals.py regions`. The sky example can be made coherent with a clean sky selection. The ball example exposes limitations: enlargement cannot recover source detail, and a net that should pass in front of the new ball is not automatically restored. Fine foreground occlusion needs a separate selection/edit; arbitrary region pairs are not certified natural.

Browser testing also found and fixed a pre-existing Body proportions UI gate that required MediaPipe Pose even when the ONNX human-segmentation fallback was installed. The editor now starts from that fallback without the pose file.

## Reproduce checks

Run these commands with the no-AVX variables above and the correct local models directory:

```bash
# Full suite from project root. generic JIT target adds a portable-CPU check.
NUMBA_CPU_NAME=generic .venv/bin/python -m pytest -q engine/tests/
(cd frontend && npm run build)

# Start the local test service, then run the browser test in a second terminal.
.venv/bin/python -m lumen_engine.server.main --port 8820 --static frontend/dist --token t
.venv/bin/python scripts/e2e_cpu.py

# Optional development tool for the browser test:
.venv/bin/python -m pip install playwright
# Use an installed Chromium executable. Default /usr/bin/chromium; override CHROMIUM.

# Real-photo contact sheets (test photos downloaded once by scripts/fetch_testdata.py):
.venv/bin/python scripts/validate_visuals.py person
.venv/bin/python scripts/validate_visuals.py faces
.venv/bin/python scripts/validate_visuals.py regions

# Harnesses are run from engine/:
(cd engine && ../.venv/bin/python -m tools.face_train.bench --backend compat --label myrun)
(cd engine && ../.venv/bin/python -m tools.recon_eval --scenes std --tag myrun)
```

The browser test starts Chromium with `--disable-gpu --disable-software-rasterizer`, asserts WebGL is absent, blocks external requests, checks actual job results and decoded CPU frames, tests both face modes, region drawing/swap, scene, Person 3D, Body 3D, and body-proportions editing. No screenshot or successful build alone is treated as a functional test.

Current validation logs and pictures are in `validation/` in the prepared workspace (generated, not source-controlled). The final validation summary is in `docs/VALIDATION-1.5.md`. The suite contains explicit skips for unavailable optional/backends; those are not counted as passes. The cloud CPU has AVX physically; the specified compatibility flags plus generic JIT exercise the requested path, not an actual emulated non-AVX processor.

## Cloud reuse and packaging

The install script and startup instructions are saved in the cloud environment draft. Review/save the settings and publish the prepared snapshot in the environment UI to reuse it. No publication or new-task restoration has been claimed. The application commit is local and has not been pushed to GitHub; a fresh clone of the original main branch alone does not contain this application. Reuse requires the prepared snapshot or the supplied source archive. Local-only commit restoration is not established. Use the existing checkout in future tasks; each cloud task is already isolated, so do not create a Git worktree unless explicitly requested.

Because the application is a subdirectory of the enclosing repository, a single-root source package is made with:

```bash
git archive --format=tar.gz --prefix=lumen-studio/ -o /workspace/artifacts/lumen-studio-1.5.0.tar.gz HEAD:lumen-studio
```

The source package does not include dependency caches, local test photographs, model weights or the generated frontend build. Setup prepares those once, after which browser editing runs offline. Keep HANDOFF.md and this change report together.
