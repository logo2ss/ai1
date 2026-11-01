# Lumen Studio 1.4.1 validation

Validated on 2026-11-01 against the uploaded 1.4.0 application. Scope: original tools and layout only. No Person 3D or Pure swap. See [REFINEMENT-1.4.1.md](REFINEMENT-1.4.1.md) for exact processing changes and setup/run commands.

## Machine and compatibility

Linux cloud machine, Python 3.12, ONNX Runtime CPU execution, installed free ONNX face/depth/segmentation models. All engine checks select `LUMEN_ASSUME_NO_AVX=1 LUMEN_FACE_BACKEND=opencv`; JIT checks use `NUMBA_CPU_NAME=generic`, with two engine/Numba threads. This tests the compatibility path and portable generated code. **The physical cloud CPU has AVX; execution on actual non-AVX Arch hardware was not available.**

Chromium uses `--disable-gpu --disable-software-rasterizer`. The browser test asserts `getContext('webgl') === null`, blocks external browser requests, and requires completed jobs plus decoded CPU frames. Inference uses local models; it never automatically downloads missing weights. ONNX inference remains local. The `.task` file is checksum-verified conversion input used during setup, not an inference backend.

The user's existing LaMa/MI-GAN files were not available here. Optional download hosts were blocked by environment policy; validation used built-in PatchMatch. Tests of learned-inpainting plumbing do not establish real LaMa/MI-GAN output quality.

## Release checks

- Full engine suite: **101 passed, 8 skipped**, one Starlette/httpx deprecation warning, 205.14 seconds. Full log included in the evidence archive.
- Frontend: TypeScript and Vite production build passed for version 1.4.1.
- Browser: **all five original workflows passed**, zero page errors, zero external browser requests; final screenshots are included in the evidence archive.
- Original README and separately uploaded HANDOFF unchanged; original Face page byte-identical; original CSS unchanged. Reconstruction tabs are exactly `3D scene`, `Body 3D`, `Body proportions`.
- `git diff --check` passed.

The eight optional-model/reference skips cover real LaMa inference, MediaPipe face reference, LBF comparison/fallback cases, and MediaPipe Pose. ONNX mesh, CPU scene/body rendering and ONNX body-editor paths are exercised separately. No expected failures remain in the grayscale face case.

## Matched 3D comparison

The baseline engine was unpacked from the uploaded 1.4.0 tarball. The common evaluation harness supplies identical cameras and handles the old environment-texture signature; baseline engine processing files were unchanged. Both runs use the same installed models, photos and default balanced reconstruction settings.

Four common scenes: `demo01`, `demo16`, `demo10`, `faces_2008_002079`; 25 camera views each, including ±35° and ±50°. Means over the same four scenes:

| Existing harness measure | Original 1.4 | Refined 1.4.1 |
| --- | ---: | ---: |
| Foreground texture bleed into background plate, lower better | 10.61% | 3.61% |
| Stretched rendered pixels, lower better | 2.68% | 2.74% |
| Small-motion stretched pixels, lower better | 0.0362% | 0.0201% |
| Reference PSNR, higher better | 25.79 dB | 25.84 dB |
| Reference layer leakage, lower better | 6.30% | 6.23% |
| Streak score, lower better | 0.0870 | 0.0836 |
| Uncovered pixels | 0% | 0% |

The texture-bleed heuristic falls about **66%**. This is a specific artifact measure, **not a claim of 66% greater depth accuracy**. Overall stretching is slightly worse. Reference fidelity is essentially unchanged. The before/after sheet includes the same ±50° cameras for every common scene, including difficult results.

The final version also completed all eight standard scenes (`home`, `demo01`, `demo02`, `demo10`, `demo11`, `demo15`, `demo16`, `faces_2008_002079`): **200 full CPU frames**, with no uncovered pixels. Eight-scene averages: stretched 2.24%, plate texture bleed 3.56%, reference PSNR 26.11 dB. These eight-scene means must not be compared directly with the four-scene baseline means.

**Wide-angle realism remains incomplete.** ±50° views still show stretched surfaces, disconnected cuts and guessed hidden scenery. Zero uncovered pixels establishes coverage only. A single photograph supplies no real back/side geometry, and there is no real-scene ground-truth depth in this evaluation. The added numerical regressions instead use known synthetic geometry to verify positive relative-depth alignment, robust outlier rejection, subject preservation during plane fitting, thin-object independence, sky semantics, retained detail, reflection-pocket constraints, sparse-background handling and fine CPU preview topology. Original Body 3D now keeps real arm/leg gaps.

## Full face benchmark

Same existing 90 cases and default settings for both engines: 44 self-swaps with known augmentation, six same-person pairs, 40 cross-person swaps. The harness forces incompatible pairs to run for measurement. Successful cases are 43 + 6 + 40 in both versions.

| Aggregate loss, lower better | Original 1.4 | Refined 1.4.1 | Change |
| --- | ---: | ---: | ---: |
| Self-swap, 43 successful | 5.751 | 4.825 | −16.1% |
| Same-person pair, 6 | 5.879 | 4.962 | −15.6% |
| Cross-person, 40 | 9.060 | 8.727 | −3.7% |

Aggregate gains do not mean every aspect or pair improved:

| Diagnostic | Original | Refined |
| --- | ---: | ---: |
| Self low-frequency colour error, lower better | 1.763 | 1.817 |
| Self boundary colour error, lower better | 0.863 | 1.048 |
| Cross seam-gradient anomaly, lower better | 0.279 | 0.178 |
| Cross boundary colour step, lower better | 1.147 | 1.884 |
| Cross skin-tone error, lower better | 2.046 | 2.258 |
| Cross source-detail correlation, higher better | 0.777 | 0.811 |
| Cross target-detail leakage, lower better | 0.191 | 0.163 |

The no-reference measures use each result's mask, whose coverage changes; they are useful diagnostics, not a human realism score or a face-recognition score. All per-case results and final contact sheets are included, rather than only selected successes. Large pose/expression differences and occluders remain difficult.

Both versions fail the same case: `2009_004587.jpg@296,312/a0`, `ProcessingError: The face region is empty after masking.` This remains unresolved rather than being counted as a pass or bypassing occluder protection.

A sequential matched **19-case** stride-five speed check used one benchmark worker, two engine threads and normal parsing-cache reuse, with no competing quality jobs. Original: mean **1.918 s**, median **1.911 s**. Refined: mean **1.882 s**, median **1.917 s**. This is effectively unchanged in one run; **no major speedup is claimed**. Profiling confirms parsing/ONNX inference, dense warping and appearance fitting remain significant costs. Single-pass parsing reduces inference work on uncached photos, but the added boundary/coverage work offsets part of that saving. Full-benchmark timings taken under different loads are not used as speed evidence.

## Region review

Final real-photo examples cover an object transfer (football into a basketball scene) and an area transfer (sky between outdoor photographs). The numerical cutout regression verifies that hostile source background colours and black pixels cannot leak into fractionally transformed edges; area-mode sampling cannot borrow pixels outside the selected source region. Pixels outside the edit remain unchanged.

The sports example still looks soft after enlarging a small source and does not reconstruct correct foreground net occlusion. The sky example remains an approximation with the source texture/lighting available. These examples demonstrate remaining limits and are not labeled universally natural. No new occlusion tool or generative service was added.

## Cloud environment reuse

The current environment has working engine/frontend dependencies and installed required ONNX models. Reusable `install_script` and `start_skill` drafts have been saved using the cloud-environment-onboarding setup skill, updated for the original 1.4 toolset and version 1.4.1. This is draft persistence; publication and restoration in a new task remain unverified. Publish the prepared environment through environment settings when desired. No extra credentials are required for the tested local workflow.
