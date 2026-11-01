# Validation of 1.5.0

Validated on 2026-11-01 in the managed Linux cloud instance, Python 3.12, ONNX Runtime CPU. Every application/test command used `LUMEN_ASSUME_NO_AVX=1 LUMEN_FACE_BACKEND=opencv`. The final suite and release server also used `NUMBA_CPU_NAME=generic`. This tests the compatibility route and portable JIT, not physical non-AVX hardware.

## Required checks

| Check | Observed result |
| --- | --- |
| Full engine suite | **90 passed, 10 skipped**, one FastAPI/Starlette deprecation warning; 260.44 s |
| Frontend | `tsc -b && vite build` passed after each priority item and after the body-UI fix |
| No-WebGL browser e2e | All seven workflows passed, decoded CPU frames checked, **0 page errors / 0 external requests** |
| Release health | `ok=true`, `version=1.5.0`, `offline=true` |
| Repeat setup | `bash scripts/setup.sh --compat` passed; optional inpainting downloads denied, fallback retained |
| Fresh model install | Commit-pinned mirror matched original SHA-256; derived ONNX matched its pinned SHA-256 |
| Scene harness | Eight standard scenes × 25 viewpoints, including ±35° and ±50°; zero uncovered pixels |
| Dependency lock integrity | Only root project versions changed; locked dependencies unchanged |

Browser workflows: Natural face swap and PNG export; Pure swap with exact aligned face pixels; region drawing and both swap directions; scene reconstruction and CPU frames; Person 3D presets/keys/zoom/reset; Body 3D build/CPU frames; Body proportions detection/preview/apply. Browser launch disabled GPU and software rasterization and the test blocked external destinations. API inference uses installed local models. The final app is still running in the prepared instance on loopback for internal validation; future tasks must restart it using the saved startup instructions.

Explicit skips remain for LBF-only, optional model and MediaPipe-reference checks whose prerequisites are not installed/usable in this ONNX compat setup. They are not counted as successes. A previous complete run before the region support test was added had 89 passed and 10 skipped. Earlier failures exposed direct APIs bypassing the no-AVX guard and an outdated harness camera count; both were fixed and the complete suite rerun. E2e also exposed the Body proportions pose-model gate, which was fixed and verified by actual preview/apply.

## Controlled face comparison

Original supplied 1.4 code and revised code were run **sequentially**, one worker, two inference threads, generic JIT target, with the same nine case IDs (3 self, 3 same-person and 3 cross-person) and installed models. Detection/materialization is outside the benchmark swap timer. This is a small sample; do not extrapolate to the full benchmark or a different CPU.

| Kind | Original loss ↓ | Revised loss ↓ | Original swap time | Revised swap time |
| --- | ---: | ---: | ---: | ---: |
| self/all | 7.193 | 5.984 | 1.510 s | 1.139 s |
| pair/all | 6.132 | 5.758 | 1.637 s | 1.375 s |
| cross/all | 9.764 | 9.613 | 0.925 s | 0.827 s |

Mean timed swap: **1.357 → 1.114 seconds**, 17.9% less time on this sample. The earlier ablations disabled parsing, used legacy appearance, or disabled mirror completion; parsing-off improved self loss but worsened cross-person loss. Parsing is retained with revised coverage. The original v1.3 benchmark score is not directly re-established or beaten by this test.

## Visual limits / outstanding quality goals

- Person 3D is a closed rounded approximation. Real-photo ±50° views still show inferred side textures and imperfect ears/limbs; it is not an accurate human scan.
- Scene coverage is complete, but estimated and stretched surfaces remain at large camera motions. Zero void pixels does **not** establish the requested “no garbage” or naturalness for all scenes.
- Pure swap preserves the aligned source exactly; markedly different lighting and shapes can leave joins despite adapting only the surrounding skin.
- The real-photo ball test shows softness from enlargement and missing foreground-net preservation. Sky/area insertion needs a clean source selection. Arbitrary region pairs are not guaranteed natural.
- LaMa/MI-GAN learned-model output was untested here because those optional downloads were blocked. Existing user weights remain supported.

See CHANGES-1.5.md for exact changes and commands. Generated screenshots, contact sheets, benchmark JSON and logs are supplied separately in the validation archive; they are not test substitutes.

The release commit is local and was not pushed. Environment draft scripts/start instructions are saved, but publication and fresh-task restoration are unverified. The source archive is the portable handoff if the snapshot cannot restore the local checkout.
