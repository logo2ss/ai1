# Face-swap "training": benchmark, tuning and learned colour model

Training a large face-swap network is not possible on a 4-core CPU without AVX in a few hours, so
"training" here means: **(1)** a benchmark with ground truth, **(2)** data-driven optimisation of the
pipeline's settings/constants on it, and **(3)** a small learned model where it beats hand-tuned
transfer. Everything is numpy/OpenCV; learned weights are plain arrays (`.npz`, no pickle).

Run from `engine/` (wrap CPU-heavy commands in the shared slot wrapper on the build machine):

```bash
python -m tools.face_train.bench --backend compat --label mytest [--save DIR] [--post]   # ~1 min at 0.5 s/swap
python -m tools.face_train.learn_color --augs 8          # refits lumen_engine/face/data/learned_color.npz (~2 min)
python -m tools.face_train.train --budget 60             # tunes FaceSwapSettings + settings.tv() constants
```

`FACE_TRAIN_CACHE` sets the cache/results directory (default: system temp). `--backend compat` runs the
no-AVX path the user's machine gets (YuNet + face mesh via ONNX), `lbf` the older LBF landmarks,
`mediapipe` the full MediaPipe path. `--ablate noparse,legacy,learned` switches components off/on for
attribution; `--stride N` / `--limit N` take subsets.

## 1. Benchmark (`bench.py`, `common.py`)

* **Faces**: `faceset.json` freezes 22 faces from `testdata/faces` (size >= 55 px, |yaw| <= 40 deg, at most
  3 per photo) by bounding-box centre, so the case list stays the same when the landmark code changes.
  Crops are downscaled to a face size of at most 200 px.
* **Self-swaps with ground truth (44)**: the source is the target photo with known changes
  (white balance, tint, exposure, gamma, contrast, saturation, directional relighting from the 3D face mesh
  - Lambert(new light)/Lambert(frontal) - plus a soft gradient and coloured light, blur or sharpening,
  noise, JPEG 35-92, scale 0.4-2x, rotation +-12 deg, translation). The ideal output is the target.
  Metrics inside the replaced area: `de_lf` (CIEDE2000 after a 1.2 %-of-face blur = colour/shading),
  `shade` (large-scale log-luminance error), `hf_fine/mid/low` (|log| texture-energy ratio per band:
  blur *and* over-sharpening are penalised; pixel-wise error would reward blur), `noise`, `seam_de` and
  `seam_grad` on the transition band, `cov` (replaced fraction of the face oval).
* **Same-person pairs (6)**: the three Obama photos swapped into each other; only large-scale colour and
  lighting are comparable (`p_de_ls`).
* **Cross-identity pairs (40, no reference)**: seam gradient anomaly (`x_seam`), colour step across the
  boundary relative to the target's own (`x_seam_step`), skin tone vs. the target face (`x_tone`;
  confounded by beards/make-up), noise/sharpness consistency, lighting-direction difference (`x_light`),
  chroma/luminance blotchiness (`x_blotch_*`), identity retention/leak (`x_id_src`, `x_id_leak` =
  how much of the *target's* mid-frequency facial detail shows through).

Loss weights are in `common.selfswap_loss` / `common.noref_loss`. Validation images (never fitted):
`VAL_IMAGES` in `common.py`.

## 2. Learned colour/lighting model (`learn_color.py` -> `lumen_engine/face/learned.py`)

Ridge regression over a stack of transfer operators (local target-minus-source skin means at 6/12/25 % of
the face size per region, global mean, Reinhard contrast, histogram matching, multiplicative-lighting
detail/chroma terms, local contrast) predicting the low-frequency CIELAB correction. Statistics use skin
only, at scales >= 6 % of the face, so the target's features (identity) do not leak. Fitted on ~190k pixels
from 8 augmentations x 12 training faces; ridge strength chosen on the validation faces.
Low-frequency Lab error of the raw warp 9.1 -> 1.7 on validation faces (train 1.6: no overfitting).
Inference ~0.05-0.1 s. Used by `swap_face` when `FaceSwapSettings.learned_color` is on.

## 3. Results (compat backend, 90 cases)

| pipeline | self loss | de_lf | pair dE | cross loss | identity leak |
|---|---|---|---|---|---|
| HEAD (ff70766) | 4.28 | 1.57 | 2.83 | 13.90 | 0.176 |
| HEAD + learned colour | 3.98 | 1.40 | 2.68 | 12.43 | 0.032 |
| HEAD + learned colour, blend_strength 0.5 | 3.28 | 1.15 | 1.53 | 8.35 | 0.022 |

(Further rows, the per-component ablation of the sprint changes and the tuned settings are in
`/tmp/claude-0/coord/face_bench.md` during the sprint and summarised below when final.)
