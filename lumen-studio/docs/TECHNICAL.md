# Technical notes

This document explains the algorithms Lumen Studio uses, why they were chosen, and where they stop working.

## 1. Shared image-processing library (`engine/lumen_engine/imaging`)

| Module | Techniques |
|---|---|
| `color.py` | sRGB↔linear, CIELAB, weighted statistics, **Reinhard colour transfer** (statistics from weighted regions only, std-ratio clamped), **weighted histogram specification** (smoothed transfer curves), **Shades-of-Grey illuminant estimation**, **von Kries adaptation** |
| `blend.py` | **Poisson image editing**: membrane formulation solved with algebraic multigrid (pyamg), with a robust boundary (outlying boundary differences are replaced by a normalised-convolution trend, which prevents colour bleeding from hair or background). Also **mixed-gradient** cloning and **Laplacian-pyramid multiband blending**. Every operation runs only inside the mask's bounding box, and pixels outside it are returned unchanged. |
| `matting.py` | Trimaps, **GrabCut edge snapping** inside an uncertainty band (keeping only components that touch the user's core selection), **colour-guided-filter alpha mattes**, **foreground colour estimation** (I = αF + (1−α)B, with B propagated under the edge) |
| `analysis.py` | **Immerkær noise estimation** on flat areas, contrast-invariant **sharpness** metric (fine vs. coarse gradient energy), **bisection-based blur/unsharp matching**, **camera-like grain synthesis** (correlated luma + weaker chroma, calibrated with the same estimator), luminance **plane fitting** for lighting gradients |
| `inpaint.py` | **Multi-scale PatchMatch completion** (Wexler et al. EM with Barnes et al. NNF search), compiled with numba. It adds onion-peel initialisation at the coarsest level (Newson et al.), gradient **texture features** in the patch distance (stops the classic blurry "mush"), parallel Jacobi-style PatchMatch at fine levels, sharp weighted voting, **joint completion of extra channels** (depth) and **restricted source regions** (copy only from the background). Large holes are completed at a bounded size and get matched grain added back. The optional **LaMa** ONNX model is used when installed. |

## 2. Face replacement (`face/`)

1. **Landmarks:** MediaPipe Face Landmarker gives 478 3D points and a facial transformation matrix, from which yaw, pitch and roll are derived. Detection runs on the full frame and on 2×2, 3×3 and 4×4 overlapping tiles to find small faces in group photos (24 of about 25 faces in the test image). Each face is then **refined** on a tight crop. The manual "mark face" mode retries several crop margins and in-plane rotations.
2. **Compatibility analysis** is computed before any pixel is changed:
   - yaw and pitch difference;
   - **self-occlusion**, measured as the area of mesh triangles whose orientation flips between the two faces;
   - resolution ratio;
   - **lighting direction and strength**, from a plane fitted to skin-only log-luminance in face-aligned coordinates;
   - **skin visibility**, an occlusion estimate from a cheek colour model.

   Hard limits (yaw > 38°, pitch > 32°, folded mesh > 6%, hard side light into flat light) refuse the operation with an explanation. The user can override this explicitly.
3. **Geometry:** a dense piecewise-affine warp over MediaPipe's own tessellation (854 triangles). One per-pixel inverse map is computed (barycentric, with nearest-triangle extrapolation outside the mesh for the feather band) and a single bicubic `remap` is applied, so there are no seams between triangles. The *preserve source proportions* option blends interior landmarks towards the similarity-aligned source (Umeyama); the contour always follows the target.
4. **Mask:** the target face oval is intersected with the warped source support. Skin likelihood is then applied:
   - in the outer band of the face, and across the whole forehead (to exclude bangs, hats, glasses rims and hands), using a robust Mahalanobis model in Lab;
   - the target's open mouth is optionally kept (teeth);
   - the mask is eroded inward, then **guided-filter feathering** uses the target as the guide image.
5. **Illumination transfer:** shading layers are estimated in log-luminance. Facial features are filled from surrounding skin first, then an edge-preserving guided filter keeps hard shadow terminators. The target's layer replaces the source's in CIE Y (luminance only), and a local low-frequency chroma correction follows.
6. **Colour:**
   - skin-weighted Reinhard transfer, plus partial histogram matching;
   - eyes, lips and brows keep more of their own colour.
7. **Camera characteristics:** sharpness is matched by bisection on blur sigma or unsharp amount, and noise by calibrated grain synthesis or gentle NL-means denoising.
8. **Blending:** a robust Poisson membrane is solved inside the mask and extended into the feather band, followed by multiband compositing.
9. **Quality report:**
   - seam visibility (gradient energy on the mask boundary relative to its neighbourhood);
   - skin ΔE before and after;
   - transition ΔE.

## 3. Region exchange (`region/`)

For each direction:
1. **Selection refinement:** GrabCut snapping, a guided-filter alpha matte, then foreground colour estimation.
2. **Geometry:** the source is fitted to the destination selection (contain, cover, stretch or equal-area; centre or ground anchor), with user scale, rotation, offset and flip, or a full **homography** from two quads (API). The source is anti-aliased before shrinking.
3. **Plate:** where the new content won't cover the destination's original content, that content is removed by inpainting.
4. **Environment analysis:** rings around both selections give:
   - illuminant colour (60% local, 40% global);
   - log-exposure and contrast;
   - lighting gradient (a plane fitted to smoothed log-luminance);
   - noise and sharpness.

   The inserted region is corrected by the *difference between the two environments*, so object colours keep their identity (a red cup stays red on green grass) while white balance, exposure and directional light adapt. Exposure is only partially adapted, because brightness differences between surroundings are partly scene content.
5. **Compositing:**
   - *Object mode:* matte compositing with **light wrap** and an optional **contact shadow** (offset away from the estimated light).
   - *Area mode:* robust Poisson, then multiband with the destination-edge-aware alpha.
   - Alternatively, Poisson, mixed-gradient or multiband blending chosen by the user.

   After compositing, a final mask guarantees that only the edited area and its transition band differ from the original.

## 4. 3D reconstruction (`depth/`, `recon/`, `frontend/src/three`)

- **Depth:** Depth Anything V2 Small (ONNX, 518²) with horizontal-flip test-time augmentation, then multi-crop detail fusion:
  - overlapping high-resolution crops are aligned to the global estimate by least-squares scale and shift;
  - their high frequencies are merged with the global low frequencies.

  Edges are snapped to the image with a guided filter followed by an OpenCV joint weighted-median filter. **Sky** (far, smooth, bright, connected to the top border) is pushed to a far dome.
- **Discontinuities:** neighbours whose depth ratio exceeds a threshold, also checked over 2 px to catch mixed edge pixels, tear the mesh. This avoids rubber-sheet triangles.
- **Occlusion band:**
  - Every pixel is linked to its nearest near-side edge pixel. It belongs to the band if it lies on that near surface and is within a width that grows with the disparity jump (parallax).
  - The band's hidden depth is the far side of that edge, smoothed.
  - Its colour is completed by **PatchMatch on [RGB, 2·disparity]**, with sources restricted to outside the band, so mostly background texture is copied.
- **Outpainting:** the frame is extended by about 6% per side (PatchMatch at half resolution). The viewer fades this periphery gently to show that it is low-confidence.
- **Container (`LSC1`):** per-vertex disparity grids and bit-packed quad masks for the visible layer and the synthesised background layer, plus textures. The vertex shader unprojects (u, v, d) with the current field of view and depth range, so those parameters can be changed live. GLB and PLY exports bake the current parameters.
- **Viewer:**
  - orbit around the median-depth pivot, or walk mode (WASD/QE, mouse-look);
  - wheel dolly and right-drag pan;
  - **safe range constraint** (the camera stays where the single-view estimate is trustworthy), with a warning at the edge;
  - motion presets, photo/depth/layer shading, point-cloud mode.

**Why not Gaussian splatting or NeRF?** Both optimise a scene representation from many calibrated views. From one photo, they need generative multi-view diffusion models (multi-GB, GPU-only, often non-commercial licences) to imagine the missing views. That conflicts with the offline, CPU-capable and licence requirements. A layered depth mesh with explicit occlusion completion is the practical alternative, and it is honest about what was estimated.

## 5. Verified behaviour

Test material: 18 face photos (varied skin tones, lighting, resolutions from 320 px to 2204 px, group photos, occlusions) and 23 scene photos (street, interiors, staircases, landscapes, paintings, macro objects, animals, glassware, buildings). Inspected results included:
- frontal ↔ frontal swaps across skin tones;
- swaps into side-lit targets (relighting works well);
- small and blurry sources (softer, with a warning);
- strongly side-lit sources (refused);
- hat and bang occlusions (bangs detected; a skin-coloured hat brim is not);
- object, area and tiny region exchanges;
- depth and occlusion layers for indoor and outdoor scenes.
