"""Similarity-only face placement with a protected, photometrically unchanged face.

The only resampling is one similarity warp. All colour adaptation, contour
removal and the gradient-domain solve take place OUTSIDE that protected face.
"""
from __future__ import annotations
from dataclasses import asdict

import cv2
import numpy as np

from ..errors import IncompatibleInputsError, ProcessingError
from ..imaging import blend, color
from ..imaging.filters import bbox_of, dilate, erode, gaussian
from ..imaging.inpaint import inpaint
from .geometry import apply_affine, poly_mask, umeyama
from .landmarks import topology
from .masking import _skin_probability


def swap_pure(src_img, src_face, dst_img, dst_face, settings, progress=None):
    from .swap import analyze_pair, _seam_score
    prog = progress or (lambda *_: None)
    report = {"settings": asdict(settings), "mode": "pure", "analysis": analyze_pair(src_img, src_face, dst_img, dst_face)}
    if not report["analysis"]["compatible"] and not settings.force:
        raise IncompatibleInputsError("These faces need similar head orientations for Pure swap.", hint=" ".join(report["analysis"]["issues"]))
    prog(0.1, "Aligning without changing face shape")
    t = topology()
    M = umeyama(src_face.xy[t["anchors"]], dst_face.xy[t["anchors"]])
    if not np.isfinite(M).all():
        raise ProcessingError("The face could not be aligned.")
    transformed = apply_affine(M, src_face.xy)
    fs = dst_face.size
    H, W = dst_img.shape[:2]
    oval_full = poly_mask((H, W), transformed[t["oval"]])
    roi = bbox_of(np.maximum(oval_full, poly_mask((H, W), dst_face.xy[t["oval"]])), pad=int(fs*0.65)+8)
    y0, y1, x0, x1 = roi
    h, w = y1-y0, x1-x0
    mr = M.copy(); mr[:, 2] -= (x0, y0)
    aligned = cv2.warpAffine(src_img.astype(np.float32), mr, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    aligned = np.clip(aligned, 0, 1)
    support = cv2.warpAffine(np.ones(src_img.shape[:2], np.uint8), mr, (w, h), flags=cv2.INTER_NEAREST)
    protected = (oval_full[y0:y1, x0:x1] > 0.5) & (support > 0)
    if protected.sum() < 20:
        raise ProcessingError("The aligned face is outside the target photo.")
    target = dst_img[y0:y1, x0:x1].astype(np.float32)
    seed = erode(protected.astype(np.float32), max(1, int(fs*0.12)))
    skin_t = _skin_probability(target, seed)
    old = poly_mask((h, w), dst_face.xy[t["oval"]], (x0, y0)) > 0.5
    # Only exposed portions of the OLD contour are removed. Never inpaint the
    # protected source, hair outside the old outline, or unrelated background.
    residual = old & ~protected
    residual &= dilate(protected.astype(np.uint8), max(2, int(fs*0.1))) > 0
    prog(0.3, "Removing exposed target contours")
    surroundings = target.copy()
    if residual.sum() > 8:
        fillmask = dilate(residual.astype(np.uint8), 2).astype(bool) & ~protected
        # Keep the source visible as context to the inpainting fallback/model.
        context = target.copy(); context[protected] = aligned[protected]
        filled, method = inpaint(context, fillmask.astype(np.float32), method="auto", max_extent=256)
        report["inpaint"] = method
        surroundings[fillmask] = filled[fillmask]
    prog(0.5, "Adapting surrounding skin to the unchanged face")
    skin_s = _skin_probability(aligned, seed)
    stats = seed * skin_s * skin_t
    if stats.sum() < 20:
        stats = seed
    lab_s, lab_t = color.rgb_to_lab(aligned), color.rgb_to_lab(target)
    sel = stats > 0.3
    delta = np.median(lab_s[sel], axis=0)-np.median(lab_t[sel], axis=0) if sel.any() else np.zeros(3)
    # Smoothly affect exposed neck/ears, never clothes or the background.
    dist = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, 5)
    falloff = np.clip(1-dist/max(8, fs*0.65), 0, 1)**2
    skin = skin_t
    try:
        from . import parsing
        pt = parsing.parse_region(dst_img, dst_face.points, roi) if parsing.available() else None
        if pt is not None:
            skin = pt.group("skin", "nose", "neck", "ears")
    except (ValueError, RuntimeError):
        pass
    weight = gaussian((skin*falloff).astype(np.float32), max(1, fs*0.015))
    shifted = color.lab_to_rgb(color.rgb_to_lab(surroundings) + weight[..., None]*delta)
    # A Dirichlet ring: both inner (source face) and outer (target) boundaries
    # are fixed. No solve, pyramid or camera matching can enter the source face.
    band = (~protected) & (dist < max(4, fs*0.28)) & (skin > 0.05)
    base = shifted.copy(); base[protected] = aligned[protected]
    prog(0.75, "Blending only outside the face")
    solved = blend.poisson_blend(surroundings, base, band.astype(np.float32), robust=False)
    out = shifted.copy()
    out[band] = solved[band]
    out[falloff <= 0] = target[falloff <= 0]
    out[protected] = aligned[protected]  # exact invariant, after every correction
    result = dst_img.copy(); result[y0:y1, x0:x1] = out
    mask = np.zeros((H, W), np.float32); mask[y0:y1, x0:x1] = protected
    report.update(mask={"target_mouth_kept": False, "coverage_px": int(protected.sum())},
        pure={"alignment": "similarity", "protected_pixels": int(protected.sum()), "max_face_change": float(np.max(np.abs(out[protected]-aligned[protected]))), "contour_pixels_removed": int(residual.sum())},
        quality={"seam_visibility": round(_seam_score(out, gaussian(protected.astype(np.float32), 1)), 3),
                 "skin_delta_e_before": float(np.linalg.norm(delta)), "skin_delta_e_after": float(np.linalg.norm(delta)), "transition_delta_e": float(color.delta_e(color.rgb_to_lab(out), color.rgb_to_lab(target))[band].mean()) if band.any() else 0.0})
    prog(1, "Done")
    return result, report, {"mask": mask, "roi": roi, "similarity": M}
