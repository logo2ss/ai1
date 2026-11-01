"""Intelligent face replacement.

Pipeline (all corrections are computed in the *target* image's coordinate
frame and restricted to the face region and its transition band):

1. **Landmarks & pose** - 478-point meshes for both faces; head pose from
   the facial transformation matrix.
2. **Compatibility analysis** - pose difference, self-occlusion (folded mesh
   triangles), resolution ratio and lighting-direction difference. Clearly
   incompatible pairs are refused with an explanation instead of producing a
   misleading result (the user may override with ``force``).
3. **Geometric alignment** - dense piecewise-affine warp over the face
   mesh (optionally preserving source proportions), one high-quality remap.
4. **Segmentation mask** - face-oval mesh mask ∩ warped source support,
   refined by skin-likelihood in the outer band (stops at hair, glasses
   rims, hands), optional exclusion of the target's open mouth (keeps its
   teeth), edge-aware guided feathering.
5. **Colour** - skin-weighted Reinhard transfer in CIELAB + partial
   weighted histogram matching.
6. **Local illumination** - large-scale log-luminance "shading" layers of
   both faces (facial features removed before smoothing) are swapped, so
   lighting direction, shadows and highlights follow the target scene.
7. **Camera characteristics** - sharpness matched by bisection on a blur /
   unsharp amount; sensor noise matched by camera-like grain synthesis or
   gentle denoising.
8. **Gradient-domain blending** - Poisson membrane with robust boundary,
   then multi-band (Laplacian pyramid) compositing with the soft mask.
9. **Quality report** - seam visibility and colour difference at the boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..errors import IncompatibleInputsError, ProcessingError
from ..imaging import analysis, blend, color
from ..imaging.filters import bbox_of, erode, gaussian, guided_filter, normalized_blur
from .geometry import (
    dense_warp_maps,
    destination_points,
    feature_mask,
    folded_triangle_fraction,
    poly_mask,
)
from . import appearance as ap
from .landmarks import Face, topology
from .settings import FaceSwapSettings, tv  # noqa: F401,E402
from .masking import _face_masks, _skin_probability, _skin_visibility  # noqa: F401,E402


def _shading_direction(img: np.ndarray, face: Face) -> tuple[np.ndarray, float]:
    """Dominant lighting direction across the face (2D, face-aligned) and strength.

    Fits a plane to the smoothed log-luminance of skin pixels in a face-aligned
    frame; the gradient of the plane points towards the light.
    """
    t = topology()
    x0, y0, x1, y1 = face.bbox
    pad = 4
    roi = (int(max(0, y0 - pad)), int(min(img.shape[0], y1 + pad)), int(max(0, x0 - pad)), int(min(img.shape[1], x1 + pad)))
    sub = img[roi[0] : roi[1], roi[2] : roi[3]]
    off = (roi[2], roi[0])
    shape = sub.shape[:2]
    skin = poly_mask(shape, face.xy[t["oval"]], off) * (1 - feature_mask(shape, face.xy, off, face.size * 0.04))
    skin = erode(skin, max(1, int(face.size * 0.05)))
    if skin.sum() < 50:
        return np.zeros(2), 0.0
    L = np.log(color.luminance(sub) + 0.02)
    L = normalized_blur(L, skin, face.size * 0.06)
    # face-aligned axes from eye line
    ex = face.xy[[362, 263]].mean(0) - face.xy[[33, 133]].mean(0)
    ex /= np.linalg.norm(ex) + 1e-9
    ey = np.array([-ex[1], ex[0]])
    ys, xs = np.nonzero(skin > 0.5)
    P = np.stack([xs + off[0], ys + off[1]], 1) - face.xy[1]
    u, v = P @ ex / face.size, P @ ey / face.size
    A = np.stack([u, v, np.ones_like(u)], 1)
    coef, *_ = np.linalg.lstsq(A, L[ys, xs], rcond=None)
    g = coef[:2]
    return g / (np.linalg.norm(g) + 1e-9), float(np.linalg.norm(g))

def analyze_pair(src_img: np.ndarray, src: Face, dst_img: np.ndarray, dst: Face) -> dict:
    """Assess whether a realistic replacement is possible; returns a report."""
    issues, warnings = [], []
    dyaw, dpitch = abs(src.yaw - dst.yaw), abs(src.pitch - dst.pitch)
    folded = folded_triangle_fraction(src.xy[:468], dst.xy[:468])
    res_ratio = dst.size / max(src.size, 1e-6)
    ls, ss = _shading_direction(src_img, src)
    lt, st = _shading_direction(dst_img, dst)
    light_angle = float(np.degrees(np.arccos(np.clip(ls @ lt, -1, 1)))) if ss > 0.15 and st > 0.15 else 0.0

    if dyaw > 38 or dpitch > 32:
        issues.append(
            f"Head orientations differ strongly (yaw {src.yaw:+.0f}° vs {dst.yaw:+.0f}°, pitch {src.pitch:+.0f}° vs {dst.pitch:+.0f}°). "
            "Parts of the face needed for the target view are not visible in the source photo."
        )
    elif dyaw > 20 or dpitch > 18:
        warnings.append("Head orientations differ noticeably; the face will be re-projected and may look slightly flattened.")
    if folded > 0.06:
        issues.append(f"{folded * 100:.0f}% of the face surface is self-occluded in one image but visible in the other.")
    elif folded > 0.02:
        warnings.append("A small part of the face is hidden in one of the photos; that area will be stretched.")
    if res_ratio > 3.0:
        warnings.append(
            f"The source face is {res_ratio:.1f}x smaller than the target face; the replacement will be softer than its surroundings "
            "(sharpness is matched by softening the surroundings' detail level, not by inventing detail)."
        )
    if light_angle > 110 and min(ss, st) > 0.35:
        warnings.append(
            "The faces are lit from clearly different directions. Shading is re-lit towards the target, "
            "but strong cast shadows from the source cannot be fully removed."
        )
    elif ss > 1.2 and ss - st > 0.9:
        issues.append(
            "The source face is lit by a hard side light while the target face is evenly lit. Removing deep shadows "
            "from a photo cannot be done reliably - the result would show patchy, discoloured skin."
        )
    elif abs(ss - st) > 0.6:
        which = "source" if ss > st else "target"
        warnings.append(
            f"The {which} face has much stronger directional lighting than the other. Shading is transferred, "
            "but deep shadows and specular highlights may not match perfectly; consider lowering 'Lighting match' if the face looks flat or patchy."
        )
    vis_s, vis_t = _skin_visibility(src_img, src), _skin_visibility(dst_img, dst)
    if vis_s < 0.7:
        warnings.append(
            f"About {100 - vis_s * 100:.0f}% of the source face appears covered (hair, hat, glasses, hands). "
            "Covered areas would be transferred too; choose an unobstructed source photo for best results."
        )
    if vis_t < 0.7:
        warnings.append("Part of the target face is covered; the mask stops at non-skin edges where it can detect them.")
    if src.eye_openness() < 0.002 <= dst.eye_openness() * 0.5:
        warnings.append("The source face has closed eyes.")
    score = 1.0
    score -= min(0.5, max(0, dyaw - 10) / 60) + min(0.3, max(0, dpitch - 10) / 60)
    score -= min(0.3, folded * 4)
    score -= 0.15 if res_ratio > 3 else 0
    score -= 0.15 if light_angle > 110 and min(ss, st) > 0.35 else 0
    score -= 0.1 if abs(ss - st) > 0.6 else 0
    score -= max(0.0, 0.7 - vis_s)
    return {
        "compatible": not issues,
        "score": round(max(0.0, score), 2),
        "issues": issues,
        "warnings": warnings,
        "yaw_difference": round(dyaw, 1),
        "pitch_difference": round(dpitch, 1),
        "self_occlusion": round(folded, 3),
        "resolution_ratio": round(res_ratio, 2),
        "light_direction_difference": round(light_angle, 1),
        "source_light_strength": round(ss, 2),
        "target_light_strength": round(st, 2),
        "source_visibility": round(vis_s, 2),
        "target_visibility": round(vis_t, 2),
    }

def _relight(warped: np.ndarray, target: np.ndarray, skin: np.ndarray, feat: np.ndarray, fs: float, strength: float) -> tuple[np.ndarray, float]:
    """Swap the large-scale shading layer of the source for that of the target.

    In log-luminance, log Y = log(albedo/detail) + log(shading). The shading
    layer is estimated by normalised (skin-only) smoothing, so the darkness of
    eyes, brows, lips and nostrils is not mistaken for shadow. The change is
    applied to luminance only (CIE Y -> L*), keeping a*/b* so brightened
    shadows do not become over-saturated. Returns (image, mean |change|).
    """
    if strength <= 0:
        return warped, 0.0
    eps = 0.03
    sigma = fs * 0.09
    lab = color.rgb_to_lab(warped)
    Ys = color.luminance(color.srgb_to_linear(warped))
    Yt = color.luminance(color.srgb_to_linear(target))
    w = (skin > 0.5).astype(np.float32)
    if w.sum() < 100:
        return warped, 0.0
    def shading(logY: np.ndarray) -> np.ndarray:
        # fill non-skin (features, hair) from surrounding skin, then an
        # edge-preserving smoothing keeps hard shadow terminators / cast
        # shadows while removing skin texture.
        filled = np.where(w > 0, logY, normalized_blur(logY, w, fs * 0.05))
        base = guided_filter(filled.astype(np.float32), filled.astype(np.float32), max(2, int(fs * 0.07)), 0.03)
        return normalized_blur(base, np.ones_like(w), max(1.0, fs * 0.015))

    bs = shading(np.log(Ys + eps))
    bt = shading(np.log(Yt + eps))
    delta = bt - bs
    delta -= (delta * w).sum() / w.sum()
    delta = np.clip(delta * strength, -1.0, 1.0)
    Ynew = np.clip((Ys + eps) * np.exp(delta) - eps, 0, 1)
    # Y -> L* (CIE)
    f = np.where(Ynew > 0.008856, np.cbrt(Ynew), 7.787 * Ynew + 16 / 116)
    L_new = 116 * f - 16
    ratio = np.clip((L_new + 1) / (lab[..., 0] + 1), 0.5, 2.0)
    lab[..., 0] = L_new
    # chroma follows lightness gently (keeps shadows from turning grey or neon)
    lab[..., 1:] *= np.sqrt(ratio)[..., None]
    # local chroma correction: coloured light / bounce light in the source
    # (e.g. warm shadows) is replaced by the target's low-frequency chroma
    # variation (mean difference is left to the global colour transfer)
    lab_t = color.rgb_to_lab(target)
    ab_s = normalized_blur(lab[..., 1:], w, fs * 0.08)
    ab_t = normalized_blur(lab_t[..., 1:], w, fs * 0.08)
    dab = ab_t - ab_s
    dab -= (dab * w[..., None]).reshape(-1, 2).sum(0) / w.sum()
    lab[..., 1:] += np.clip(dab * strength, -12, 12)
    return color.lab_to_rgb(lab), float((np.abs(delta) * w).sum() / w.sum())

# "sh": physically based relighting (face/appearance.py); "legacy": shading-layer swap + Reinhard
APPEARANCE_MODE = "sh"


def _illuminant(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, 400.0 / max(h, w))
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    return color.estimate_illuminant(small)


def _face_geometry(src_face, dst_face, dst_pts, shape, off):
    """Mesh raster of the target ROI and per-pixel normals of the source / target surfaces."""
    raster = ap.MeshRaster(dst_pts, shape, off)
    nv_s, nv_t = ap.vertex_normals(src_face.xy), ap.vertex_normals(dst_face.xy)
    if nv_s is None or nv_t is None:
        raise ValueError("face template unavailable")
    n_src, n_dst = raster.interp(nv_s), raster.interp(nv_t)
    n_src /= np.linalg.norm(n_src, axis=-1, keepdims=True) + 1e-6
    n_dst /= np.linalg.norm(n_dst, axis=-1, keepdims=True) + 1e-6
    return {"raster": raster, "nv_s": nv_s, "nv_t": nv_t, "n_src": n_src, "n_dst": n_dst}


def _symmetric_completion(src_img, src_face, dst_pts, warped, geo, fs):
    """Re-sample parts of the face that the source shows folded / foreshortened / turned away
    from the mirrored (well visible) half of the source face. Normals follow, so the
    relighting removes the correct source shading from the mirrored texture."""
    raster, nv_s = geo["raster"], geo["nv_s"]
    mir = ap.canonical()["mirror"]
    src_xy = src_face.xy[:468].astype(np.float64)
    q = ap.warp_quality(raster, src_xy, dst_pts, nv_s)
    q_m = ap.warp_quality(raster, src_xy[mir], dst_pts, nv_s[mir])
    w = np.clip((q_m - q) * 2.0, 0, 1) * (1 - q) * raster.inside
    w = np.clip(gaussian(w.astype(np.float32), max(1.0, fs * 0.015)) * tv("mirror_fill", 1.0), 0, 1)
    frac = float((w > 0.5).sum() / max(int(raster.inside.sum()), 1))
    rep = {"mirrored_fraction": round(frac, 3), "poorly_sampled_fraction": round(float(((q < 0.5) & raster.inside).sum() / max(int(raster.inside.sum()), 1)), 3)}
    if w.max() < 0.05:
        return warped, rep
    mxy = raster.interp(src_xy[mir].astype(np.float32))
    wm = cv2.remap(src_img.astype(np.float32), mxy[..., 0], mxy[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    out = warped * (1 - w[..., None]) + np.clip(wm, 0, 1) * w[..., None]
    n_m = raster.interp(nv_s[mir])
    n = geo["n_src"] * (1 - w[..., None]) + n_m * w[..., None]
    geo["n_src"] = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)
    return out.astype(np.float32), rep


def _appearance_sh(src_img, dst_img, warped, target, dst_pts, src_support, feat, roi, fs, s, geo):
    """Mesh normals + SH relighting + colour transfer (see face/appearance.py)."""
    if geo is None:
        raise ValueError("no face geometry")
    ry0, ry1, rx0, rx1 = roi
    shape = (ry1 - ry0, rx1 - rx0)
    off = (rx0, ry0)
    hull = poly_mask(shape, dst_pts[topology()["oval"]], off) * (src_support > 0.5)
    region = erode(hull, max(1, int(fs * tv("light_region_erode", 0.03)))) * (1 - feat)
    parts = ap.part_masks(shape, dst_pts, off, fs)
    out, rep = ap.appearance_transfer(
        warped, target, geo["n_src"], geo["n_dst"], region, parts, fs, s.lighting_match, s.color_match,
        src_illum=_illuminant(src_img), dst_illum=_illuminant(dst_img), debug=_DEBUG,
    )
    if _DEBUG is not None:
        _DEBUG.update(n_src=geo["n_src"], n_dst=geo["n_dst"], region=region, warped=warped, target=target)
    return out, rep


_DEBUG: dict | None = None  # set to a dict to collect intermediate images (development)


def swap_face(
    src_img: np.ndarray,
    src_face: Face,
    dst_img: np.ndarray,
    dst_face: Face,
    settings: FaceSwapSettings | None = None,
    progress=None,
) -> tuple[np.ndarray, dict, dict]:
    """Replace ``dst_face`` in ``dst_img`` with ``src_face`` from ``src_img``.

    Returns (result_image, report, debug_images).
    """
    s = settings or FaceSwapSettings()
    if s.mode == "pure":
        from .pure import swap_pure
        return swap_pure(src_img, src_face, dst_img, dst_face, s, progress)
    prog = progress or (lambda *_: None)
    report = {"settings": asdict(s)}
    analysis_report = analyze_pair(src_img, src_face, dst_img, dst_face)
    report["analysis"] = analysis_report
    if not analysis_report["compatible"] and not s.force:
        raise IncompatibleInputsError(
            "These two faces cannot be combined realistically.",
            hint=" ".join(analysis_report["issues"])
            + " Choose a source photo with a similar head orientation, or enable 'Proceed anyway' to see the (limited) result.",
            analysis=analysis_report,
        )
    prog(0.1, "Aligning faces")
    H, W = dst_img.shape[:2]
    fs = dst_face.size
    dst_pts = destination_points(src_face, dst_face, s.preserve_source_shape)
    x0, y0, x1, y1 = dst_face.bbox
    pad = int(fs * 0.35) + 8
    roi = (max(0, int(y0) - pad), min(H, int(y1) + pad + 1), max(0, int(x0) - pad), min(W, int(x1) + pad + 1))
    ry0, ry1, rx0, rx1 = roi
    target = dst_img[ry0:ry1, rx0:rx1].astype(np.float32)
    try:
        map_x, map_y, inside = dense_warp_maps(src_face.xy[:468], dst_pts, (H, W), roi)
    except ValueError as exc:
        raise ProcessingError("The face mesh could not be warped (degenerate landmarks).") from exc
    # downscaling? pre-filter source to avoid aliasing when the source face is larger
    scale = fs / max(src_face.size, 1e-6)
    src_for_warp = src_img
    if scale < 0.8:
        src_for_warp = gaussian(src_img, 0.45 / scale)
    warped = cv2.remap(src_for_warp.astype(np.float32), map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    warped = np.clip(warped, 0, 1)
    src_oval = poly_mask(src_img.shape[:2], src_face.xy[topology()["oval"]])
    src_support = cv2.remap(src_oval, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    src_support *= ((map_x >= 0) & (map_y >= 0) & (map_x < src_img.shape[1] - 1) & (map_y < src_img.shape[0] - 1)).astype(np.float32)

    # pose / self-occlusion: badly sampled parts come from the mirrored half of the source face
    geo = None
    try:
        geo = _face_geometry(src_face, dst_face, dst_pts, target.shape[:2], (rx0, ry0))
        if tv("mirror_fill", 1.0) > 0 and abs(src_face.yaw - dst_face.yaw) > 12 and ap.canonical() is not None:
            warped, report["pose_completion"] = _symmetric_completion(src_for_warp, src_face, dst_pts, warped, geo, fs)
    except ValueError:
        geo = None

    prog(0.25, "Building face mask")
    try:
        soft, skin, feat, mouth_kept, skin_both = _face_masks(
            target, warped, dst_face, dst_pts, src_support, roi, s, src_face, src_img=src_img, warp_maps=(map_x, map_y)
        )
    except TypeError:  # older masking module without the keyword arguments
        soft, skin, feat, mouth_kept, skin_both = _face_masks(target, warped, dst_face, dst_pts, src_support, roi, s, src_face)
    if soft.max() < 0.5:
        raise ProcessingError("The face region is empty after masking.")
    report["mask"] = {"target_mouth_kept": mouth_kept, "coverage_px": int((soft > 0.5).sum())}

    skin_w = skin * (soft > 0.9)
    lab_before = color.weighted_stats(color.rgb_to_lab(warped), skin_w)[0]
    lab_target = color.weighted_stats(color.rgb_to_lab(target), skin_w)[0]

    prog(0.4, "Transferring illumination")
    warped_c = None
    if APPEARANCE_MODE == "sh" and fs >= tv("small_face_px", 56):  # tiny faces: the legacy transfer is more robust
        try:
            warped_c, app_rep = _appearance_sh(src_img, dst_img, warped, target, dst_pts, src_support, feat, roi, fs, s, geo)
            report["lighting"] = app_rep
        except (ValueError, np.linalg.LinAlgError) as exc:  # too little skin, degenerate mesh...
            report["lighting"] = {"method": "legacy", "reason": str(exc)}
    if warped_c is None:
        warped_c, relight_amount = _relight(warped, target, skin_w, feat, fs, s.lighting_match)
        report.setdefault("lighting", {})["mean_log_change"] = round(relight_amount, 3)
        prog(0.5, "Matching skin colour")
        if s.color_match > 0:
            cc = color.reinhard_transfer(warped_c, skin_w, target, skin_w, strength=s.color_match, std_clamp=(0.75, 1.35))
            cc = color.histogram_match(cc, skin_w, target, skin_w, strength=0.35 * s.color_match)
            # eyes/brows/lips keep more of their own colour (skin-tone shift is person-specific)
            keep = gaussian(feat, max(1.0, fs * 0.01))[..., None] * 0.5
            warped_c = cc * (1 - keep) + (warped_c + (cc - warped_c) * 0.5) * keep

    prog(0.6, "Matching camera characteristics")
    sharp_report = noise_report = None
    if s.match_sharpness:
        warped_c, sharp_report = analysis.match_sharpness(warped_c, target, skin_w)
    # skin micro-texture: match the energy of fine detail (pores, fine wrinkles) so an airbrushed or
    # upscaled face does not look pasted into a textured photo. Missing detail is borrowed from the
    # target's own skin at the finest scale (identity-neutral: pores, sensor noise, JPEG texture);
    # excess detail is attenuated.
    tex_sigma = max(0.8, fs * tv("texture_sigma", 0.012))
    det_s = warped_c - gaussian(warped_c, tex_sigma)
    det_t = target - gaussian(target, tex_sigma)
    sw_ = skin_w > 0.5
    if sw_.sum() > 200:
        e_s, e_t = float(np.sqrt((det_s[sw_] ** 2).mean())), float(np.sqrt((det_t[sw_] ** 2).mean()))
        gain = float(np.clip(e_t / max(e_s, 1e-5), tv("texture_gain_lo", 0.7), tv("texture_gain_hi", 1.5)))
        if gain < 0.95:
            warped_c = np.clip(warped_c + det_s * (gain - 1) * 0.8, 0, 1)
        elif e_t > e_s * 1.05:
            # own detail up to +25 %, the remaining energy from the target's skin texture
            own = min(gain, 1.25)
            warped_c = warped_c + det_s * (own - 1)
            missing = np.sqrt(max(e_t**2 - (e_s * own) ** 2, 0.0)) / max(e_t, 1e-6)
            tw = gaussian((soft * (1 - feat) * np.clip(skin_both * 1.5, 0, 1)).astype(np.float32), max(0.8, fs * 0.01))
            warped_c = np.clip(warped_c + det_t * (missing * tv("texture_borrow", 0.9)) * tw[..., None], 0, 1)
            report["texture_borrowed"] = round(missing, 2)
        report["texture_gain"] = round(gain, 2)
    if s.match_grain:
        warped_c, noise_report = analysis.match_noise(warped_c, target, skin_w, skin_w)
    report["camera"] = {"sharpness": sharp_report, "noise": noise_report}

    prog(0.72, "Gradient-domain blending")
    if s.blend_strength > 0:
        binary = (soft > 0.5).astype(np.float32)
        # boundary samples that fall on hair, glasses or shadow (non-skin in either
        # photo) must not shift the face's tone: weight them by skin likelihood
        # The membrane is solved in CIELAB so its colour part can be limited: a boundary that touches a red
        # collar or a coloured background must not tint the face (the colour transfer already matched the skin).
        lab_w, lab_t = color.rgb_to_lab(warped_c), color.rgb_to_lab(target)
        k_lab = np.float32(0.01)
        poisson = blend.poisson_blend(lab_w * k_lab, lab_t * k_lab, binary, mode="normal", robust=True, boundary_weight=skin_both) / k_lab
        memb = (poisson - lab_w) * binary[..., None]
        memb[..., 0] = np.clip(memb[..., 0], -tv("poisson_L_clip", 22.0), tv("poisson_L_clip", 22.0))
        memb[..., 1:] = np.clip(memb[..., 1:], -tv("poisson_ab_clip", 4.0), tv("poisson_ab_clip", 4.0))
        # safety: a residual global shift from the boundary beyond ~3 dE is an artefact - remove it
        core = (skin_w > 0.5) & (binary > 0)
        if core.sum() > 100:
            off_ = memb[core].mean(0)
            mag = float(np.linalg.norm(off_))
            lim = tv("tone_clamp_de", 3.0)
            if mag > lim:
                memb[binary > 0] -= off_ * (1 - lim / mag)
                report["tone_correction_de"] = round(mag, 1)
        # Poisson only defines values inside the mask; extend the correction smoothly outward for the feather band
        corr_ext = normalized_blur(memb, binary, max(2.0, fs * 0.02))
        corr = np.where(binary[..., None] > 0, memb, corr_ext)
        fg = color.lab_to_rgb(lab_w + s.blend_strength * corr)
    else:
        fg = warped_c

    prog(0.85, "Multi-band compositing")
    # the coarse pyramid levels spread whatever lies just outside the mask (source hair, background,
    # occluders) into the edge as a haze: hand the pyramid the target there
    supp = cv2.dilate((soft > 0.002).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)[..., None]
    fg = fg * supp + target * (1 - supp)
    out_roi = blend.multiband_blend(fg, target, soft)
    result = dst_img.copy()
    result[ry0:ry1, rx0:rx1] = out_roi

    # ---- quality report
    lab_res, lab_tgt = color.rgb_to_lab(out_roi), color.rgb_to_lab(target)
    ring = ((soft > 0.05) & (soft < 0.95)).astype(np.float32)
    inner = ((soft > 0.95) & (skin > 0.5)).astype(np.float32)
    seam = _seam_score(out_roi, soft)
    report["quality"] = {
        "seam_visibility": round(seam, 3),
        "skin_delta_e_before": round(float(np.linalg.norm(lab_before - lab_target)), 2),
        "skin_delta_e_after": round(float(np.linalg.norm(color.weighted_stats(lab_res, inner)[0] - lab_target)), 2),
        "transition_delta_e": round(float((color.delta_e(lab_res, lab_tgt) * ring).sum() / max(ring.sum(), 1)), 2),
    }
    prog(1.0, "Done")
    dbg_mask = np.zeros(dst_img.shape[:2], np.float32)
    dbg_mask[ry0:ry1, rx0:rx1] = soft
    return result, report, {"mask": dbg_mask, "roi": roi}

def _seam_score(img: np.ndarray, soft: np.ndarray) -> float:
    """Ratio of gradient energy on the mask boundary vs. its neighbourhood (≈1 = invisible seam)."""
    g = color.luminance(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edge = ((soft > 0.35) & (soft < 0.65)).astype(np.float32)
    near = ((soft > 0.02) & (soft < 0.98)).astype(np.float32) - edge
    near = np.clip(near, 0, 1)
    if edge.sum() < 10 or near.sum() < 10:
        return 1.0
    return float((mag * edge).sum() / edge.sum() / ((mag * near).sum() / near.sum() + 1e-6))
