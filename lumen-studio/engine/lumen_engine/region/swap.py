"""Context-aware exchange of arbitrary regions between two images.

For each direction (A's region into B, B's region into A) the engine:

1. **Refines the selection** - GrabCut edge snapping inside an uncertainty
   band, then a colour-guided-filter alpha matte, then foreground colour
   estimation (removes the source background from semi-transparent edges).
2. **Fits the geometry** - similarity fit of the source region into the
   destination selection (contain / cover / stretch / equal-area, centre or
   ground anchor), user scale/rotation/offset/flip, or a full perspective
   homography from two user quads.
3. **Prepares the destination plate** - the destination's original content
   inside its selection is removed with exemplar inpainting where the new
   content will not cover it.
4. **Analyses both environments** - rings of pixels around each selection
   give the illuminant colour (shades-of-grey), exposure and contrast
   (log-luminance statistics), lighting gradient (plane fit of low-frequency
   log-luminance), noise level and sharpness. The inserted region is adapted
   by the *difference between the environments* - not by forcing its colours
   towards the background (a red cup stays red on green grass, but takes on
   the warm light of the new scene).
5. **Composites** - matte compositing with light-wrap and optional contact
   shadow (objects), or robust Poisson + multi-band blending (areas/textures),
   or user-selected Poisson / mixed-gradient cloning.

Everything outside the destination selection's transition band is left
bit-identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import cv2
import numpy as np

from ..errors import InvalidInputError
from ..imaging import analysis, blend, color, matting
from ..imaging.filters import bbox_of, dilate, erode, gaussian, normalized_blur
from ..imaging.inpaint import inpaint
from .refine import classify_selection, extract_object


@dataclass
class RegionSwapSettings:
    mode: str = "auto"  # auto | "object": cut-out insertion | "area": fill the destination shape (textures, sky, walls)
    fit: str = "auto"  # auto | contain | cover | stretch | area
    anchor: str = "auto"  # auto | center | bottom
    extract_object: bool = True  # object mode: cut the object out of a loose loop (graph cut + saliency)
    remove_original: bool = True
    snap_edges: bool = True
    edge_softness: float = 0.4  # 0..1
    color_harmony: float = 0.8  # adapt white balance/exposure to the destination environment
    lighting_match: float = 0.6  # adapt the illumination gradient across the region
    blend_mode: str = "auto"  # auto | matte | poisson | mixed | multiband
    match_grain: bool = True
    match_sharpness: bool = True
    light_wrap: float = 0.35
    contact_shadow: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0  # degrees
    offset_x: float = 0.0  # fraction of destination selection width
    offset_y: float = 0.0
    flip: bool = False
    src_quad: list | None = None  # optional 4 points (perspective)
    dst_quad: list | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RegionSwapSettings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                cur = getattr(s, k)
                setattr(s, k, v if cur is None or isinstance(cur, (list, dict)) else type(cur)(v))
        return s


# --------------------------------------------------------------------------- geometry
def _transform(src_alpha: np.ndarray, dst_mask: np.ndarray, s: RegionSwapSettings) -> np.ndarray:
    """3x3 matrix mapping source pixel coords to destination pixel coords."""
    if s.src_quad and s.dst_quad and len(s.src_quad) == 4 and len(s.dst_quad) == 4:
        return cv2.getPerspectiveTransform(np.float32(s.src_quad), np.float32(s.dst_quad)).astype(np.float64)
    sy0, sy1, sx0, sx1 = bbox_of(src_alpha > 0.5)
    dy0, dy1, dx0, dx1 = bbox_of(dst_mask > 0.5)
    sw, sh = max(sx1 - sx0, 1), max(sy1 - sy0, 1)
    dw, dh = max(dx1 - dx0, 1), max(dy1 - dy0, 1)
    if s.fit == "stretch":
        kx, ky = dw / sw, dh / sh
    elif s.fit == "cover":
        kx = ky = max(dw / sw, dh / sh)
    elif s.fit == "area":
        kx = ky = float(np.sqrt((dst_mask > 0.5).sum() / max((src_alpha > 0.5).sum(), 1)))
    else:
        kx = ky = min(dw / sw, dh / sh)
    kx, ky = kx * s.scale, ky * s.scale
    if s.anchor == "bottom":
        sp = np.array([(sx0 + sx1) / 2, sy1], np.float64)
        dp = np.array([(dx0 + dx1) / 2, dy1], np.float64)
    else:
        sp = np.array([(sx0 + sx1) / 2, (sy0 + sy1) / 2], np.float64)
        dp = np.array([(dx0 + dx1) / 2, (dy0 + dy1) / 2], np.float64)
    dp += np.array([s.offset_x * dw, s.offset_y * dh])
    T1 = np.array([[1, 0, -sp[0]], [0, 1, -sp[1]], [0, 0, 1]], np.float64)
    F = np.diag([-1.0 if s.flip else 1.0, 1.0, 1.0])
    S = np.diag([kx, ky, 1.0])
    a = np.radians(s.rotation)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T2 = np.array([[1, 0, dp[0]], [0, 1, dp[1]], [0, 0, 1]], np.float64)
    return T2 @ R @ S @ F @ T1


def _warp(img: np.ndarray, M: np.ndarray, shape: tuple[int, int], interp=cv2.INTER_CUBIC) -> np.ndarray:
    h, w = shape
    return cv2.warpPerspective(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _warp_foreground(fg, alpha, M, shape, scale):
    """Resample premultiplied colour so background/black cannot fringe edges."""
    premult = fg.astype(np.float32)*alpha[...,None]
    a = alpha.astype(np.float32)
    if scale < .8:
        sigma = .45/max(scale,1e-6)
        premult, a = gaussian(premult,sigma), gaussian(a,sigma)
    a = np.clip(_warp(a,M,shape),0,1)
    color = _warp(premult,M,shape)
    color = np.where(a[...,None]>1e-5, color/np.maximum(a[...,None],1e-5),0)
    return np.clip(color,0,1).astype(np.float32),a


def _linear_scale(M: np.ndarray, at: np.ndarray) -> float:
    """Local isotropic scale of a homography at point ``at``."""
    x, y = at
    J = np.zeros((2, 2))
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    for i in range(2):
        n = M[i, 0] * x + M[i, 1] * y + M[i, 2]
        J[i, 0] = (M[i, 0] * d - n * M[2, 0]) / d**2
        J[i, 1] = (M[i, 1] * d - n * M[2, 1]) / d**2
    return float(np.sqrt(abs(np.linalg.det(J))))


# --------------------------------------------------------------------------- environment analysis
def _ring(mask: np.ndarray, inner: int, width: int) -> np.ndarray:
    m = (mask > 0.5).astype(np.uint8)
    return (dilate(m, inner + width) - dilate(m, inner)).astype(np.float32)


def environment(img: np.ndarray, mask: np.ndarray) -> dict:
    """Measure the surroundings of a selection (used for both images)."""
    area = max(float((mask > 0.5).sum()), 1.0)
    rad = int(np.clip(np.sqrt(area) * 0.35, 6, 300))
    ring = _ring(mask, 2, rad)
    if ring.sum() < 30:  # selection fills the image: use everything
        ring = 1 - (mask > 0.5).astype(np.float32)
    if ring.sum() < 30:
        ring = np.ones_like(mask)
    lum = color.luminance(color.srgb_to_linear(img))
    loglum = np.log(lum + 0.01)
    mean, std = color.weighted_stats(loglum, ring)
    y0, y1, x0, x1 = bbox_of(dilate((mask > 0.5).astype(np.uint8), rad))
    sub_ll = normalized_blur(loglum[y0:y1, x0:x1], ring[y0:y1, x0:x1], max(2.0, rad * 0.3))
    plane = analysis.fit_luminance_plane(sub_ll, ring[y0:y1, x0:x1])
    # express the gradient per bounding box of the selection itself
    by0, by1, bx0, bx1 = bbox_of(mask > 0.5)
    sx = (bx1 - bx0) / max(x1 - x0, 1)
    sy = (by1 - by0) / max(y1 - y0, 1)
    illum_local = color.estimate_illuminant(img, ring)
    illum_global = color.estimate_illuminant(img, 1 - (mask > 0.5).astype(np.float32))
    illum = illum_local * 0.6 + illum_global * 0.4
    illum /= np.linalg.norm(illum) + 1e-9
    return {
        "ring": ring,
        "log_mean": float(mean[0]),
        "log_std": float(std[0]),
        "gradient": np.array([plane[0] * sx, plane[1] * sy], np.float32),
        "illuminant": illum,
        "noise": analysis.noise_sigma(img, ring),
        "black": float(color.weighted_quantile(lum[ring > 0.5].ravel(), np.ones(int((ring > 0.5).sum())), np.array([0.02]))[0]) if (ring > 0.5).sum() > 30 else 0.0,
        "sharpness": analysis.sharpness(img, ring),
    }


def _env_report(e: dict) -> dict:
    return {
        "exposure_log": round(e["log_mean"], 3),
        "contrast_log": round(e["log_std"], 3),
        "light_gradient": [round(float(v), 3) for v in e["gradient"]],
        "illuminant_rgb": [round(float(v), 3) for v in e["illuminant"]],
        "noise_8bit": round(e["noise"] * 255, 2),
        "sharpness": round(e["sharpness"], 3),
    }


# --------------------------------------------------------------------------- main
def insert_region(
    src_img: np.ndarray,
    src_mask: np.ndarray,
    dst_img: np.ndarray,
    dst_mask: np.ndarray,
    settings: RegionSwapSettings | None = None,
    progress=None,
) -> tuple[np.ndarray, dict, dict]:
    """Insert the selected region of ``src_img`` into the selection of ``dst_img``."""
    s = replace(settings) if settings is not None else RegionSwapSettings()  # never mutate the caller's settings
    prog = progress or (lambda *_: None)
    if (src_mask > 0.5).sum() < 16 or (dst_mask > 0.5).sum() < 16:
        raise InvalidInputError("Both selections must contain at least a few pixels.", hint="Draw a larger selection.")
    H, W = dst_img.shape[:2]
    report: dict = {"settings": {k: v for k, v in asdict(s).items() if k != "extra"}}

    # ---- 0. what is moved: an object (cut out + placed) or an area/texture (fills the other shape)
    prog(0.02, "Analysing the selections")
    pre_obj = None
    if s.mode not in ("object", "area"):
        s.mode, report["content"] = classify_selection(src_img, src_mask)
        if s.mode == "object" and s.extract_object:
            pre_obj, info = extract_object(src_img, src_mask)
            if info.get("method") == "selection":  # nothing separates from its surroundings: a texture/area
                s.mode, pre_obj = "area", None
                report["content"]["reason"] = "no object separates from the surroundings"
            else:
                report["source_object"] = info
    area_mode = s.mode == "area"
    if s.fit not in ("contain", "cover", "stretch", "area") or (area_mode and s.fit == "contain"):
        s.fit = "cover" if area_mode else "contain"  # an area fill must cover the destination shape
    if s.anchor not in ("center", "bottom"):
        s.anchor = "center" if area_mode else "bottom"  # objects rest where the old one stood
    report["mode"], report["fit"], report["anchor"] = s.mode, s.fit, s.anchor

    # ---- 1. selection refinement
    prog(0.05, "Refining selection edges")
    soft_r = int(2 + 10 * s.edge_softness)
    smask = (src_mask > 0.5).astype(np.float32)
    if not area_mode and s.extract_object:
        if pre_obj is None:
            pre_obj, report["source_object"] = extract_object(src_img, smask)
        smask = pre_obj.astype(np.float32)
    elif s.snap_edges and not area_mode:
        smask = matting.snap_to_edges(src_img, smask, band=max(4, int(np.sqrt(smask.sum()) * 0.04)))
    src_alpha = matting.refine_alpha(src_img, smask, radius=soft_r, eps=1e-3 if area_mode else 2e-4, feather=s.edge_softness * 1.5)
    src_fg = matting.estimate_foreground(src_img, src_alpha) if not area_mode else src_img

    dmask = (dst_mask > 0.5).astype(np.float32)
    dplace = dmask  # where the new content is fitted: the old object itself when one is found
    if not area_mode and s.extract_object:
        dobj, report["destination_object"] = extract_object(dst_img, dmask)
        if dobj.sum() >= 16 and report["destination_object"].get("fill", 1.0) >= 0.25:
            dplace = dobj.astype(np.float32)
    env_s = environment(src_img, smask)

    # ---- 2. geometry
    prog(0.15, "Fitting geometry")
    M = _transform(src_alpha, dplace, s)
    cy, cx = [float(v.mean()) for v in np.nonzero(smask > 0.5)]
    k = _linear_scale(M, np.array([cx, cy]))
    src_pre = src_fg
    if k < 0.8:  # anti-alias before shrinking
        src_pre = gaussian(src_fg, 0.45 / k)
    warped = np.clip(_warp(src_pre.astype(np.float32), M, (H, W)), 0, 1)
    walpha = np.clip(_warp(src_alpha.astype(np.float32), M, (H, W), cv2.INTER_LINEAR), 0, 1)
    if not area_mode:
        warped, walpha = _warp_foreground(src_fg, src_alpha, M, (H,W), k)
    if area_mode:
        # content fills the destination shape; edges follow the destination's own image edges
        valid = _warp(smask, M, (H, W), cv2.INTER_LINEAR) > 0.99
        alpha = matting.refine_alpha(dst_img, dmask * valid, radius=soft_r, eps=2e-3, feather=1.0 + 3 * s.edge_softness)
        if (dmask * (~valid)).sum() > 0.02 * dmask.sum():
            report.setdefault("warnings", []).append(
                "The source region does not fully cover the destination selection; uncovered parts keep the original content."
            )
    else:
        alpha = walpha
    report["geometry"] = {
        "matrix": np.round(M, 5).tolist(),
        "scale": round(k, 3),
        "perspective": bool(s.src_quad and s.dst_quad),
    }
    if k > 2.5:
        report.setdefault("warnings", []).append(
            f"The source region is enlarged {k:.1f}x; it will look softer than its surroundings."
        )

    # ---- 3. destination plate
    plate = dst_img.astype(np.float32)
    method = "none"
    if s.remove_original and not area_mode:
        prog(0.25, "Removing original content")
        # the whole old object (+ a margin for its soft edge) is re-synthesised from the
        # SURROUNDINGS only - never from the removed content - and the new object goes on top
        # (everything the user selected is replaced: parts the new object does not cover are filled too)
        g = max(2, int(np.sqrt(dplace.sum()) * 0.04))
        sel_u8 = dmask.astype(np.uint8)
        fill = dilate(np.maximum(sel_u8, dplace.astype(np.uint8)), g) > 0
        source = ~(dilate((fill | (sel_u8 > 0)).astype(np.uint8), g + 3) > 0)
        if fill.sum() > 0:
            plate, method = inpaint(plate, fill, source=source if source.sum() > 64 else None, progress=lambda p: prog(0.25 + 0.3 * p, "Removing original content"))
    report["inpainting"] = method
    env_d = environment(plate, dmask)

    # ---- 4. environment-aware harmonisation
    prog(0.6, "Adapting to the destination lighting")
    fg = warped.copy()
    support = alpha > 0.01
    if s.color_harmony > 0:
        fg = color.adapt_illuminant(fg, env_s["illuminant"], env_d["illuminant"], strength=s.color_harmony)
        # exposure: geometric-mean brightness of the two environments
        # Environment brightness differences are partly scene content (dark
        # foliage vs. bright sky), not light: adapt only part of the way.
        gain = (env_d["log_mean"] - env_s["log_mean"]) * 0.35 * s.color_harmony
        gain = float(np.clip(gain, -0.6, 0.6))
        contrast = float(np.clip(env_d["log_std"] / max(env_s["log_std"], 1e-3), 0.85, 1.15)) ** (0.5 * s.color_harmony)
        lin = color.srgb_to_linear(fg)
        lum = color.luminance(lin) + 1e-4
        ll = np.log(lum)
        wmask = (alpha > 0.5).astype(np.float32)
        mu = (ll * wmask).sum() / max(wmask.sum(), 1)
        ll_new = mu + (ll - mu) * contrast + gain
        fg = color.linear_to_srgb(lin * np.exp(ll_new - ll)[..., None])
        # black level / atmospheric haze: a hazy or low-contrast scene lifts the
        # darkest tones; a crisp scene deepens them
        bs_, bd_ = env_s["black"], env_d["black"]
        if abs(bd_ - bs_) > 0.004:
            k = 0.6 * s.color_harmony
            lin = color.srgb_to_linear(fg)
            tgt = bd_ + (lin - bs_) * (1 - bd_) / max(1 - bs_, 1e-3)
            fg = color.linear_to_srgb(np.clip(lin + (np.clip(tgt, 0, 1) - lin) * k, 0, 1))
            report["black_level"] = {"source": round(bs_, 4), "destination": round(bd_, 4)}
    if s.lighting_match > 0:
        # remove the source's illumination gradient, add the destination's
        a = np.radians(s.rotation)
        gs = env_s["gradient"].astype(np.float64)
        if s.flip:
            gs[0] = -gs[0]
        gs = np.array([np.cos(a) * gs[0] - np.sin(a) * gs[1], np.sin(a) * gs[0] + np.cos(a) * gs[1]])
        dg = (env_d["gradient"] - gs) * s.lighting_match
        dg = np.clip(dg, -0.6, 0.6)
        y0, y1, x0, x1 = bbox_of(alpha > 0.5)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = (xx - (x0 + x1) / 2) / max((x1 - x0) / 2, 1)
        v = (yy - (y0 + y1) / 2) / max((y1 - y0) / 2, 1)
        shade = np.clip(dg[0] * u + dg[1] * v, -0.8, 0.8)
        lin = color.srgb_to_linear(fg)
        fg = color.linear_to_srgb(lin * np.exp(shade)[..., None])
    fg = np.where(support[..., None], fg, warped)

    if s.match_sharpness:
        # content differs between regions, so only correct large discrepancies, at half strength
        ws = (alpha > 0.9).astype(np.float32)
        sh_fg, sh_env = analysis.sharpness(fg, ws), env_d["sharpness"]
        if not area_mode:  # an in-focus subject in front of a blurred background must stay sharp: compare with the old object too
            old = erode((dplace > 0.5).astype(np.uint8), 2).astype(np.float32)
            if old.sum() > 100:
                sh_env = max(sh_env, analysis.sharpness(dst_img, old))
        if ws.sum() > 100 and abs(sh_fg - sh_env) / max(sh_env, 1e-6) > 0.2:
            target_sh = sh_fg + (sh_env - sh_fg) * 0.5
            lo, hi = 0.0, 3.0
            if sh_fg > target_sh:
                for _ in range(12):
                    mid = (lo + hi) / 2
                    if analysis.sharpness(gaussian(fg, mid), ws) > target_sh:
                        lo = mid
                    else:
                        hi = mid
                fg = np.where(support[..., None], gaussian(fg, (lo + hi) / 2), fg)
                report["sharpness"] = {"action": "blur", "sigma": round((lo + hi) / 2, 2)}
            else:
                report["sharpness"] = {"action": "none (source softer; detail is not invented)"}
    if s.match_grain:
        ws = (alpha > 0.9).astype(np.float32)
        fg_n, nrep = analysis.match_noise(fg, plate, ws, env_d["ring"])
        fg = np.where(support[..., None], fg_n, fg)
        report["noise"] = nrep

    # ---- 5. compositing
    prog(0.75, "Compositing")
    bm = s.blend_mode
    if bm == "auto":
        bm = "poisson" if area_mode else "matte"
    plate_c = plate.copy()
    if s.contact_shadow > 0 and not area_mode:
        plate_c = _contact_shadow(plate_c, alpha, env_d, s.contact_shadow)
    if bm == "matte":
        if s.light_wrap > 0:
            fg = _light_wrap(fg, plate_c, alpha, s.light_wrap)
        result = blend.alpha_composite(fg, plate_c, alpha)
    elif bm in ("poisson", "mixed"):
        if area_mode:
            region = (alpha > 0.5).astype(np.float32)
        else:
            region = dilate((alpha > 0.5).astype(np.uint8), max(2, soft_r)).astype(np.float32)
        cloned = blend.poisson_blend(fg, plate_c, region, mode="normal" if bm == "poisson" else "mixed", robust=True)
        if area_mode:
            # extend the Poisson correction into the feather band, then multi-band composite
            corr = (cloned - fg) * region[..., None]
            corr = np.where(region[..., None] > 0, corr, normalized_blur(corr, region, 4.0))
            result = blend.multiband_blend(np.clip(fg + corr, 0, 1), plate_c, alpha)
        else:
            result = blend.alpha_composite(cloned, plate_c, np.maximum(alpha, gaussian(region, 1.0) * (alpha > 0.02)))
    else:  # multiband
        result = blend.multiband_blend(fg, plate_c, alpha)

    # guarantee: nothing outside the edited area changes
    changed = dilate(((alpha > 0.002) | (np.abs(plate - dst_img).max(-1) > 0)).astype(np.uint8), 2) > 0
    if s.contact_shadow > 0:
        changed |= np.abs(plate_c - plate).max(-1) > 0
    result = np.where(changed[..., None], np.clip(result, 0, 1), dst_img)

    report["environment"] = {"source": _env_report(env_s), "destination": _env_report(env_d)}
    report["blend_mode"] = bm
    report["changed_fraction"] = round(float(changed.mean()), 4)
    report["quality"] = {"seam_visibility": round(_seam(result, alpha), 3)}
    prog(1.0, "Done")
    return result.astype(np.float32), report, {"alpha": alpha, "plate": plate}


def _light_wrap(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray, amount: float) -> np.ndarray:
    """Let background light spill over the object's edges (as real optics do)."""
    y0, y1, x0, x1 = bbox_of(alpha > 0.01, pad=40)
    a = alpha[y0:y1, x0:x1]
    size = max(2.0, np.sqrt((a > 0.5).sum()) * 0.02)
    blurred_bg = normalized_blur(bg[y0:y1, x0:x1], 1 - a, size)
    edge = np.clip(a * (1 - gaussian(a, size)) * 2.5, 0, 1) * amount
    out = fg.copy()
    sub = out[y0:y1, x0:x1]
    # screen-like blend keeps highlights natural
    wrapped = 1 - (1 - sub) * (1 - blurred_bg * 0.6)
    out[y0:y1, x0:x1] = sub * (1 - edge[..., None]) + wrapped * edge[..., None]
    return out


def _contact_shadow(plate: np.ndarray, alpha: np.ndarray, env: dict, strength: float) -> np.ndarray:
    """Soft ambient-occlusion shadow under the object, offset away from the estimated light."""
    a = (alpha > 0.5).astype(np.float32)
    y0, y1, x0, x1 = bbox_of(a)
    h = max(y1 - y0, 1)
    # squash vertically towards the base and shift opposite to the light gradient
    g = env["gradient"]
    dx = float(np.clip(-g[0], -0.5, 0.5)) * (x1 - x0) * 0.3
    M = np.float32([[1, 0, dx], [0, 0.18, (y1 - 0.18 * y1) + h * 0.02]])
    sh = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
    sh = gaussian(sh, max(2.0, h * 0.05)) * (1 - a)
    sh = np.clip(sh * 1.8, 0, 1) * strength * 0.65
    return plate * (1 - sh[..., None])


def _seam(img: np.ndarray, alpha: np.ndarray) -> float:
    g = color.luminance(img)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    edge = ((alpha > 0.3) & (alpha < 0.7)).astype(np.float32)
    near = np.clip(dilate(edge.astype(np.uint8), 6).astype(np.float32) - edge, 0, 1)
    if edge.sum() < 10:
        return 1.0
    return float((mag * edge).sum() / edge.sum() / ((mag * near).sum() / max(near.sum(), 1) + 1e-6))


def swap_regions(
    img_a: np.ndarray,
    mask_a: np.ndarray,
    img_b: np.ndarray,
    mask_b: np.ndarray,
    settings: RegionSwapSettings | None = None,
    direction: str = "both",
    progress=None,
) -> dict:
    """Exchange the selected regions. Returns dict with results/reports per direction."""
    prog = progress or (lambda *_: None)
    out = {}
    if direction in ("both", "b_to_a"):
        r, rep, dbg = insert_region(img_b, mask_b, img_a, mask_a, settings, lambda p, m: prog(p * (0.5 if direction == "both" else 1.0), f"A: {m}"))
        out["a"] = (r, rep, dbg)
    if direction in ("both", "a_to_b"):
        base = 0.5 if direction == "both" else 0.0
        r, rep, dbg = insert_region(img_a, mask_a, img_b, mask_b, settings, lambda p, m: prog(base + p * (1 - base), f"B: {m}"))
        out["b"] = (r, rep, dbg)
    return out
