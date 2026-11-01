"""Single-image 3D scene reconstruction (layered depth mesh).

A single photo only shows the surfaces visible from one viewpoint, so the
engine builds a *two-layer depth image* (in the spirit of layered depth
images / "3D photo inpainting") plus a coarse far backdrop:

1. **Depth**: Depth Anything V2 (or MiDaS) disparity with detail fusion and
   edge snapping; the sky is detected and pushed to a far dome distance.
2. **Foreground decomposition**: people (U2-Net human segmentation) and
   salient objects (U2-Net "silueta") that are clearly nearer than their
   surroundings in the depth map are cut out *as a whole*. The depth is
   snapped to their silhouettes (no blurred depth ramp, no "flying" pixels)
   and the main mesh is torn exactly along them, so they separate cleanly
   from what is behind them when the camera moves.
3. **Background plate**: everything behind the foreground is completed - its
   depth by a harmonic (Laplace) continuation of the surrounding background
   only (floors and walls continue smoothly, always behind the foreground),
   its colour by a learned inpainting model when installed (LaMa, MI-GAN) or
   by depth-guided PatchMatch that copies real background texture (patches
   are matched on colour, *depth of the hidden surface* and height, and only
   background pixels may be copied, never the person).
4. **Other depth edges**: behind every remaining depth discontinuity (an
   object the segmenters did not pick up) a band of hidden background is
   completed the same way.
5. **Out-of-frame margins**: the borders are outpainted (photo and
   background plate together, so a person cut by the frame continues and so
   does the background behind them); out-of-frame depth continues the slope
   of the border (floors keep receding) and is kept smooth. The margin is
   marked as low-confidence and rendered with a subtle falloff.
6. **Output**: two grids (main + background layer) of per-vertex disparity,
   quad-validity masks and textures, plus a coarse backdrop that shows soft
   *background* colours wherever both layers are missing. The viewer
   unprojects them in a vertex shader, so field-of-view and depth strength
   stay adjustable in real time.

Limitations are explicit: content that is hidden by large objects or lies
outside the frame is *estimated*, not recovered; large camera moves expose
it. The viewer constrains movement to a safe range by default.
"""

from __future__ import annotations

import io
import json
import struct
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from ..depth.estimator import estimate_depth
from ..imaging.filters import gaussian, normalized_blur
from ..imaging.inpaint import available_learned, inpaint, learned_inpaint, patchmatch_inpaint
from ..imaging.io import encode_image, resize_max

_DEBUG: dict | None = None  # set to a dict to collect intermediate maps (development aid)


def _ab(flag: str) -> bool:
    """Development A/B switch (LUMEN_RECON_AB=flag1,flag2 disables the named refinements) - lets the
    evaluation harness compare one change at a time while the rest of the code base moves."""
    import os

    return flag in os.environ.get("LUMEN_RECON_AB", "").split(",")

from .geometry import (SKY_DISP, REF_ZFAR, disp_to_z, z_to_disp, _behind, detect_sky, _cuts, _quad_keep, _disk, _dil, _ero, _fill_small_holes, _sharpen_depth_edges, _flip_small, _snap_ramps, _snap_inside, _near_side, _cut_sides)  # noqa: F401,E402
from .completion import (_separated, _foreground, _occlusion_band, _background_border, _harmonic_fill)  # noqa: F401,E402

try:  # D2: planar continuation of hidden floors/walls (drop-in for _harmonic_fill)
    from .completion import _hidden_depth  # noqa: E402
except ImportError:  # pragma: no cover
    _hidden_depth = None
from .outpaint import (_extend_depth, _border_objects, _extend_mask, _embed, _outpaint)  # noqa: F401,E402


@dataclass
class ReconSettings:
    model: str = "depth-anything-v2-small"
    quality: str = "balanced"  # fast | balanced | high
    detail_fusion: bool = True
    occlusion_width: float = 0.05  # fraction of the long side completed behind other depth edges
    edge_threshold: float = 0.045  # relative depth jump that tears the mesh
    outpaint_margin: float = 0.10  # fraction of the long side synthesised beyond each side of the frame
    detect_sky: bool = True
    remove_foreground: bool = True  # complete the whole background behind people and separated objects
    inpainting: str = "auto"  # auto (learned model when installed, else PatchMatch) | patchmatch
    complete_foreground: bool = True  # grow partially segmented people/objects over their depth-separated region

    @classmethod
    def from_dict(cls, d: dict | None) -> "ReconSettings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                cur = getattr(s, k)
                if isinstance(cur, bool) and isinstance(v, str):
                    v = v.strip().lower() in ("1", "true", "yes", "on")
                setattr(s, k, type(cur)(v))
        return s


# work resolution (grid), texture resolution, max. hole extent of the PatchMatch background plate
_QUALITY = {"fast": (640, 1600, 300), "balanced": (900, 2048, 420), "high": (1280, 3072, 640)}


def reconstruct(rgb: np.ndarray, settings: ReconSettings | None = None, progress=None) -> dict:
    s = settings or ReconSettings()
    user_prog = progress or (lambda *_: None)
    work_side, tex_side, plate_extent = _QUALITY.get(s.quality, _QUALITY["balanced"])
    timings: dict[str, float] = {}
    report: dict = {"settings": asdict(s), "warnings": [], "timings": timings}
    stage = {"name": None, "t": time.perf_counter(), "t0": time.perf_counter()}

    def prog(p, label=""):
        # wall time per stage (keyed by the progress label) for the report / benchmarks
        now = time.perf_counter()
        if label != stage["name"]:
            if stage["name"] is not None:
                timings[stage["name"]] = round(timings.get(stage["name"], 0.0) + now - stage["t"], 3)
            stage.update(name=label, t=now)
        user_prog(p, label)

    tex, _ = resize_max(rgb, tex_side)
    work, _ = resize_max(rgb, work_side)
    h, w = work.shape[:2]
    L = max(h, w)
    thr = s.edge_threshold
    fill_method = "patchmatch" if s.inpainting == "patchmatch" else "auto"

    # ---- depth
    prog(0.02, "Estimating depth")
    depth_src, _ = resize_max(rgb, 1600)
    # detail crops finer than ~1.25x the mesh resolution are averaged away (fast/balanced); 'high' keeps them all
    detail_side = None if s.quality == "high" or _ab("no_detail_cap") else int(1.25 * work_side)
    try:
        disp_full = estimate_depth(depth_src, s.model, detail=s.detail_fusion, detail_side=detail_side,
                                   progress=lambda p: prog(0.02 + 0.28 * p, "Estimating depth"))
    except TypeError:  # estimator without the detail cap
        disp_full = estimate_depth(depth_src, s.model, detail=s.detail_fusion, progress=lambda p: prog(0.02 + 0.28 * p, "Estimating depth"))
    disp = cv2.resize(disp_full, (w, h), interpolation=cv2.INTER_AREA)
    from ..depth.estimator import refine_edges

    disp = refine_edges(work, disp)
    disp = _sharpen_depth_edges(work, disp, thr)
    sky = detect_sky(work, disp) if s.detect_sky else np.zeros((h, w), bool)
    report["sky_fraction"] = round(float(sky.mean()), 3)
    d = disp.copy()
    d[sky] = SKY_DISP

    # ---- foreground: people and objects that stand clearly in front of their surroundings
    prog(0.31, "Finding people and objects")
    if s.remove_foreground:
        fg, people, n_obj = _foreground(work, d, sky, thr, report)
        if s.complete_foreground:
            fg, people, grown = _complete_foreground(work, d, fg, people, sky, thr)
            report["foreground_grown"] = round(grown, 4)
        # mixed edge pixels (part person, part background) go with the person: left on the
        # background they would trail a fringe of the person's colour when the camera moves
        fg = _dil(fg, max(1, round(0.0012 * L))) & ~sky
    else:
        fg, people, n_obj = np.zeros((h, w), bool), np.zeros((h, w), bool), 0
    report["foreground"] = {"people": bool(people.any()), "objects": int(n_obj), "fraction": round(float(fg.mean()), 3)}
    r_in, r_halo, r_bleed = max(2, round(0.018 * L)), max(2, round(0.006 * L)), max(4, round(0.022 * L))
    # other depth edges: resolve blurred ramps into steps (never inside a person or object,
    # whose surface must stay in one piece - what is behind an arm is not in the background plate)
    keep_fg = _dil(fg, r_bleed)
    d = np.where(keep_fg, d, _snap_ramps(work, d, thr, sky | keep_fg)).astype(np.float32)
    disp = np.where(sky, disp, d).astype(np.float32)
    depth_vis = disp.copy()

    # ---- occlusion analysis on the silhouette-snapped depth
    prog(0.37, "Analysing occlusions")
    d_main = _snap_inside(d, fg, r_in)
    near, far_d = _near_side(d_main, thr)
    band, nsurf = _occlusion_band(d_main, near, far_d, thr, s.occlusion_width)
    nsurf &= ~sky

    # ---- depth of the hidden background: harmonic continuation of the surrounding background
    prog(0.4, "Estimating hidden background depth")
    near_fg = _dil(fg, r_bleed)  # monocular depth "bleeds" a few pixels beyond a silhouette
    hole_d = near_fg | band
    dirichlet = _background_border(d_main, hole_d, fg, nsurf, thr, r_bleed)
    if _hidden_depth is not None and not _ab("no_hidden_planes"):
        bg_full = _hidden_depth(d_main, hole_d, dirichlet, sky=sky)
    else:
        bg_full = _harmonic_fill(d_main, hole_d, dirichlet)
    behind_needed = fg | (band & ~near_fg)
    bg_full = np.where(behind_needed, np.minimum(bg_full, _behind(d_main, 2.0 * thr)), bg_full).astype(np.float32)
    # around a silhouette the photo shows background: it takes the continued background
    # depth, so the silhouette becomes a clean step instead of a blurred ramp
    halo = hole_d & ~behind_needed
    d_main = np.where(halo, bg_full, d_main).astype(np.float32)

    # ---- colour of the hidden background (person removed, not just a band)
    prog(0.43, "Synthesising hidden background")
    hole_c = _dil(fg, r_halo) | band
    source = ~_dil(hole_c, r_halo) & ~near_fg & ~nsurf
    # guide channels (known inside the hole too): depth of the hidden surface - weak, because
    # monocular depth of plain walls is not consistent enough to drive the match - and image
    # position, which keeps the fill local (texture from the same height and the neighbourhood)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) * (0.24 / h)
    guide = np.dstack([np.maximum(bg_full, SKY_DISP) * 0.15, yy, xx])
    info: dict = {}
    plate, method = inpaint(work, hole_c, source=source, method=fill_method, guide=guide, max_extent=plate_extent, info=info,
                            progress=lambda p: prog(0.43 + 0.3 * p, "Synthesising hidden background"))
    plate = np.clip(plate, 0, 1).astype(np.float32)
    report["inpainting"] = method
    if info.get("fallback"):
        report["warnings"].append("Learned inpainting failed, PatchMatch was used instead: " + "; ".join(info["fallback"]))

    # ---- out-of-frame margins (photo + background plate), depth continued beyond the border
    prog(0.74, "Extending beyond the frame")
    m = int(round(max(0.0, s.outpaint_margin) * L))
    W, H = w + 2 * m, h + 2 * m
    margin = np.ones((H, W), bool)
    margin[m : m + h, m : m + w] = False
    # people and near objects cut by the frame continue as a rounded cap; everything
    # else beyond the border is the continuation of the background behind them
    border_obj = (_border_objects(d_main, thr) | fg) & ~sky
    ext_big = _extend_mask(border_obj, m)
    ext_fg = _extend_mask(fg, m) & margin  # segmented foreground: gets a background layer behind it
    fg_big = _embed(np.zeros((h, w), bool), m)
    fg_big[m : m + h, m : m + w] = fg
    fg_big |= ext_big & margin
    bg_trend = bg_full
    if (border_obj & ~fg).any():
        # background behind generic border objects: harmonic continuation from around them
        bo = _dil(border_obj & ~fg, 2)
        bg_trend = _harmonic_fill(bg_full, bo, _background_border(bg_full, bo, border_obj, nsurf & ~bo, thr, r_bleed))
        bg_trend = np.where(bo, np.minimum(bg_trend, _behind(d_main, 2.0 * thr)), bg_full).astype(np.float32)
    dbg_big = _extend_depth(bg_trend, m)
    sky_big = _embed(sky, m)
    dbg_big[sky_big & margin] = SKY_DISP
    dmain_big = _embed(d_main, m)  # replicated: an object keeps its depth beyond the border
    dmain_big = np.where(margin & ~ext_big, dbg_big, dmain_big).astype(np.float32)
    dmain_big[m : m + h, m : m + w] = d_main
    if m > 0:
        big, plate_big, out_method = _outpaint(work, plate, dmain_big, m, ext_fg, fill_method, lambda p: prog(0.74 + 0.13 * p, "Extending beyond the frame"))
    else:
        big, plate_big, out_method = work.astype(np.float32).copy(), plate.copy(), "none"
    report["outpainting"] = out_method

    # ---- grids
    prog(0.88, "Building mesh")
    hole_big = _embed(np.zeros((h, w), bool), m)
    hole_big[m : m + h, m : m + w] = hole_c
    hole_big |= _dil(ext_fg, r_halo) & margin
    bgd_big = dbg_big.copy()
    bgd_big[m : m + h, m : m + w] = bg_full
    bgd_big = np.where(fg_big & hole_big, np.minimum(bgd_big, _behind(dmain_big, 2.0 * thr)), bgd_big).astype(np.float32)

    # main layer: depth cuts everywhere, plus an exact cut along the silhouettes; the
    # two-pixel ramp test is not needed there (the depth was snapped)
    ch, cv_, _ = _cuts(dmain_big, thr)
    if fg_big.any():
        zb = disp_to_z(np.maximum(dmain_big, SKY_DISP))
        step_h = np.maximum(zb[:, 1:], zb[:, :-1]) / np.minimum(zb[:, 1:], zb[:, :-1]) - 1
        step_v = np.maximum(zb[1:, :], zb[:-1, :]) / np.minimum(zb[1:, :], zb[:-1, :]) - 1
        sil_h = fg_big[:, 1:] != fg_big[:, :-1]
        sil_v = fg_big[1:, :] != fg_big[:-1, :]
        zone = _dil(fg_big, 2) & ~_ero(fg_big, 2)
        zh, zv = zone[:, 1:] | zone[:, :-1], zone[1:, :] | zone[:-1, :]
        ch = np.where(zh, (sil_h & (step_h > 0.5 * thr)) | (~sil_h & (step_h > thr)), ch)
        cv_ = np.where(zv, (sil_v & (step_v > 0.5 * thr)) | (~sil_v & (step_v > thr)), cv_)
        # inside a person/object the surface stays in one piece: a tear there would show the
        # far background through the body (what is behind an arm is unknown)
        # ... but only small steps (an arm in front of the torso): a large jump inside the mask is
        # background seen through a gap (between legs, bike spokes, a person and a pole) - the plate
        # behind the foreground is completed, so tearing there shows plausible background instead of
        # a rubber sheet stretched across the gap
        core = _ero(fg_big, 3)
        big_h, big_v = step_h > 3.0 * thr, step_v > 3.0 * thr
        if _ab("no_core_big_tear"):
            big_h, big_v = np.zeros_like(big_h), np.zeros_like(big_v)
        ch &= ~(core[:, 1:] & core[:, :-1] & ~big_h)
        cv_ &= ~(core[1:, :] & core[:-1, :] & ~big_v)
    valid = np.ones((H, W), bool)
    keep_main = _quad_keep(dmain_big, valid, thr, cuts=(ch, cv_))
    # every tear needs the background layer behind it, otherwise it shows the backdrop or
    # nothing - even from the original viewpoint. Near-side pixels the occlusion analysis did
    # not cover get the depth of the surface right behind them.
    near_c, far_c, far_cd, thin_c = _cut_sides(dmain_big, ch, cv_)
    need = near_c & ~hole_big & np.isfinite(far_cd)
    hole_big |= need
    bgd_big = np.where(need, far_cd, bgd_big).astype(np.float32)
    torn = ~keep_main
    bg_valid = _dil(hole_big, 1)
    bg_valid[:-1, :-1] |= torn
    bg_valid[1:, :-1] |= torn
    bg_valid[:-1, 1:] |= torn
    bg_valid[1:, 1:] |= torn
    bg_disp = np.where(bg_valid, np.where(hole_big, bgd_big, dmain_big), np.nan).astype(np.float32)
    keep_bg = _quad_keep(np.nan_to_num(bg_disp, nan=0.5), bg_valid, thr * 3.0)

    # background-layer colours: the photo wherever that layer lies on the main surface (it is
    # seen through tears, and WebGL resolves exact depth ties in favour of the layer drawn
    # last), the completed plate where it lies behind it. Next to a tear the plate starts with
    # the colour of the far side, so the original view stays clean and the revealed background
    # continues the visible one without a seam; parts one pixel thin keep their own colour.
    behind = hole_big & (bgd_big < dmain_big - 1e-5)
    bg_tex = np.where(behind[..., None], plate_big, big).astype(np.float32)
    far_ok = far_c & ~behind
    if far_ok.any() and behind.any():
        dist, (iy0, ix0) = ndimage.distance_transform_edt(~far_ok, return_indices=True)
        wgt = (np.clip((3.5 - dist) / 2.5, 0.0, 1.0) * behind).astype(np.float32)[..., None]
        # the far-side pixels right at a tear are anti-aliased mixes that still carry the
        # occluder's colour: sample the colour a few pixels into the visible background instead,
        # otherwise a thin outline of the person/object stays painted on the background
        clean = ~behind & ~_dil(near_c | fg_big, 3) & _dil(far_ok, 6)
        src = clean if clean.any() and not _ab("no_clean_edge") else far_ok
        _, (iy, ix) = ndimage.distance_transform_edt(~src, return_indices=True)
        # ... except in the one-pixel crack a tear opens even at the reference view, which must
        # keep the photo's own edge colour
        col = np.where((dist <= 1.0)[..., None], big[iy0, ix0], big[iy, ix])
        bg_tex = bg_tex * (1.0 - wgt) + col * wgt
    bg_tex[thin_c] = big[thin_c]

    # backdrop: coarse surface at the farthest local background depth, always behind the
    # real geometry; large disocclusions reveal soft background colour instead of a void
    bs = 4
    hb, wb = (H + bs - 1) // bs, (W + bs - 1) // bs
    dpad = np.pad(bgd_big, ((0, hb * bs - H), (0, wb * bs - W)), mode="edge")
    back = dpad.reshape(hb, bs, wb, bs).min(axis=(1, 3))
    back = ndimage.grey_erosion(back, size=(7, 7))
    back = gaussian(back.astype(np.float32), 2.0)
    back = np.minimum(back, ndimage.grey_erosion(back, size=(3, 3))) - 0.01
    # its colour comes from far surfaces only (near objects that were not cut out would
    # otherwise appear as blurred copies behind themselves)
    plate_depth = np.where(hole_big, bgd_big, dmain_big)
    back_up = cv2.resize(back, (W, H), interpolation=cv2.INTER_LINEAR)
    wfar = np.exp(-np.clip(plate_depth - back_up - 0.02, 0, None) / 0.03).astype(np.float32)
    bt = normalized_blur(plate_big, wfar + 1e-3, max(2.0, 0.015 * max(H, W)))
    back_tex = gaussian(cv2.resize(bt, (W // 2, H // 2), interpolation=cv2.INTER_AREA), max(2.0, 0.004 * max(H, W)))

    if _DEBUG is not None:
        _DEBUG.update(sky=sky, fg=fg, people=people, band=band, nsurf=nsurf, hole_c=hole_c, hole_d=hole_d, d=d, d_main=d_main, bg_full=bg_full,
                      plate=plate, source=source, m=m, dmain_big=dmain_big, bgd_big=bgd_big, keep_main=keep_main, keep_bg=keep_bg, hole_big=hole_big, work=work)
    report["occluded_fraction"] = round(float(hole_big.mean()), 3)
    report["main_triangles"] = int(keep_main.sum() * 2)
    report["background_triangles"] = int(keep_bg.sum() * 2)
    if report["sky_fraction"] > 0.6:
        report["warnings"].append("Most of the image is sky; there is little geometry to reconstruct.")
    if disp.std() < 0.05:
        report["warnings"].append("The scene is almost flat (little depth variation); parallax will be subtle.")

    # main texture at high resolution: original pixels in the frame, outpainted margin around it
    th_, tw_ = tex.shape[:2]
    tmx, tmy = int(round(m * tw_ / w)), int(round(m * th_ / h))
    main_tex = cv2.resize(big, (tw_ + 2 * tmx, th_ + 2 * tmy), interpolation=cv2.INTER_CUBIC)
    main_tex[tmy : tmy + th_, tmx : tmx + tw_] = tex
    prog(0.95, "Packaging scene")
    header = {
        "version": 1,
        "grid_w": W,
        "grid_h": H,
        "frame": {"x0": m / W, "y0": m / H, "x1": (m + w) / W, "y1": (m + h) / H},
        "extent_x": W / w,
        "extent_y": H / h,
        "aspect": w / h,
        "sky_disparity": SKY_DISP,
        "ref_zfar": REF_ZFAR,
        "default_fov": 60.0,
        "default_zfar": 8.0,
        "safe_radius": 0.35,
    }
    header["back_w"], header["back_h"] = int(wb), int(hb)
    blob = pack_scene(header, dmain_big.astype(np.float32), keep_main, np.nan_to_num(bg_disp, nan=-1.0).astype(np.float32), keep_bg, back.astype(np.float32))
    depth_png = encode_image(np.dstack([np.clip(depth_vis, 0, 1)] * 3), "png", bit_depth=16)
    report["grid"] = [W, H]
    prog(1.0, "Done")
    report["time_total"] = round(time.perf_counter() - stage["t0"], 3)
    return {
        "scene": blob,
        "main_texture": encode_image(np.clip(main_tex, 0, 1), "jpg", quality=92),
        "bg_texture": encode_image(np.clip(bg_tex, 0, 1), "jpg", quality=90),
        "back_texture": encode_image(np.clip(back_tex, 0, 1), "jpg", quality=85),
        "depth_png": depth_png,
        "depth_preview": encode_image(_colorize(depth_vis), "jpg", quality=90),
        "report": report,
        "header": header,
    }

def _complete_foreground(work: np.ndarray, d: np.ndarray, fg: np.ndarray, people: np.ndarray, sky: np.ndarray,
                         thr: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Grow partially segmented people/objects over the depth-separated region they belong to.

    Segmenters often return only part of a person (a head without its shoulders, half a face at
    the edge of a group). Left like that, the rest of the person is treated as background: the
    halo next to the silhouette is pushed back onto the background plate and the mesh is torn
    along the segmentation boundary, so faces and bodies split apart when the camera moves. A
    grey-scale opening with a large disk reproduces floors and walls but removes narrow near
    objects, so pixels clearly nearer than the opened map stand in front of their surroundings;
    such regions that touch the foreground at the same depth (depth continuity at the contact)
    join it - never floors (the opening reproduces them) and never the sky. Returns (fg, people,
    added area fraction)."""
    h, w = d.shape
    if not fg.any():
        return fg, people, 0.0
    L = max(h, w)
    f = 4
    small = cv2.resize(d.astype(np.float32), (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    r = max(3, int(round(0.12 * L / f)))
    ext = _extend_depth(small, r + 1)
    opened = cv2.morphologyEx(ext, cv2.MORPH_OPEN, _disk(r), borderType=cv2.BORDER_REPLICATE)[r + 1 : -(r + 1), r + 1 : -(r + 1)]
    opened = cv2.resize(opened, (w, h), interpolation=cv2.INTER_LINEAR)
    z = disp_to_z(np.maximum(d, SKY_DISP))
    near = (disp_to_z(np.maximum(opened, SKY_DISP)) / z - 1 > 2.0 * thr) & ~sky & ~fg
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, _disk(1)) > 0
    if not near.any():
        return fg, people, 0.0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(near.astype(np.uint8), connectivity=8)
    contact = _dil(fg, 2) & near
    ids = np.unique(lab[contact])
    ids = ids[ids > 0]
    fg_area = float(fg.sum())
    # bounded: a head-only mask may need its whole body (several times its area), but the
    # foreground never grows past half the image
    budget = min(max(2.0 * fg_area, 0.10 * h * w), 0.45 * h * w - fg_area)
    add = np.zeros((h, w), bool)
    add_people = np.zeros((h, w), bool)
    for i in sorted(ids, key=lambda k: -stats[k, cv2.CC_STAT_AREA]):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area > budget or area < 20:
            continue
        x, y, bw, bh = stats[i, :4]
        pad = 4
        y0, y1, x0, x1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
        comp = lab[y0:y1, x0:x1] == i
        cfg = fg[y0:y1, x0:x1]
        cz = z[y0:y1, x0:x1]
        touch = _dil(cfg, 2) & comp
        side = _dil(touch, 2) & cfg
        if touch.sum() < 3 or side.sum() < 3:
            continue
        z_c, z_f = float(np.median(cz[touch])), float(np.median(cz[side]))
        if abs(z_c / max(z_f, 1e-6) - 1) > 2.0 * thr:
            continue  # a different surface that merely touches the silhouette in the image
        add[y0:y1, x0:x1] |= comp
        if (people[y0:y1, x0:x1] & side).any():
            add_people[y0:y1, x0:x1] |= comp
        budget -= area
    if not add.any():
        return fg, people, 0.0
    min_area = max(40.0, 0.002 * h * w)
    fg2 = _fill_small_holes(fg | add, min_area * 0.25) & ~sky
    return fg2, people | (add_people & fg2), float((fg2 & ~fg).mean())


def _colorize(d: np.ndarray) -> np.ndarray:
    u8 = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32) / 255

def pack_scene(header: dict, disp: np.ndarray, keep: np.ndarray, bg_disp: np.ndarray, keep_bg: np.ndarray, back: np.ndarray) -> bytes:
    """Binary scene container: 'LSC1' | u32 header length | JSON | aligned buffers."""
    bufs = [disp.astype("<f4").tobytes(), np.packbits(keep.ravel()).tobytes(), bg_disp.astype("<f4").tobytes(), np.packbits(keep_bg.ravel()).tobytes(), back.astype("<f4").tobytes()]
    layout, off = [], 0
    for b in bufs:
        layout.append([off, len(b)])
        off += len(b)
        off += (-off) % 4
    header = dict(header, buffers=layout, buffer_names=["disp", "keep", "bg_disp", "keep_bg", "back_disp"])
    hj = json.dumps(header).encode()
    hj += b" " * ((-len(hj) - 8) % 4)
    out = io.BytesIO()
    out.write(b"LSC1")
    out.write(struct.pack("<I", len(hj)))
    out.write(hj)
    for b in bufs:
        out.write(b)
        out.write(b"\0" * ((-len(b)) % 4))
    return out.getvalue()
