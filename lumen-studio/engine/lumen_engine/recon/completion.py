"""Completion of what the photo does not show behind people and objects: foreground
decomposition, occlusion bands and smooth continuation of the background depth."""

from __future__ import annotations

import io
import json
import struct
from dataclasses import asdict, dataclass
import cv2
import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from ..depth.estimator import estimate_depth
from ..imaging.filters import gaussian, normalized_blur
from ..imaging.inpaint import available_learned, inpaint, learned_inpaint, patchmatch_inpaint
from ..imaging.io import encode_image, resize_max
from .geometry import (SKY_DISP, REF_ZFAR, disp_to_z, z_to_disp, _behind, detect_sky, _cuts, _quad_keep, _disk, _dil, _ero, _fill_small_holes, _sharpen_depth_edges, _flip_small, _snap_ramps, _snap_inside, _near_side, _cut_sides)  # noqa: F401


def _separated(d: np.ndarray, comp: np.ndarray, thr: float, ring_r: int) -> bool:
    """Is the region clearly nearer than a ring of its surroundings (real depth separation)?"""
    core = _ero(comp, 2)
    if core.sum() < 20:
        core = comp
    ring = _dil(comp, ring_r) & ~_dil(comp, 2)
    if ring.sum() < 20:
        return False
    d_in, d_ring = float(np.median(d[core])), float(np.median(d[ring]))
    z_in, z_ring = disp_to_z(max(d_in, SKY_DISP)), disp_to_z(max(d_ring, SKY_DISP))
    return z_ring / z_in - 1 > max(3 * thr, 0.12) and d_in - d_ring > 0.04

def _foreground(work: np.ndarray, d: np.ndarray, sky: np.ndarray, thr: float, report: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Binary masks of people and depth-separated salient objects (work resolution)."""
    from ..imaging import u2net
    from ..models import manager

    h, w = d.shape
    min_area = max(40.0, 0.002 * h * w)
    people = np.zeros((h, w), bool)
    objects = np.zeros((h, w), bool)
    n_obj = 0
    if manager.is_installed("segment-human"):
        try:
            pm = u2net.person_mask(work)
            n, lab, stats, _ = cv2.connectedComponentsWithStats((pm > 0.5).astype(np.uint8), connectivity=8)
            ok = np.zeros(n, bool)
            ok[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
            # soft edges (hair, motion blur) go with the person: left on the background they would
            # trail a person-coloured outline when the camera moves
            people = _hysteresis(ok[lab], pm > 0.2, max(2, round(0.008 * max(h, w))))
            people = _fill_small_holes(people, min_area * 0.25) & ~sky
        except Exception as exc:  # noqa: BLE001 - segmentation is an optional refinement
            report["warnings"].append(f"Person segmentation failed ({exc}); only depth edges are used.")
    if manager.is_installed("segment-silueta"):
        try:
            sal = u2net.predict(work, "segment-silueta")
            # people are already covered; what remains of the salient area is split into
            # separate objects, each kept only when it really stands in front of its surroundings
            cand = (sal > 0.5) & ~sky & ~_dil(people, 2)
            cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, _disk(1)) > 0
            n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
            ring_r = max(4, int(0.03 * max(h, w)))
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area or area > 0.75 * h * w:
                    continue
                x, y, bw, bh = stats[i, :4]
                pad = ring_r + 3
                y0, y1, x0, x1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
                comp = lab[y0:y1, x0:x1] == i
                if _separated(d[y0:y1, x0:x1], comp, thr, ring_r):
                    objects[y0:y1, x0:x1] |= _fill_small_holes(comp, min_area * 0.25)
                    n_obj += 1
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append(f"Object segmentation failed ({exc}); only depth edges are used.")
    return people | objects, people, n_obj

def _hysteresis(core: np.ndarray, weak: np.ndarray, steps: int) -> np.ndarray:
    """``core`` grown into the connected ``weak`` pixels by at most ``steps`` pixels."""
    out = core.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        nxt = (cv2.dilate(out.astype(np.uint8), k) > 0) & weak | out
        if np.array_equal(nxt, out):
            break
        out = nxt
    return out

def _occlusion_band(d: np.ndarray, near: np.ndarray, far_d: np.ndarray, thr: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """(band, near_surface): the strip of near surface behind which hidden background is
    completed, and every pixel that belongs to the near side of a depth edge."""
    h, w = d.shape
    if not near.any():
        z = np.zeros((h, w), bool)
        return z, z
    band_max = max(3, int(width * max(h, w)))
    dist_near, (iy, ix) = ndimage.distance_transform_edt(~near, return_indices=True)
    d_edge = d[iy, ix]  # disparity of the nearest near-edge pixel
    d_far = far_d[iy, ix]  # disparity of the surface hidden behind it
    d_far = np.where(np.isfinite(d_far), d_far, d)
    jump = np.clip(d_edge - d_far, 0, 1)
    # disocclusion width grows with the disparity jump (parallax)
    band = band_max * np.clip(jump / 0.25, 0.25, 1.0)
    z_p, z_e, z_f = disp_to_z(np.maximum(d, SKY_DISP)), disp_to_z(np.maximum(d_edge, SKY_DISP)), disp_to_z(np.maximum(d_far, SKY_DISP))
    on_near = (z_f / z_p - 1 > thr) & (np.abs(z_p / z_e - 1) < 0.6 * (z_f / z_e - 1))
    occl = (dist_near <= band) & on_near
    # every near-side edge pixel needs background behind it, otherwise the
    # tear leaves a crack even at the original viewpoint
    occl |= near & np.isfinite(far_d)
    nsurf = (on_near & (dist_near <= band * 1.5 + 3)) | occl
    return occl, nsurf

def _background_border(d: np.ndarray, hole: np.ndarray, fg: np.ndarray, nsurf: np.ndarray, thr: float, r_fg: int) -> np.ndarray:
    """Trustworthy background pixels on the border of ``hole`` (data for the depth continuation).

    Rejected: near surfaces of depth edges, depth ramps (blurred edges) and,
    close to the foreground, pixels that are not clearly behind it (depth
    contaminated by the foreground, or another object at the same depth) -
    except ground contact *below* the foreground (feet on a floor).
    """
    border = _dil(hole, 1) & ~hole & ~nsurf
    z = disp_to_z(np.maximum(d, SKY_DISP))
    step = np.zeros(d.shape, np.float32)
    # steps towards hole pixels do not count: the far side of a depth edge is exactly the
    # surface that continues behind the near side (its step to the near side is the edge itself)
    rh = np.where(hole[:, 1:] | hole[:, :-1], 0.0, np.maximum(z[:, 1:], z[:, :-1]) / np.minimum(z[:, 1:], z[:, :-1]) - 1)
    rv = np.where(hole[1:, :] | hole[:-1, :], 0.0, np.maximum(z[1:, :], z[:-1, :]) / np.minimum(z[1:, :], z[:-1, :]) - 1)
    step[:, 1:] = np.maximum(step[:, 1:], rh)
    step[:, :-1] = np.maximum(step[:, :-1], rh)
    step[1:, :] = np.maximum(step[1:, :], rv)
    step[:-1, :] = np.maximum(step[:-1, :], rv)
    border &= step < 0.5 * thr
    if fg.any() and border.any():
        dist, (iy, ix) = ndimage.distance_transform_edt(~fg, return_indices=True)
        z_fg = disp_to_z(np.maximum(d[iy, ix], SKY_DISP))
        behind = z >= z_fg * (1.0 + 2.0 * thr)
        below = iy < (np.arange(d.shape[0])[:, None] - 2)
        border &= (dist > 2 * r_fg + 2) | behind | below
    return border

def _harmonic_fill(values: np.ndarray, unknown: np.ndarray, dirichlet: np.ndarray) -> np.ndarray:
    """Laplace (membrane) interpolation of ``values`` into ``unknown`` pixels.

    Only ``dirichlet`` pixels next to the unknown region constrain the
    solution; every other neighbour (foreground, near surfaces, the image
    border) is a natural zero-flux boundary, so nothing leaks in from the
    foreground. Unknown regions without any data keep their values.
    """
    out = values.astype(np.float32).copy()
    unknown = unknown.astype(bool)
    dirichlet = dirichlet.astype(bool) & ~unknown
    if not unknown.any() or not dirichlet.any():
        return out
    lab, nl = ndimage.label(unknown)
    touch = _dil(dirichlet, 1) & unknown
    has = np.zeros(nl + 1, bool)
    has[np.unique(lab[touch])] = True
    has[0] = False
    solve = has[lab]
    ys, xs = np.nonzero(solve)
    n = len(ys)
    if n == 0:
        return out
    h, w = unknown.shape
    idx = -np.ones((h, w), np.int64)
    idx[ys, xs] = np.arange(n)
    diag = np.zeros(n)
    b = np.zeros(n)
    rows, cols = [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inb = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nyc, nxc = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
        is_u = inb & solve[nyc, nxc]
        is_d = inb & dirichlet[nyc, nxc]
        diag += is_u | is_d
        r_ = np.nonzero(is_u)[0]
        rows.append(r_)
        cols.append(idx[nyc[r_], nxc[r_]])
        b[is_d] += values[nyc[is_d], nxc[is_d]]
    rows_a, cols_a = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((np.full(len(rows_a), -1.0), (rows_a, cols_a)), shape=(n, n)) + sp.diags(diag + 1e-9)
    # initial guess: smooth extension of the boundary data (speeds up convergence)
    x0 = normalized_blur(values.astype(np.float32), dirichlet.astype(np.float32), max(3.0, 0.05 * np.sqrt(n)))[ys, xs]
    try:
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(A.tocsr(), symmetry="symmetric")
        x = ml.solve(b, x0=x0.astype(np.float64), tol=1e-7, maxiter=200, accel="cg")
    except Exception:  # noqa: BLE001 - pyamg missing or failed: direct sparse solve
        x = sp.linalg.spsolve(A.tocsc(), b)
    out[ys, xs] = x.astype(np.float32)
    return out


HIDDEN_DEPTH_LOG: list = []  # diagnostics of the last _hidden_depth calls (planes found, share of the hole continued planarly)


def _hidden_depth(values: np.ndarray, unknown: np.ndarray, dirichlet: np.ndarray, sky: np.ndarray | None = None,
                  min_support: float = 0.04, tol: float = 0.06) -> np.ndarray:
    """Depth of the hidden background inside ``unknown`` (drop-in for :func:`_harmonic_fill`).

    A membrane bends floors and walls behind a large foreground object into a
    bulge. Wherever the hole's trustworthy border (``dirichlet``) lies on large
    planes of the visible background (floor, walls, ground - D1's
    :func:`~.geometry.fit_planes`), those planes are continued behind the
    foreground instead: the visible surface is the *nearest* of the planes
    (exact for the inside of a room: floor up to the floor-wall line, then the
    wall), plus a harmonic correction of the residual so the result meets the
    border exactly (no step, no streak). A hole component whose border the
    planes do not explain (< 65 % of its border pixels within ``tol`` relative
    depth) keeps the membrane.
    """
    base = _harmonic_fill(values, unknown, dirichlet)
    unknown = unknown.astype(bool)
    dirichlet = dirichlet.astype(bool) & ~unknown
    if not unknown.any() or dirichlet.sum() < 40:
        return base
    try:
        from .geometry import fit_planes, plane_disparity
    except ImportError:  # plane helpers not available: membrane only
        return base
    h, w = values.shape
    vis = ~_dil(unknown, 2)
    if sky is not None:
        vis &= ~sky
    try:
        planes = [p for p in fit_planes(values, mask=vis) if p.support >= min_support]
    except Exception:  # noqa: BLE001 - plane fitting is a refinement only
        return base
    if not planes:
        return base
    pd = [plane_disparity(p, (h, w)) for p in planes]
    pm = [_dil(p.mask, 3) for p in planes]
    zv = disp_to_z(np.maximum(values, SKY_DISP))
    lab, n = ndimage.label(unknown)
    D = np.zeros((h, w), np.float32)
    use = np.zeros((h, w), bool)
    ring_all = np.zeros((h, w), bool)
    sl = ndimage.find_objects(lab)
    for k in range(1, n + 1):
        ys, xs = sl[k - 1]
        pad = 4
        y0, y1, x0, x1 = max(0, ys.start - pad), min(h, ys.stop + pad), max(0, xs.start - pad), min(w, xs.stop + pad)
        comp = lab[y0:y1, x0:x1] == k
        ring = _dil(comp, 1) & dirichlet[y0:y1, x0:x1]
        nr = int(ring.sum())
        if nr < 30:
            continue
        cand = [i for i in range(len(planes)) if (ring & pm[i][y0:y1, x0:x1]).sum() >= max(15, 0.15 * nr)]
        if not cand:
            continue
        stack = np.stack([pd[i][y0:y1, x0:x1] for i in cand])
        Dk = stack.max(0)
        if not np.isfinite(Dk).all() or (Dk[comp] <= SKY_DISP).any():
            continue  # a plane recedes beyond the horizon inside the hole: not a closed room/ground
        zr = disp_to_z(np.maximum(Dk[ring], SKY_DISP))
        ok = np.abs(zv[y0:y1, x0:x1][ring] / zr - 1.0) < tol
        if ok.mean() < 0.65:
            continue
        D[y0:y1, x0:x1][comp] = Dk[comp]
        D[y0:y1, x0:x1][ring] = Dk[ring]
        use[y0:y1, x0:x1] |= comp
        ring_all[y0:y1, x0:x1] |= ring
    HIDDEN_DEPTH_LOG.append({"planes": len(planes), "components": int(n), "planar_share": round(float(use.sum()) / max(1, int(unknown.sum())), 3)})
    del HIDDEN_DEPTH_LOG[:-4]
    if not use.any():
        return base
    resid = np.where(ring_all, values - D, 0.0).astype(np.float32)
    corr = _harmonic_fill(resid, use, ring_all)
    out = base.copy()
    out[use] = (D + corr)[use]
    return out
