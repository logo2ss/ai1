"""Depth post-processing and mesh geometry for the 3D reconstruction: disparity/depth
conversion, sky detection, depth-edge sharpening, discontinuity cuts and occlusion sides."""

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

SKY_DISP = -0.08  # stored disparity for sky (=> far dome in the shader)
REF_ZFAR = 10.0  # reference depth range used for cut decisions


def disp_to_z(d: np.ndarray, zfar: float = REF_ZFAR, znear: float = 1.0) -> np.ndarray:
    return 1.0 / (d * (1.0 / znear - 1.0 / zfar) + 1.0 / zfar)

def z_to_disp(z: np.ndarray, zfar: float = REF_ZFAR, znear: float = 1.0) -> np.ndarray:
    return (1.0 / z - 1.0 / zfar) / (1.0 / znear - 1.0 / zfar)

def _behind(d: np.ndarray, k: float) -> np.ndarray:
    """Disparity of a surface ``k`` (relative) farther away than disparity ``d``."""
    return np.maximum(z_to_disp(disp_to_z(np.maximum(d, SKY_DISP)) * (1.0 + k)), SKY_DISP * 1.5)

def detect_sky(rgb: np.ndarray, disp: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
    """Far, smooth, bright region connected to the top border.

    ``hint``: optional map of where the depth network's raw output is zero (see
    ``estimate_depth(info=...)``) - the network's own sky signal. With it, sky that the
    edge-aware filters lifted slightly off zero (between branches, behind girders) is found,
    and sky seen through foliage counts even when it does not touch the top border; walls and
    ceilings never have a zero raw output, so white walls are not mistaken for sky. Without
    it, a slightly relaxed depth test applies to clearly blue pixels only.
    """
    h, w = disp.shape
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    tex = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (9, 9))
    far = disp < 0.03
    blue = (rgb[..., 2] > rgb[..., 0] + 0.04) & (rgb[..., 2] >= rgb[..., 1] - 0.02) & (g > 0.3)
    far |= (disp < 0.07) & blue
    hz = None
    if hint is not None and hint.shape == disp.shape:
        hz = hint > 0.5
        far |= hz & (disp < 0.06)
    cand = far & (tex < 0.06) & (g > 0.25)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    top = set(np.unique(lab[0, :])) - {0}
    if hz is None:
        sky = np.isin(lab, list(top))
    else:
        # a sky component must be (largely) where the network itself sees sky; it touches the top
        # border or is a large piece in the upper part of the frame (sky behind branches)
        # (blurred far backgrounds can be zero too, so pieces that do not touch the top border must
        # also be sky-blue: blue sky between branches, not a defocused wall)
        zf = ndimage.mean(hz.astype(np.float32), lab, np.arange(n)) if n > 1 else np.zeros(1)
        bf = ndimage.mean(blue.astype(np.float32), lab, np.arange(n)) if n > 1 else np.zeros(1)
        ok = np.zeros(n, bool)
        for i in range(1, n):
            big_up = stats[i, cv2.CC_STAT_AREA] >= 0.005 * h * w and cent[i][1] < 0.5 * h and bf[i] > 0.6
            ok[i] = (i in top or big_up) and zf[i] >= 0.3
        sky = ok[lab]
    if hz is not None and sky.mean() >= 0.005:
        # sky seen through foliage / structures: blue zero-output pieces with the colour of the main sky
        mc = rgb[sky].mean(0)
        sd = float(np.sqrt(((rgb[sky] - mc) ** 2).sum(-1).mean()))
        extra = cand & ~sky & hz & blue & (np.sqrt(((rgb - mc) ** 2).sum(-1)) < max(0.12, 2.0 * sd)) & (np.arange(h)[:, None] < 0.75 * h)
        sky |= extra
    if sky.mean() < 0.01:
        return np.zeros_like(sky)
    sky = (sky | ndimage.binary_closing(sky, iterations=2)) & far
    # mixed pixels along the skyline (leaves, branches, roof edges) that are coloured like the
    # adjacent sky go with the sky; left on the object they would float with it as a bright halo
    rr = max(2, int(round(0.004 * max(h, w))))
    ring = _dil(sky, rr) & ~sky
    if ring.any():
        local = normalized_blur(rgb.astype(np.float32), sky.astype(np.float32), 3.0 * rr)
        cd = np.sqrt(((rgb - local) ** 2).sum(-1))
        sky |= ring & (cd < 0.07) & (g > 0.3) & (tex < 0.12)
    return sky


# --------------------------------------------------------------------------- planar structure
#
# The viewer unprojects a vertex with 1/z = d * (1 - 1/zfar) + 1/zfar along the ray of its pixel,
# so inverse depth is affine in the stored disparity d. A 3D plane has an inverse depth that is
# affine in the image coordinates, hence: a surface is planar in 3D  <=>  its disparity is an
# affine function d = a*x + b*y + c of the image coordinates - for every field of view and depth
# range the viewer may use. Planes are therefore fitted and evaluated directly in (x, y, d).


@dataclass
class Plane:
    """A dominant planar surface (floor, wall, ceiling, table top ...).

    Disparity on the plane: ``d = a*x + b*y + c`` with *normalised frame coordinates*
    ``x = 2*col/(w-1) - 1`` and ``y = 2*row/(h-1) - 1`` (-1 at the left/top vertex of the frame,
    +1 at the right/bottom one - the viewer's convention). Outside the frame the same formula
    continues the plane (use :func:`plane_disparity` with ``extent``).
    """

    a: float
    b: float
    c: float
    kind: str = "other"  # floor | ceiling | wall | other  (from the 3D normal, see plane_normal)
    normal: tuple = (0.0, 0.0, 1.0)  # unit normal in camera space (x right, y up, z towards the viewer)
    support: float = 0.0  # fraction of the image covered by the plane's region
    rms: float = 0.0  # rms residual of the region, as a fraction of depth
    mask: np.ndarray | None = None  # region of the plane (bool, resolution of the fitted disparity)

    def disparity(self, shape: tuple[int, int], extent: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        return plane_disparity(self, shape, extent)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c, "kind": self.kind, "normal": [round(float(v), 4) for v in self.normal],
                "support": round(float(self.support), 4), "rms": round(float(self.rms), 5)}


def norm_coords(h: int, w: int, extent: tuple[float, float] = (1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Normalised coordinates of the vertex columns (x, length w) and rows (y, length h)."""
    xs = (2.0 * np.arange(w, dtype=np.float64) / max(w - 1, 1) - 1.0) * extent[0]
    ys = (2.0 * np.arange(h, dtype=np.float64) / max(h - 1, 1) - 1.0) * extent[1]
    return xs.astype(np.float32), ys.astype(np.float32)


def plane_disparity(plane: "Plane", shape: tuple[int, int], extent: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Disparity of ``plane`` on a grid of ``shape`` = (h, w) covering the frame.

    ``extent`` = (ex, ey) makes the grid cover [-ex, ex] x [-ey, ey] instead of [-1, 1]^2: for a
    grid that extends a w x h frame by m pixels on every side pass ((w + 2m) / w, (h + 2m) / h)
    (the scene container's ``extent_x`` / ``extent_y``) - the plane then continues beyond the
    frame exactly as the viewer would unproject it.
    """
    h, w = shape
    xs, ys = norm_coords(h, w, extent)
    return (plane.a * xs[None, :] + plane.b * ys[:, None] + plane.c).astype(np.float32)


def plane_normal(a: float, b: float, c: float, fov: float = 60.0, zfar: float = REF_ZFAR, aspect: float = 1.5) -> tuple[np.ndarray, float]:
    """Unit normal (camera space: x right, y up, z towards the viewer; pointing from the camera
    to the plane) and camera distance of the plane d = a x + b y + c, for the viewer's camera
    (horizontal ``fov`` in degrees, depth range ``zfar``, frame ``aspect`` = w / h)."""
    t = np.tan(np.radians(fov) / 2.0)
    k, q = 1.0 - 1.0 / zfar, 1.0 / zfar
    # 1/z = k (a x + b y + c) + q  with  x = u / t, y = -v * aspect / t  for the ray (u, v, -1)
    n = np.array([k * a / t, -k * b * aspect / t, -(k * c + q)], np.float64)
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return np.array([0.0, 0.0, 1.0]), float("inf")
    return n / ln, 1.0 / ln


def _plane_kind(nrm: np.ndarray) -> str:
    ny = float(nrm[1])
    if ny < -np.cos(np.radians(35)):
        return "floor"
    if ny > np.cos(np.radians(35)):
        return "ceiling"
    if abs(ny) < np.sin(np.radians(25)):
        return "wall"
    return "other"


def _disp_tol(d: np.ndarray, rel: float, zfar: float = REF_ZFAR) -> np.ndarray:
    """Disparity change that moves a surface by ``rel`` of its depth (viewer model)."""
    q_k = (1.0 / zfar) / (1.0 - 1.0 / zfar)
    return rel * (np.maximum(d, SKY_DISP) + q_k)


def _local_plane_stats(d: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares plane over the (2r+1)^2 window of every pixel: slopes (per pixel) gx, gy,
    window mean, and rms residual of the fit."""
    d = d.astype(np.float32)
    k = 2 * r + 1
    ax = np.arange(-r, r + 1, dtype=np.float32)
    kx = np.tile(ax, (k, 1)) / float(k * k)
    var = float((ax**2).mean())
    mean = cv2.blur(d, (k, k), borderType=cv2.BORDER_REFLECT)
    mxd = cv2.filter2D(d, -1, kx, borderType=cv2.BORDER_REFLECT)
    myd = cv2.filter2D(d, -1, np.ascontiguousarray(kx.T), borderType=cv2.BORDER_REFLECT)
    mdd = cv2.blur(d * d, (k, k), borderType=cv2.BORDER_REFLECT)
    gx, gy = mxd / var, myd / var
    res2 = mdd - mean * mean - (gx * gx + gy * gy) * var
    return gx, gy, mean, np.sqrt(np.maximum(res2, 0.0))


def planar_slopes(d: np.ndarray, r: int, rel: float = 0.06) -> np.ndarray:
    """Pixels on smooth *linear* slopes (floors, walls seen at an angle): the disparity is
    affine over a (2r+1)^2 window up to ``rel`` of the change across the window. A blurred
    depth step narrower than the window leaves a much larger residual (>= 0.12)."""
    gx, gy, _, res = _local_plane_stats(d, max(2, int(r)))
    g = np.hypot(gx, gy)
    return res < rel * g * max(2, int(r)) + 2e-4


def fit_planes(disp: np.ndarray, mask: np.ndarray | None = None, fov: float = 60.0, zfar: float = REF_ZFAR,
               max_planes: int = 6, min_support: float = 0.03, tol: float = 0.045, seed: int = 0) -> list[Plane]:
    """Detect dominant planes in a disparity map (RANSAC in (x, y, d), see the note above).

    ``mask``: pixels that may belong to a plane (e.g. ``~sky`` or ``~(sky | people)``).
    ``tol``: inlier distance as a fraction of depth. Inliers must also agree with the plane's
    slope locally, and a plane's region is made of connected inlier components only, so two
    surfaces crossing each other or parallel surfaces at different depths are not merged.
    Returns planes sorted by support, each with its region mask at the input resolution.
    """
    h, w = disp.shape
    valid = np.ones((h, w), bool) if mask is None else mask.astype(bool)
    valid = valid & np.isfinite(disp) & (disp > SKY_DISP + 1e-3)
    s = min(1.0, 200.0 / max(h, w))
    sh, sw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    ds = cv2.resize(np.nan_to_num(disp, nan=0.0).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.99
    vs = _ero(vs, 1)
    xs, ys = norm_coords(sh, sw)
    X = np.broadcast_to(xs[None, :], (sh, sw))
    Y = np.broadcast_to(ys[:, None], (sh, sw))
    r = max(2, int(round(0.025 * max(sh, sw))))
    gx, gy, _, res = _local_plane_stats(ds, r)
    # slopes per normalised unit
    gxn, gyn = gx * (sw - 1) / 2.0, gy * (sh - 1) / 2.0
    tl = _disp_tol(ds, tol, zfar)
    cand = vs & (res < 0.35 * tl)
    n_all = float(sh * sw)
    rng = np.random.default_rng(seed)
    planes: list[Plane] = []
    free = cand.copy()
    for _ in range(max_planes):
        idx = np.flatnonzero(free)
        if idx.size < min_support * n_all:
            break
        best, best_n = None, 0
        for i in rng.choice(idx, size=min(160, idx.size), replace=False):
            a, b = gxn.flat[i], gyn.flat[i]
            c = ds.flat[i] - a * X.flat[i] - b * Y.flat[i]
            inl = _plane_inliers(ds, X, Y, a, b, c, gxn, gyn, tl, free)
            nin = int(inl.sum())
            if nin > best_n:
                best, best_n = (a, b, c), nin
        if best is None or best_n < min_support * n_all:
            break
        a, b, c = best
        for _it in range(3):  # least-squares refinement on the inliers
            inl = _plane_inliers(ds, X, Y, a, b, c, gxn, gyn, tl, free)
            if inl.sum() < 10:
                break
            A = np.stack([X[inl], Y[inl], np.ones(int(inl.sum()), np.float32)], 1).astype(np.float64)
            rr = (ds[inl] - (a * X[inl] + b * Y[inl] + c)) / np.maximum(tl[inl] / 3.0, 1e-6)
            sw_ = np.sqrt(1.0 / (1.0 + rr.astype(np.float64) ** 2))  # robust (Cauchy) weights: bumps do not tilt the plane
            (a, b, c), *_ = np.linalg.lstsq(A * sw_[:, None], ds[inl].astype(np.float64) * sw_, rcond=None)
        inl = _plane_inliers(ds, X, Y, a, b, c, gxn, gyn, tl, free)
        # region: connected inlier components that are large enough
        reg = cv2.morphologyEx(inl.astype(np.uint8), cv2.MORPH_OPEN, _disk(1)) > 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(reg.astype(np.uint8), connectivity=8)
        okc = np.zeros(n, bool)
        okc[1:] = stats[1:, cv2.CC_STAT_AREA] >= 0.4 * min_support * n_all
        reg = okc[lab]
        if reg.sum() < min_support * n_all:
            free &= ~inl  # do not draw the same hypothesis again
            continue
        # grow into the margin that the local (windowed) tests miss next to occluders and pockets
        near_pl = vs & (np.abs(ds - (a * X + b * Y + c)) < tl)
        reg = reg | (_dil(reg, r + 1) & near_pl)
        reg = _fill_small_holes(reg, 0.002 * n_all) & vs
        dist = np.abs(ds - (a * X + b * Y + c)) / np.maximum(ds + (1.0 / zfar) / (1.0 - 1.0 / zfar), 1e-6)
        nrm, _ = plane_normal(float(a), float(b), float(c), fov, zfar, w / h)
        full = cv2.resize(reg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        planes.append(Plane(float(a), float(b), float(c), _plane_kind(nrm), tuple(float(v) for v in nrm), float(reg.mean()),
                            float(np.sqrt(np.mean(dist[reg] ** 2))), full & valid))
        free &= ~_dil(reg, 1)
    planes.sort(key=lambda p: -p.support)
    return planes


def _plane_inliers(ds, X, Y, a, b, c, gxn, gyn, tl, free) -> np.ndarray:
    pd = a * X + b * Y + c
    gtol = 0.25 * np.hypot(a, b) + 0.06
    return free & (np.abs(ds - pd) < tl) & (np.abs(gxn - a) + np.abs(gyn - b) < gtol)


def flatten_planes(disp: np.ndarray, planes: list[Plane], strength: float = 1.0, tol: float = 0.06,
                   feather: float | None = None, zfar: float = REF_ZFAR) -> np.ndarray:
    """Regularise the disparity inside confident plane regions towards the fitted planes, so
    floors and walls stay flat (do not bend or bulge) when the camera moves.

    Only pixels within ``tol`` (fraction of depth) of the plane are changed; the blend weight
    ramps up from the region border over ``feather`` pixels, so no step is created.
    """
    if not planes:
        return disp.astype(np.float32).copy()
    h, w = disp.shape
    out = disp.astype(np.float32).copy()
    fe = feather if feather is not None else max(3.0, 0.02 * max(h, w))
    for p in planes:
        if p.mask is None or not p.mask.any():
            continue
        m = p.mask if p.mask.shape == (h, w) else cv2.resize(p.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        pd = plane_disparity(p, (h, w))
        tl = _disp_tol(out, tol, zfar)
        dev = np.abs(out - pd)
        close = dev < tl
        soft = np.clip((tl - dev) / (0.5 * tl), 0.0, 1.0)  # fades out towards the tolerance: no new steps
        # small pockets enclosed by the plane that look *farther* than it are reflections or
        # see-through noise (glass tables, polished floors): a solid surface has no holes
        m = m | (_dil(m, max(2, int(round(0.012 * max(h, w))))) & close)
        # Enclosed far pockets in a well-supported floor are frequently
        # reflections. Keep foreground bumps and wall openings untouched.
        pocket_limit = (0.04 if p.kind == "floor" and p.support >= 0.08 else 0.006) * h * w
        filled = _fill_small_holes(m, pocket_limit)
        pocket = filled & ~(m & close) & (out < pd)
        if pocket.any():
            n, lab, stats, _ = cv2.connectedComponentsWithStats(pocket.astype(np.uint8), connectivity=4)
            small = np.zeros(n, bool)
            small[1:] = stats[1:, cv2.CC_STAT_AREA] <= pocket_limit
            pocket = small[lab]
        tgt = (m & close) | pocket
        if not tgt.any():
            continue
        dist = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 3)
        wgt = np.where(tgt, np.clip(dist / fe, 0.0, 1.0) * strength * np.where(pocket, 1.0, soft), 0.0)
        out = out + wgt.astype(np.float32) * (pd - out)
    return out.astype(np.float32)


def _cuts(d: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal/vertical depth-jump flags between neighbours, using z ratios."""
    z = disp_to_z(np.maximum(d, SKY_DISP), REF_ZFAR)
    ch = np.maximum(z[:, 1:], z[:, :-1]) / np.minimum(z[:, 1:], z[:, :-1]) - 1 > thr
    cv_ = np.maximum(z[1:, :], z[:-1, :]) / np.minimum(z[1:, :], z[:-1, :]) - 1 > thr
    # wider check (2 px) catches ramps of mixed edge pixels
    ch2 = np.zeros_like(ch)
    ch2[:, :-1] = np.maximum(z[:, 2:], z[:, :-2]) / np.minimum(z[:, 2:], z[:, :-2]) - 1 > thr * 2.2
    cv2_ = np.zeros_like(cv_)
    cv2_[:-1, :] = np.maximum(z[2:, :], z[:-2, :]) / np.minimum(z[2:, :], z[:-2, :]) - 1 > thr * 2.2
    edge_px = np.zeros(d.shape, bool)
    for c, sl_a, sl_b in ((ch | ch2, (slice(None), slice(1, None)), (slice(None), slice(None, -1))), (cv_ | cv2_, (slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        edge_px[sl_a] |= c
        edge_px[sl_b] |= c
    return ch | ch2, cv_ | cv2_, edge_px

def _quad_keep(d: np.ndarray, valid: np.ndarray, thr: float, cuts: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    ch, cv_, _ = _cuts(d, thr) if cuts is None else (*cuts, None)
    keep = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    keep &= ~ch[:-1, :] & ~ch[1:, :] & ~cv_[:, :-1] & ~cv_[:, 1:]
    return keep

def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def _dil(m: np.ndarray, r: int) -> np.ndarray:
    return m.copy() if r <= 0 else cv2.dilate(m.astype(np.uint8), _disk(r)) > 0

def _ero(m: np.ndarray, r: int) -> np.ndarray:
    return m.copy() if r <= 0 else cv2.erode(m.astype(np.uint8), _disk(r)) > 0

def _fill_small_holes(m: np.ndarray, max_area: float) -> np.ndarray:
    holes = ndimage.binary_fill_holes(m) & ~m
    if not holes.any():
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    return m | small[lab]

def _sharpen_depth_edges(rgb: np.ndarray, d: np.ndarray, thr: float, sky: np.ndarray | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Turn blurred depth ramps at object borders into steps that follow the photo's edges,
    then regularise the dominant planes.

    Monocular depth smears a discontinuity over several pixels; the ramp would
    become a stretched "rubber sheet" when the camera moves. Inside transition
    zones (the local depth range spans a real jump) a colour-weighted median
    with a wide support snaps every pixel to the side it belongs to. Broad
    linear slopes - floors, lawns, roads, walls seen at an angle, whose
    disparity is affine over a whole window - are excluded: a median guided by
    their texture would break them into blotchy steps that the mesh then tears.
    Finally floors, walls and table tops found by :func:`fit_planes` are
    flattened (:func:`regularize_planes`), so they stay flat in novel views and
    depth noise on reflective surfaces does not tear them.
    """
    out = d.astype(np.float32)
    h, w = d.shape
    if hasattr(cv2, "ximgproc"):
        r = max(3, int(round(0.01 * max(h, w))))
        k = _disk(r)
        z = disp_to_z(np.maximum(d, SKY_DISP))
        zone = (cv2.dilate(z, k) / np.maximum(cv2.erode(z, k), 1e-6) - 1) > 1.5 * thr
        if zone.any():
            zone &= ~_broad_slopes(d, 2 * r)
        if zone.any():
            u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            try:
                wm = cv2.ximgproc.weightedMedianFilter(u8, d.astype(np.float32), r, 12.0)
                out = np.where(zone, wm, out).astype(np.float32)
            except cv2.error:
                pass
    regular = regularize_planes(rgb, out, sky=sky, mask=mask)
    return regular if mask is None else np.where(mask, regular, d).astype(np.float32)


def _broad_slopes(d: np.ndarray, r: int) -> np.ndarray:
    """Planar slopes that extend over a broad area (floors, walls). The centre of a wide
    blurred step is locally linear too, but only on a thin band, which the opening removes."""
    p = planar_slopes(d, r)
    return cv2.morphologyEx(p.astype(np.uint8), cv2.MORPH_OPEN, _disk(max(1, r // 2))) > 0


def regularize_planes(rgb: np.ndarray, d: np.ndarray, sky: np.ndarray | None = None, mask: np.ndarray | None = None,
                      planes_out: list | None = None) -> np.ndarray:
    """Fit the dominant planes of ``d`` (outside the sky / outside ``mask`` = False) and flatten
    them (see :func:`fit_planes`, :func:`flatten_planes`). The planes are appended to
    ``planes_out`` when given."""
    if sky is None:
        sky = detect_sky(rgb, d)
    valid = ~sky if mask is None else (mask.astype(bool) & ~sky)
    try:
        planes = fit_planes(d, valid)
    except Exception:  # noqa: BLE001 - a refinement only, never fatal
        return d.astype(np.float32)
    if planes_out is not None:
        planes_out.extend(planes)
    return flatten_planes(d, planes)


def _flip_small(mask: np.ndarray, zone: np.ndarray, min_area: int) -> np.ndarray:
    """Flip connected pieces of ``mask & zone`` (and of ``~mask & zone``) smaller than ``min_area``."""
    out = mask.copy()
    for val in (True, False):
        part = ((out == val) & zone).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=4)
        if n <= 1:
            continue
        small = np.zeros(n, bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < min_area
        out[small[lab]] = not val
    return out

def _snap_ramps(rgb: np.ndarray, d: np.ndarray, thr: float, sky: np.ndarray | None = None) -> np.ndarray:
    """Resolve wide depth ramps at object borders into clean steps.

    Monocular networks blur a discontinuity over tens of pixels. Such a ramp
    is far steeper in *disparity* than any real surface (a floor changes its
    disparity linearly and slowly), so ramps are found by their disparity
    gradient, restricted to places where the neighbourhood spans a real depth
    jump. Every pixel of the transition zone then joins the near or the far
    side - the side whose colour it resembles more, its position in the ramp
    breaking ties; tiny islands are merged - and takes the depth of the
    closest pixel of that side, so no strip of intermediate depth survives.
    Floors and walls are left untouched.
    """
    h, w = d.shape
    L = max(h, w)
    r = max(3, int(round(0.02 * L)))
    gy, gx = np.gradient(cv2.GaussianBlur(d.astype(np.float32), (0, 0), 1.0))
    core = np.hypot(gx, gy) > 0.008 * 900.0 / L
    if sky is not None:
        core &= ~sky
    if core.any():
        # steep but linear slopes (near ground, walls at grazing angles) are surfaces, not edges
        core &= ~_broad_slopes(d, r)
    if not core.any():
        return d
    k = _disk(r)
    dmax, dmin = cv2.dilate(d, k), cv2.erode(d, k)
    span = disp_to_z(np.maximum(dmin, SKY_DISP)) / disp_to_z(np.maximum(dmax, SKY_DISP)) - 1
    zone = _dil(core, max(2, r // 2)) & (span > 2.0 * thr)
    if not zone.any():
        return d
    mid = 0.5 * (dmax + dmin)
    near_p = ~zone & (d > mid)
    far_p = ~zone & (d <= mid)
    if not near_p.any() or not far_p.any():
        return d
    sig = max(2.0, r / 2.0)
    c_near = normalized_blur(rgb.astype(np.float32), near_p.astype(np.float32), sig)
    c_far = normalized_blur(rgb.astype(np.float32), far_p.astype(np.float32), sig)
    dn = np.sqrt(((rgb - c_near) ** 2).sum(-1))
    df = np.sqrt(((rgb - c_far) ** 2).sum(-1))
    t = (d - dmin) / np.maximum(dmax - dmin, 1e-6)
    score = gaussian(((df - dn) + 0.25 * (t - 0.5)).astype(np.float32), 1.0)
    near = _flip_small(score > 0, zone, max(4, (r // 2) ** 2))
    dist_n, (ny, nx) = ndimage.distance_transform_edt(~near_p, return_indices=True)
    dist_f, (fy, fx) = ndimage.distance_transform_edt(~far_p, return_indices=True)
    v_near = np.where(dist_n <= 2 * r, d[ny, nx], dmax)
    v_far = np.where(dist_f <= 2 * r, d[fy, fx], dmin)
    out = np.where(zone, np.where(near, v_near, v_far), d).astype(np.float32)
    return _despeckle(out, thr, None if sky is None else ~sky)

def _cut_components(d: np.ndarray, thr: float) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected pieces of the mesh that the depth cuts would produce: (n, labels, areas)."""
    h, w = d.shape
    ch, cv_, _ = _cuts(d, thr)
    g = np.zeros((2 * h - 1, 2 * w - 1), np.uint8)
    g[::2, ::2] = 1
    g[::2, 1::2] = ~ch
    g[1::2, ::2] = ~cv_
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=4)
    lab = lab[::2, ::2]
    area = np.bincount(lab.ravel(), minlength=n)
    return n, lab, area


def _despeckle(d: np.ndarray, thr: float, mask: np.ndarray | None = None, max_area: float | None = None) -> np.ndarray:
    """Remove "flying pixels": tiny mesh islands (cut off on all sides) whose depth lies between
    the depths around them - mixed pixels on object borders that would float in mid-air when the
    camera moves. Each such pixel takes the depth of the nearest pixel outside the island, so it
    joins the surface it touches. Islands nearer than all their surroundings (a bird, a pole
    segment) and farther ones (sky seen through leaves) are real and kept. ``mask``: pixels that
    may change."""
    h, w = d.shape
    ma = max_area if max_area is not None else max(12.0, 0.0002 * h * w)
    n, lab, area = _cut_components(d, thr)
    small = area <= ma
    small[0] = False
    isl = small[lab]
    if mask is not None:
        isl &= mask
    if not isl.any():
        return d
    ids = np.unique(lab[isl])
    z = disp_to_z(np.maximum(d, SKY_DISP))
    ring = _dil(isl, 1) & ~isl
    # ring pixels inherit the id of the island they surround (max id among their island neighbours)
    rl = ndimage.grey_dilation(np.where(isl, lab, 0), size=(3, 3))
    rl = np.where(ring, rl, 0)
    zmin = ndimage.minimum(z, rl, ids)
    zmax = ndimage.maximum(z, rl, ids)
    zmed = ndimage.median(z, lab, ids)
    ok = np.isfinite(zmin) & np.isfinite(zmax)
    inter = ok & (zmed > zmin * (1 + 0.5 * thr)) & (zmed < zmax / (1 + 0.5 * thr))
    if not inter.any():
        return d
    sel = np.zeros(n, bool)
    sel[ids[inter]] = True
    fix = sel[lab] & isl
    _, (iy, ix) = ndimage.distance_transform_edt(fix, return_indices=True)
    return np.where(fix, d[iy, ix], d).astype(np.float32)


def _snap_inside(d: np.ndarray, fg: np.ndarray, r: int) -> np.ndarray:
    """Depth just inside a silhouette takes the nearer interior value (removes the ramp
    towards the background that monocular depth has at object borders)."""
    out = d.astype(np.float32).copy()
    labels, n = ndimage.label(fg, structure=np.ones((3, 3)))
    # Erosion can delete a thin object's entire core. It must not then borrow
    # depth from an unrelated, thicker person elsewhere in the photograph.
    for k, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        comp = labels[slices] == k
        # Pad before erosion so a component bounding box is not treated as solid.
        padded = np.pad(comp, r+1)
        core = _ero(padded, r)[r+1:-r-1, r+1:-r-1]
        if not core.any():
            continue
        _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
        local = out[slices]
        ring = comp & ~core
        local[ring] = np.maximum(local[ring], local[iy[ring], ix[ring]])
    return out


def _near_side(d: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Near-side pixels of every depth cut and the disparity of the surface hidden behind them."""
    h, w = d.shape
    ch, cv_, _ = _cuts(d, thr)
    z = disp_to_z(np.maximum(d, SKY_DISP), REF_ZFAR)
    near = np.zeros((h, w), bool)
    far_d = np.full((h, w), np.inf, np.float32)
    dd = d.astype(np.float32)
    hl = z[:, :-1] < z[:, 1:]  # left pixel nearer
    for cond, nsl, fsl in (
        (ch & hl, (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        (ch & ~hl, (slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ):
        near[nsl] |= cond
        far_d[nsl] = np.where(cond, np.minimum(far_d[nsl], dd[fsl]), far_d[nsl])
    vl = z[:-1, :] < z[1:, :]
    for cond, nsl, fsl in (
        (cv_ & vl, (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        (cv_ & ~vl, (slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ):
        near[nsl] |= cond
        far_d[nsl] = np.where(cond, np.minimum(far_d[nsl], dd[fsl]), far_d[nsl])
    return near, far_d

def _cut_sides(d: np.ndarray, ch: np.ndarray, cv_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sides of the tears actually made in a mesh (cut flags ``ch``: H x W-1, ``cv_``: H-1 x W).

    Returns (near, far, far_d, thin): pixels on the near / far side of a cut
    (a pixel that is near for one neighbour and far for another counts as
    near), the farthest disparity among each near pixel's far neighbours, and
    near pixels cut on two opposite sides (parts only one pixel thin, which
    the mesh cannot keep).
    """
    h, w = d.shape
    z = disp_to_z(np.maximum(d, SKY_DISP))
    dd = d.astype(np.float32)
    near = np.zeros((h, w), bool)
    far = np.zeros((h, w), bool)
    far_d = np.full((h, w), np.inf, np.float32)
    thin = np.zeros((h, w), bool)
    a_near = ch & (z[:, :-1] < z[:, 1:])  # edge (x, x+1): x is the near side
    b_near = ch & ~(z[:, :-1] < z[:, 1:])  # x+1 is the near side
    near[:, :-1] |= a_near
    far[:, 1:] |= a_near
    far_d[:, :-1] = np.where(a_near, np.minimum(far_d[:, :-1], dd[:, 1:]), far_d[:, :-1])
    near[:, 1:] |= b_near
    far[:, :-1] |= b_near
    far_d[:, 1:] = np.where(b_near, np.minimum(far_d[:, 1:], dd[:, :-1]), far_d[:, 1:])
    thin[:, 1:-1] |= a_near[:, 1:] & b_near[:, :-1]
    a_near = cv_ & (z[:-1, :] < z[1:, :])
    b_near = cv_ & ~(z[:-1, :] < z[1:, :])
    near[:-1, :] |= a_near
    far[1:, :] |= a_near
    far_d[:-1, :] = np.where(a_near, np.minimum(far_d[:-1, :], dd[1:, :]), far_d[:-1, :])
    near[1:, :] |= b_near
    far[:-1, :] |= b_near
    far_d[1:, :] = np.where(b_near, np.minimum(far_d[1:, :], dd[:-1, :]), far_d[1:, :])
    thin[1:-1, :] |= a_near[1:, :] & b_near[:-1, :]
    return near, far & ~near, far_d, thin
