"""Face masks: semantic face parsing, occlusion handling and the soft replacement mask.

With the face-parsing model installed (model id ``face-parsing``, see
:mod:`.parsing`) every mask is derived from per-pixel class probabilities of
BOTH photos (the source's are pulled through the same warp as its pixels):

* **Replaced area** ``K = P_target(face) x P_source(face) x prior``, where
  face = skin, brows, eyes, nose, lips, inner mouth and the prior is the
  target's mesh oval (slightly enlarged, most at the forehead). Target hair
  strands, glasses, hats, hands and microphones are never painted over; the
  source's fringe, glasses or hat are never transplanted - where the source
  lacks skin the target shows, the area shrinks there.
* **Transition** - the soft mask is 0 on the boundary of ``K`` and rises inside
  it, so the whole feather band lies on skin in both photos (no halo of
  foreign hair/background, no double jaw: it ends at whichever jaw line is
  inner). Target occluders keep a sharp, anti-aliased matte edge (they lie on
  top of the new face); along the face outline and at source occluders the
  fade is smooth and wider towards the forehead/hairline. Its position is a
  minimum-cost closed seam through the band (polar dynamic programming over
  colour/gradient disagreement and structure), so the fade runs where both
  photos agree and does not cut through brows, folds or shadow edges.
* **Mouth/eyes** - the inner mouth (teeth, tongue) and the lips come from the
  parsing labels, which decide whether the target's mouth is kept.
* **Statistics** - skin used for colour/lighting statistics and the Poisson
  boundary weights are skin in both photos by their labels.

Without the model (or if it fails) the previous heuristic masks are used:
landmark polygons refined by a seeded colour skin model.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..imaging import color
from ..imaging.filters import dilate, erode, gaussian, guided_filter
from . import parsing
from .geometry import dense_warp_maps, feature_mask, poly_mask
from .landmarks import Face, topology
from .settings import FaceSwapSettings  # noqa: F401

log = logging.getLogger("lumen.face.masking")

# Debug data of the most recent parsed mask (label maps, regions) for tools/face_masks.
LAST_DEBUG: dict = {}
USE_SEAM = True  # min-cost seam placement of the outline fade (False: plain distance fade; for A/B tests)


def _skin_visibility(img: np.ndarray, face: Face) -> float:
    """Fraction of the face that is visible (not covered by hair, glasses, hat, hands).

    Uses the face-parsing model when installed (this also caches the parse of
    the full photo for the mask step); otherwise the fraction of the face oval
    whose colour matches a cheek skin model.
    """
    t = topology()
    if parsing.available():
        x0, y0, x1, y1 = face.bbox
        roi = (int(max(0, y0)), int(min(img.shape[0], y1 + 1)), int(max(0, x0)), int(min(img.shape[1], x1 + 1)))
        fp = parsing.parse_region(img, face.points, roi) if roi[1] > roi[0] + 4 and roi[3] > roi[2] + 4 else None
        if fp is not None:
            region = erode(poly_mask(fp.shape, face.xy[t["oval"]], (roi[2], roi[0])), max(1, int(face.size * 0.06)))
            if region.sum() > 20:
                return float((fp.face() * region).sum() / region.sum())
    x0, y0, x1, y1 = face.bbox
    roi = (int(max(0, y0)), int(min(img.shape[0], y1 + 1)), int(max(0, x0)), int(min(img.shape[1], x1 + 1)))
    sub = img[roi[0] : roi[1], roi[2] : roi[3]]
    if sub.size == 0:
        return 1.0
    off = (roi[2], roi[0])
    shape = sub.shape[:2]
    oval = poly_mask(shape, face.xy[t["oval"]], off)
    feat = feature_mask(shape, face.xy, off, face.size * 0.03)
    region = oval * (1 - feat)
    cheeks = np.zeros(shape, np.float32)
    for idx in (50, 280, 4, 205, 425):  # cheeks, nose, mid-cheeks: rarely occluded
        cx, cy = face.xy[idx] - np.asarray(off)
        cv2.circle(cheeks, (int(cx), int(cy)), max(2, int(face.size * 0.06)), 1.0, -1)
    p = _skin_probability(sub, cheeks * region)
    return float((p * region).sum() / max(region.sum(), 1))

def _skin_probability(img: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Per-pixel likelihood of matching the skin colour model learned from ``seed``."""
    lab = color.rgb_to_lab(img)
    w = seed > 0.5
    if w.sum() < 30:
        return np.ones(img.shape[:2], np.float32)
    X = lab[w][:, :]
    mu = X.mean(0)
    # robust covariance (trim outliers such as stubble or specular highlights)
    d0 = X - np.median(X, 0)
    keep = (np.abs(d0) < 3 * (np.median(np.abs(d0), 0) * 1.4826 + 1e-3)).all(1)
    if keep.sum() > 20:
        X = X[keep]
    mu = X.mean(0)
    cov = np.cov(X.T) + np.diag([16.0, 2.0, 2.0])
    inv = np.linalg.inv(cov)
    d = lab - mu
    m2 = np.einsum("...i,ij,...j->...", d, inv, d)
    # chi-square(3): 95 % of skin pixels have m2 < 7.8
    return np.exp(-0.5 * np.maximum(m2 - 7.8, 0) / 2.0).astype(np.float32)

def _heuristic_masks(target: np.ndarray, warped: np.ndarray, dst_face: Face, dst_pts: np.ndarray, src_support: np.ndarray, roi, s: FaceSwapSettings, src_face: Face | None = None):
    """Masks without the parsing model: landmark polygons + seeded colour skin model."""
    t = topology()
    y0, y1, x0, x1 = roi
    shape = (y1 - y0, x1 - x0)
    off = (x0, y0)
    fs = dst_face.size
    # face oval, optionally scaled around the face centre
    oval = dst_pts[t["oval"]]
    c = dst_pts[[1, 4, 6, 168]].mean(0)
    oval = c + (oval - c) * s.mask_extent
    hull = poly_mask(shape, oval, off)
    hull *= (src_support > 0.5).astype(np.float32)
    feat = feature_mask(shape, dst_pts, off, fs * 0.03)
    core_skin = erode(hull, int(fs * 0.12)) * (1 - feat)

    # skin likelihood in the outer band (hair, glasses rims, hands, beards stop the mask)
    p_t = _skin_probability(target, core_skin)
    p_s = _skin_probability(warped, core_skin)
    p = np.minimum(p_t, p_s)
    p = gaussian(p, max(1.0, fs * 0.01))
    band_depth = fs * 0.14
    dist_in = cv2.distanceTransform((hull > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    inner_w = np.clip(dist_in / band_depth, 0, 1)
    # Above the eyebrows hair/bangs/hats can reach far into the face: apply the
    # skin test across the whole forehead, not only in the outer band.
    brow = dst_pts[[105, 66, 107, 336, 296, 334]]
    ex = dst_pts[263] - dst_pts[33]
    ex = ex / (np.linalg.norm(ex) + 1e-9)
    ey = np.array([-ex[1], ex[0]])
    if ey @ (dst_pts[152] - dst_pts[10]) < 0:
        ey = -ey  # ey points from forehead to chin
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    v = (xx + off[0] - brow[:, 0].mean()) * ey[0] + (yy + off[1] - brow[:, 1].mean()) * ey[1]
    forehead = np.clip(-(v + fs * 0.03) / (fs * 0.05), 0, 1)
    inner_w = inner_w * (1 - forehead)
    mask = hull * (inner_w + (1 - inner_w) * p)
    # never let the skin test punch holes into the inner face (below the brows)
    mask = np.maximum(mask, erode(hull, int(band_depth)) * (1 - forehead))

    mouth_kept = False
    open_t = dst_face.mouth_openness()
    open_s = src_face.mouth_openness() if src_face is not None else open_t
    if s.keep_target_mouth and abs(open_s - open_t) > 0.012:
        # the expressions differ (smiling with teeth vs closed mouth): warping one mouth
        # into the other stretches lips and teeth unnaturally - keep the target's mouth
        lips = poly_mask(shape, dst_face.xy[t["outer_lip"]], off)
        lips = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(fs * 0.05)) | 1,) * 2))
        mask *= 1 - gaussian(lips, max(1.0, fs * 0.015))
        mouth_kept = "expressions differ"
    elif s.keep_target_mouth and open_t > 0.003:
        inner = poly_mask(shape, dst_face.xy[t["inner_lip"]], off)
        inner = cv2.dilate(inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask *= 1 - gaussian(inner, max(1.0, fs * 0.006))
        mouth_kept = True

    # keep the transition inside the face region, then edge-aware feathering
    er = s.edge_refinement
    shrink = int(fs * (0.01 + 0.02 * er))
    mask = erode(mask, shrink)
    radius = max(2, int(fs * (0.015 + 0.05 * er)))
    soft = guided_filter(target.astype(np.float32), mask.astype(np.float32), radius, 2e-3)
    soft = gaussian(np.clip(soft, 0, 1), max(0.8, fs * (0.006 + 0.02 * er)))
    soft = np.clip(soft, 0, 1) * (hull > 0)  # do not spill outside the face contour
    soft = gaussian(soft, 0.8)
    return np.clip(soft, 0, 1).astype(np.float32), core_skin, feat, mouth_kept, p_t * p_s


# --------------------------------------------------------------------------- parsing-based masks
def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _down_axis(pts: np.ndarray) -> np.ndarray:
    """Unit vector in the image pointing from the forehead to the chin."""
    ex = pts[263] - pts[33]
    ex = ex / (np.linalg.norm(ex) + 1e-9)
    ey = np.array([-ex[1], ex[0]])
    if ey @ (pts[152] - pts[10]) < 0:
        ey = -ey
    return ey


def _dist_to(region: np.ndarray) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest pixel of ``region`` (bool)."""
    if not region.any():
        return np.full(region.shape, 1e6, np.float32)
    return cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5)


def _fill_specks(p: np.ndarray, region: np.ndarray, fs: float) -> np.ndarray:
    """Set small, compact holes of a face probability map inside ``region`` to face."""
    holes = ((p < 0.5) & region).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    if n <= 1:
        return p
    amax, lmax = (0.06 * fs) ** 2, 0.12 * fs
    small = (st[:, cv2.CC_STAT_AREA] < amax) & (np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) < lmax)
    small[0] = False
    if not small.any():
        return p
    fill = dilate(small[cc].astype(np.uint8), 1) > 0
    return np.where(fill, 1.0, p).astype(np.float32)


def _seam_cost(target: np.ndarray, warped: np.ndarray, stats: np.ndarray, fs: float) -> np.ndarray:
    """Per-pixel cost of letting the transition run through a pixel.

    High where either photo has structure (brows, folds, shadow edges: a half
    faded structure ghosts) or where the photos disagree after a global colour
    normalisation over ``stats`` (common skin), low on smooth matching skin.
    """
    lab_t, lab_w = color.rgb_to_lab(target), color.rgb_to_lab(warped)
    w = stats > 0.5
    if w.sum() > 30:
        mt, st = lab_t[w].mean(0), lab_t[w].std(0) + 1e-3
        mw, sw_ = lab_w[w].mean(0), lab_w[w].std(0) + 1e-3
        lab_w = (lab_w - mw) * np.clip(st / sw_, 0.5, 2.0) + mt
    sig = max(0.8, fs * 0.012)
    lt, lw = gaussian(lab_t, sig), gaussian(lab_w, sig)
    diff = np.sqrt(((lt - lw) ** 2 * np.array([1.0, 1.5, 1.5], np.float32)).sum(-1))

    def gmag(L):
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy) / 8.0

    struct = gaussian(gmag(lt[..., 0]) + gmag(lw[..., 0]), sig)
    return (diff / 6.0 + struct / 2.5).astype(np.float32)


def _polar_seam(inside: np.ndarray, outer_edge: np.ndarray, cost: np.ndarray, centre, w_min: float, w_max: float) -> np.ndarray | None:
    """Minimum-cost closed seam inside ``inside`` within [w_min, w_max] px of ``outer_edge``.

    Rays from ``centre``: for every angle the first pixel of ``outer_edge`` sets
    the edge radius; dynamic programming over angles picks a seam radius per
    angle (smoothness |dr| <= 1 px per step, slight preference for the outer
    part of the band to keep the replaced area large). Returns the filled seam
    region (bool), or None if the region is not star-shaped enough.
    """
    h, w = inside.shape
    cx, cy = float(centre[0]), float(centre[1])
    if not (0 <= cx < w and 0 <= cy < h) or not inside[int(cy), int(cx)]:
        return None
    rmax = float(np.hypot(max(cx, w - cx), max(cy, h - cy)))
    n_ang = int(np.clip(2 * np.pi * rmax / 3.0, 90, 540))
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rs = np.arange(0, rmax, 1.0, dtype=np.float32)
    X = cx + np.cos(ang)[:, None] * rs[None, :]
    Y = cy + np.sin(ang)[:, None] * rs[None, :]
    valid = (X >= 0) & (X <= w - 1) & (Y >= 0) & (Y <= h - 1)
    Xi, Yi = np.clip(np.round(X), 0, w - 1).astype(int), np.clip(np.round(Y), 0, h - 1).astype(int)
    hit = outer_edge[Yi, Xi] | ~valid
    r_edge = np.argmax(hit, axis=1).astype(np.float32)  # first hit along each ray
    r_edge[~hit.any(1)] = len(rs) - 1
    nb = int(np.ceil(w_max - w_min)) + 1
    if nb < 2 or np.median(r_edge) < w_max + 2:
        return None
    off = w_min + np.arange(nb, dtype=np.float32)  # inward offsets from the edge
    R = np.clip(r_edge[:, None] - off[None, :], 0, None)
    Cs = cv2.remap(cost, (cx + np.cos(ang)[:, None] * R).astype(np.float32), (cy + np.sin(ang)[:, None] * R).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Cs = Cs + 0.15 * (off[None, :] - w_min) / max(1.0, w_max - w_min)
    # doubled sequence so the seam closes on itself; keep the middle lap
    C2 = np.vstack([Cs, Cs])
    acc = C2[0].copy()
    back = np.zeros(C2.shape, np.int8)
    big = 1e9
    for i in range(1, len(C2)):
        left = np.concatenate([[big], acc[:-1]])
        right = np.concatenate([acc[1:], [big]])
        stack = np.stack([left, acc, right])
        k = stack.argmin(0)
        acc = stack[k, np.arange(nb)] + C2[i]
        back[i] = k.astype(np.int8) - 1
    path = np.zeros(len(C2), int)
    path[-1] = int(acc.argmin())
    for i in range(len(C2) - 1, 0, -1):
        path[i - 1] = np.clip(path[i] + back[i, path[i]], 0, nb - 1)  # back = previous - current state
    sel = path[n_ang // 2 : n_ang // 2 + n_ang]
    a_sel = ang[(np.arange(n_ang // 2, n_ang // 2 + n_ang)) % n_ang]
    r_sel = np.clip(r_edge[(np.arange(n_ang // 2, n_ang // 2 + n_ang)) % n_ang] - off[sel], 0, None)
    order = np.argsort(a_sel)
    poly = np.stack([cx + np.cos(a_sel[order]) * r_sel[order], cy + np.sin(a_sel[order]) * r_sel[order]], 1)
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [np.round(poly * 16).astype(np.int32)], 1, lineType=cv2.LINE_AA, shift=4)
    return region > 0


def _source_parse(warped, dst_pts, src_face, roi, shape, src_img=None, warp_maps=None):
    """Class probabilities of the source face in the target ROI frame.

    Preferably the source is parsed in its own photo (natural context) and the
    probabilities are pulled through the same warp as its pixels; the parse of
    the source photo is normally already cached by the compatibility analysis.
    Otherwise the warped source itself is parsed.
    """
    if src_face is not None:
        try:
            if warp_maps is not None:
                mx, my = warp_maps[0], warp_maps[1]
            else:
                mx, my, _ = dense_warp_maps(src_face.xy[:468], dst_pts, shape, roi)
            if src_img is not None:
                ps = parsing.parse_through_warp(src_img, src_face.points, mx, my)
            else:
                ps = parsing.cached_through_warp(src_face.points, mx, my)
            if ps is not None:
                return ps, "source photo"
        except Exception:  # noqa: BLE001 - degenerate mesh etc.: parse the warped image instead
            log.debug("warp pull-through of the source parse failed", exc_info=True)
    ps = parsing.parse_region(warped, dst_pts, roi, offset=(roi[2], roi[0]), tag=":warped")
    return ps, "warped source"


def _parsed_masks(target, warped, dst_face, dst_pts, src_support, roi, s, src_face, src_img=None, warp_maps=None):
    if not parsing.available():
        return None
    t = topology()
    y0, y1, x0, x1 = roi
    shape = (y1 - y0, x1 - x0)
    off = (x0, y0)
    fs = float(dst_face.size)
    pt = parsing.parse_region(target, dst_face.points, roi, offset=off)
    if pt is None:
        return None
    ps, ps_from = _source_parse(warped, dst_pts, src_face, roi, shape, src_img, warp_maps)
    if ps is None:
        return None

    # ---- geometric prior: the target's mesh oval, enlarged a little (most at the forehead,
    # where the mesh stops below the hairline); the parse decides the exact edge inside it
    oval = dst_pts[t["oval"]].astype(np.float64)
    c = dst_pts[[1, 4, 6, 168]].mean(0)
    rel = (oval - c) * s.mask_extent
    nrm = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9
    ey = _down_axis(dst_pts)
    up = np.clip(-(rel @ ey) / nrm[:, 0], 0, 1)  # 1 at the top of the oval
    grow = fs * (0.025 + 0.10 * up**2)
    prior = poly_mask(shape, c + rel + rel / nrm * grow[:, None], off)
    hull = poly_mask(shape, c + rel, off)
    sup = dilate((src_support > 0.5).astype(np.uint8), max(1, int(fs * 0.12))).astype(np.float32)

    inner_hull = erode(hull, max(1, int(fs * 0.03))) > 0.5
    # crisp, anti-aliased versions of the face probabilities (small/blurry faces give soft
    # probabilities); small compact non-face specks (moles, highlights) are parse noise
    face_t = _fill_specks(_smoothstep((pt.face() - 0.2) / 0.6), inner_hull, fs)
    face_s = _fill_specks(_smoothstep((ps.face() - 0.2) / 0.6), inner_hull, fs)
    K = face_t * face_s * prior * sup

    # ---- mouth: keep the target's inner mouth (teeth/tongue) when it is open, its whole
    # mouth when the expressions differ (warping one mouth into another looks wrong)
    mouth_t, lips_t = pt.group("mouth"), pt.group("lips")
    a_t = float(((mouth_t > 0.5) & inner_hull).sum()) / fs**2
    open_t_lm = dst_face.mouth_openness()
    open_s_lm = src_face.mouth_openness() if src_face is not None else open_t_lm
    a_s = float(((ps.group("mouth") > 0.5) & inner_hull).sum()) / fs**2
    keep = np.zeros(shape, np.float32)
    mouth_kept: bool | str = False
    if s.keep_target_mouth:
        differ = abs(open_s_lm - open_t_lm) > 0.012 or abs(a_s - a_t) > 0.012
        if differ:
            lips = np.clip(mouth_t + lips_t, 0, 1)
            lips = dilate((lips > 0.35).astype(np.uint8), max(1, int(fs * 0.02))).astype(np.float32)
            keep = np.maximum(keep, gaussian(lips, max(0.8, fs * 0.012)))
            mouth_kept = "expressions differ"
        elif a_t > 0.0015 or open_t_lm > 0.003:
            inner = dilate((mouth_t > 0.4).astype(np.uint8), 1).astype(np.float32)
            keep = np.maximum(keep, gaussian(inner, max(0.6, fs * 0.004)))
            mouth_kept = True
    # ---- eyes: if the source's eyes are (nearly) closed where the target's are open, the
    # stretched lids give hazy eyes without irises - keep the target's eyes instead
    eyes_t, eyes_s = pt.group("eyes"), ps.group("eyes")
    e_t = float(((eyes_t > 0.5) & inner_hull).sum()) / fs**2
    e_s = float(((eyes_s > 0.5) & inner_hull).sum()) / fs**2
    eyes_kept = bool(e_t > 0.002 and e_s < 0.45 * e_t)
    if eyes_kept:
        ek = dilate((eyes_t > 0.4).astype(np.uint8), max(1, int(fs * 0.025))).astype(np.float32)
        keep = np.maximum(keep, gaussian(ek, max(0.8, fs * 0.012)))
    K = K * (1 - keep)

    # ---- transition
    er = float(np.clip(s.edge_refinement, 0, 1))
    Kb = K > 0.5
    # target occluders (hair strands, glasses, hands, microphone, kept mouth) inside its face
    # outline lie on top of the new face: sharp matte edge there
    occ_t = ((face_t < 0.5) & inner_hull) | (keep > 0.5)
    smooth_edge = ~Kb & ~occ_t
    d_s = _dist_to(smooth_edge)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    brow = dst_pts[[105, 66, 107, 336, 296, 334]].mean(0)
    v = ((xx + off[0] - brow[0]) * ey[0] + (yy + off[1] - brow[1]) * ey[1]) / fs
    forehead = np.clip(-v / 0.12, 0, 1)  # 0 below the brows -> 1 high on the forehead
    bw = max(1.5, fs * (0.018 + 0.03 * er))  # nominal transition width
    band = bw * (1.0 + 0.35 * forehead)  # wider fade into the forehead/hairline
    a_out = _smoothstep(d_s / band)
    skin_t, skin_s = pt.group("skin", "nose"), ps.group("skin", "nose")
    skin_both = np.clip(skin_t * skin_s, 0, 1).astype(np.float32)
    seam_used = False
    if smooth_edge.any() and USE_SEAM:
        # outer outline = smooth-edge pixels connected to the ROI border; holes inside the
        # face (source glasses, a source microphone blob) keep the plain distance fade
        n_cc, cc = cv2.connectedComponents(smooth_edge.astype(np.uint8), connectivity=8)
        border = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
        outer = np.isin(cc, border[border > 0])
        holes = smooth_edge & ~outer
        centre = dst_pts[[1, 4, 6, 168]].mean(0) - np.asarray(off, np.float64)
        try:
            cost = _seam_cost(target, warped, (skin_both > 0.5) & Kb, fs)
            region = _polar_seam(Kb | occ_t, outer, cost, centre, 0.5 * bw, 1.7 * bw)
        except Exception:  # noqa: BLE001
            log.debug("seam search failed", exc_info=True)
            region = None
        if region is not None:
            sd = _dist_to(~region) - _dist_to(region)  # signed distance to the seam (+ inside)
            a_seam = _smoothstep(0.5 + sd / band)
            a_out = a_seam * _smoothstep(_dist_to(outer) / (0.35 * bw)) * _smoothstep(_dist_to(holes) / band)
            seam_used = True
    # no thin slivers of replaced skin (e.g. a strip up the nose bridge between source glasses)
    r_open = int(fs * 0.035)
    if r_open >= 2:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_open + 1, 2 * r_open + 1))
        opened = cv2.morphologyEx(a_out.astype(np.float32), cv2.MORPH_OPEN, k_open)
        a_out = np.minimum(a_out, gaussian(opened, max(0.8, fs * 0.01)))
    matte = np.clip(face_t, 0, 1) * (1 - keep)  # anti-aliased occluder edges from the parse
    # The parse locates occluders; it must not erode a clear face twice (the
    # min-cost seam plus its wide inward fade caused the v1.4 coverage loss).
    geometric = _heuristic_masks(target, warped, dst_face, dst_pts, src_support, roi, s, src_face)[0]
    semantic = np.clip(face_t * face_s * (1-keep) * prior * sup, 0, 1)
    soft = np.clip(np.maximum(a_out * matte, geometric) * semantic, 0, 1).astype(np.float32)

    # ---- statistics masks
    feat_s = np.clip(ps.group("brows", "eyes", "lips", "mouth"), 0, 1)
    feat = np.maximum(dilate((feat_s > 0.4).astype(np.uint8), max(1, int(fs * 0.015))).astype(np.float32), feature_mask(shape, dst_pts, off, fs * 0.01) * (feat_s > 0.05))
    feat = np.clip(gaussian(feat, 0.8) * 1.2, 0, 1).astype(np.float32)
    core0 = ((skin_both > 0.5) & (soft > 0.9) & (feat < 0.05)).astype(np.float32)
    core = erode(core0, max(1, int(fs * 0.015)))
    if core.sum() < max(200.0, 0.5 * core0.sum()):  # small faces: keep enough samples
        core = core0

    LAST_DEBUG.clear()
    LAST_DEBUG.update(
        method="parsing", source_parse=ps_from, roi=roi, labels_t=pt.labels(), labels_s=ps.labels(), K=K.astype(np.float32),
        occ_t=occ_t, seam=seam_used, eyes_kept=eyes_kept, eye_area_t=round(e_t, 4), eye_area_s=round(e_s, 4), mouth_area_t=round(a_t, 4), mouth_area_s=round(a_s, 4), mouth_lm_t=round(open_t_lm, 4), mouth_lm_s=round(open_s_lm, 4),
    )
    return soft, core.astype(np.float32), feat, mouth_kept, skin_both


def _face_masks(target: np.ndarray, warped: np.ndarray, dst_face: Face, dst_pts: np.ndarray, src_support: np.ndarray, roi, s: FaceSwapSettings, src_face: Face | None = None, *, src_img: np.ndarray | None = None, warp_maps=None):
    """Soft replacement mask and helper masks in the target ROI frame.

    Returns ``(soft, skin, feat, mouth_kept, skin_both)``: the soft composite
    mask, skin pixels for colour/lighting statistics, the source's facial
    features (eyes, brows, lips), whether/why the target mouth was kept, and
    per-pixel skin weights for the Poisson boundary. ``src_img`` and
    ``warp_maps`` (map_x, map_y of the source warp) are optional; without them
    the source parse comes from the cache or from the warped image.
    """
    try:
        res = _parsed_masks(target, warped, dst_face, dst_pts, src_support, roi, s, src_face, src_img, warp_maps)
    except Exception:  # noqa: BLE001 - never fail a swap because of the optional model
        log.exception("Face-parsing masks failed; using the heuristic masks")
        res = None
    if res is not None:
        return res
    LAST_DEBUG.clear()
    LAST_DEBUG["method"] = "heuristic"
    return _heuristic_masks(target, warped, dst_face, dst_pts, src_support, roi, s, src_face)
