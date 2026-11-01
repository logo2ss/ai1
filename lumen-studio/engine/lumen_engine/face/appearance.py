"""Physically motivated appearance transfer for face replacement.

The warped source face carries the *source* photo's lighting, colour cast,
highlights and camera look. This module re-renders it under the *target*
photo's conditions:

1. **Geometry** - per-pixel surface normals of both faces from the 3D face
   mesh. The depth comes from a symmetric mean-face template fitted to the
   observed landmarks with a weak-perspective rotation, so it works the same
   with MediaPipe and with the compatibility (YuNet + LBF) landmarks.
2. **Lighting** - a 2nd-order spherical-harmonics irradiance model (9
   coefficients per colour channel, ridge-regularised, fitted by iteratively
   re-weighted least squares so beards, brows, moles, highlights and cast
   shadows are rejected as outliers) for the source skin and the target skin.
3. **Specular highlights** - dichromatic decomposition of skin pixels into
   a diffuse (skin chromaticity) and a specular (light colour) part; source
   highlights are removed before re-lighting, target highlights are added back
   sparingly at their own (geometrically correct) positions.
4. **Relighting** - ``albedo = source / source_shading``;
   ``result = albedo x target_shading`` (in linear RGB).
5. **Colour** - skin-weighted transfer of the albedo statistics in log-RGB,
   then a smooth, spatially varying local correction (normalised convolution
   of robust skin samples of both faces) for regional tone differences
   (forehead / cheeks / nose / chin), coloured light and cast shadows the
   low-order lighting model cannot represent. Lips and brows keep more of
   their own colour; eye whites and teeth get only the lighting change.
6. **Warp quality** - triangles that are folded or strongly stretched by the
   warp (parts of the face that are turned away in the source photo) are
   re-sampled from the mirrored, well-visible half of the source face.

Internal constants are read through :func:`settings.tv` so the data-driven
optimiser (engine/tools/face_train) can tune them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..imaging import color
from ..imaging.filters import erode, gaussian, normalized_blur
from .landmarks import topology
from .settings import tv

# mirror pairs used to find the face's symmetry plane (eye/brow index lists are mirror-ordered)
_LIP_PAIRS = [(61, 291), (146, 375), (91, 321), (181, 405), (84, 314), (185, 409), (40, 270), (39, 269), (37, 267)]


# --------------------------------------------------------------------------- geometry
@lru_cache(maxsize=1)
def canonical() -> dict | None:
    """Symmetric mean-face template: vertices, outward vertex normals, mirror map."""
    try:
        d = np.load(Path(__file__).parent / "data" / "face_template.npz")
        T = np.asarray(d["template478"], np.float64)[:468, :3]
    except Exception:  # noqa: BLE001 - template missing/changed: callers fall back
        return None
    t = topology()
    pairs = list(zip(t["left_eye"], t["right_eye"])) + list(zip(t["left_brow"], t["right_brow"])) + _LIP_PAIRS
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    u = (T[a] - T[b]).sum(0)
    u /= np.linalg.norm(u) + 1e-12
    c = ((T[a] + T[b]) / 2).mean(0)
    refl = T - 2 * ((T - c) @ u)[:, None] * u
    d2 = ((refl[:, None, :] - T[None]) ** 2).sum(-1)
    mirror = d2.argmin(1)
    Ts = (T + refl[mirror]) / 2  # symmetrised template
    tris = t["triangles"]
    A, B, C = Ts[tris[:, 0]], Ts[tris[:, 1]], Ts[tris[:, 2]]
    n = np.cross(B - A, C - A)
    # orient every triangle outwards (away from a point behind the face)
    depth = Ts[:, 2].max() - Ts[:, 2].min()
    centre = Ts.mean(0) + np.array([0, 0, 0.9 * depth])
    out = (A + B + C) / 3 - centre
    n *= np.sign((n * out).sum(1, keepdims=True) + 1e-12)
    vn = np.zeros_like(Ts)
    for k in range(3):
        np.add.at(vn, tris[:, k], n)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return {"vertices": Ts, "normals": vn, "mirror": mirror, "mirror_ok": float((mirror[mirror] == np.arange(len(mirror))).mean())}


def face_rotation(xy: np.ndarray) -> np.ndarray | None:
    """Weak-perspective rotation (3x3, camera frame: x right, y down, z away) of the template onto 2D landmarks."""
    can = canonical()
    if can is None:
        return None
    X = can["vertices"] - can["vertices"].mean(0)
    x = xy[:468].astype(np.float64)
    x = x - x.mean(0)
    A = np.linalg.lstsq(X, x, rcond=None)[0].T  # (2,3): x ~ X @ A.T
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    R12 = U @ Vt
    r3 = np.cross(R12[0], R12[1])
    return np.vstack([R12, r3])


def vertex_normals(xy: np.ndarray) -> np.ndarray | None:
    """Unit normals (468, 3) of the face surface in the camera frame (visible surface: n_z < 0)."""
    R = face_rotation(xy)
    if R is None:
        return None
    return canonical()["normals"] @ R.T


class MeshRaster:
    """Triangle ids and barycentric coordinates of every ROI pixel for a 2D face mesh.

    Pixels outside the mesh take the nearest triangle (barycentrics clamped),
    which gives a smooth extension into the feathered transition band.
    """

    def __init__(self, pts: np.ndarray, shape: tuple[int, int], off=(0, 0)):
        tris = topology()["triangles"]
        h, w = shape
        o = np.asarray(off, np.float64)
        d = pts[:468, :2].astype(np.float64)[tris] - o  # (K,3,2)
        e1, e2 = d[:, 1] - d[:, 0], d[:, 2] - d[:, 0]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        self.tri_sign = np.sign(det)
        mats = np.zeros((len(tris) + 1, 2, 3))
        ids = np.zeros((h, w), np.int32)
        for k in np.argsort(-np.abs(det)):
            if abs(det[k]) < 1e-6:
                continue
            inv = np.array([[e2[k, 1], -e2[k, 0]], [-e1[k, 1], e1[k, 0]]]) / det[k]
            mats[k + 1, :, :2] = inv
            mats[k + 1, :, 2] = -inv @ d[k, 0]
            cv2.fillConvexPoly(ids, np.round(d[k] * 16).astype(np.int32), int(k + 1), lineType=cv2.LINE_8, shift=4)
        self.inside = ids > 0
        if not self.inside.any():
            raise ValueError("empty mesh")
        if (~self.inside).any():
            from scipy.ndimage import distance_transform_edt

            _, (iy, ix) = distance_transform_edt(~self.inside, return_indices=True)
            ids = ids[iy, ix]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        M = mats[ids]
        l1 = M[..., 0, 0] * xx + M[..., 0, 1] * yy + M[..., 0, 2]
        l2 = M[..., 1, 0] * xx + M[..., 1, 1] * yy + M[..., 1, 2]
        bary = np.stack([1 - l1 - l2, l1, l2], -1)
        bary = np.clip(bary, 0, None)
        bary /= bary.sum(-1, keepdims=True) + 1e-12
        self.bary = bary.astype(np.float32)
        self.ids = ids - 1  # triangle index per pixel
        self.tv = tris[self.ids]  # (h,w,3) vertex ids

    def interp(self, attr: np.ndarray) -> np.ndarray:
        """Barycentric interpolation of per-vertex attributes (468, C) -> (h, w, C)."""
        a = np.asarray(attr, np.float32)
        return (a[self.tv] * self.bary[..., None]).sum(-2)


# --------------------------------------------------------------------------- spherical harmonics
def sh_basis(n: np.ndarray) -> np.ndarray:
    """Real 2nd-order SH basis (unnormalised; constants are absorbed by the fit)."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1], -1).astype(np.float32)


_RIDGE = np.array([0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0])


def fit_sh(lin: np.ndarray, Y: np.ndarray, w: np.ndarray, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Robust ridge fit of per-channel SH lighting: lin (N,C) ~ Y (N,9) @ L (9,C).

    Returns (L, robust_weights). Residuals are measured on log-luminance so dark
    beards, brows and cast shadows as well as specular highlights are rejected.
    """
    w = np.asarray(w, np.float64)
    lam = tv("sh_ridge", 0.06)
    rw = np.ones_like(w)
    L = np.zeros((Y.shape[1], lin.shape[1]))
    Yd = Y.astype(np.float64)
    ld = lin.astype(np.float64)
    lum = ld @ np.array([0.2126, 0.7152, 0.0722])[: ld.shape[1]] if ld.shape[1] == 3 else ld[:, 0]
    for _ in range(iters):
        ww = w * rw
        sw = ww.sum() + 1e-9
        G = (Yd * ww[:, None]).T @ Yd
        scale = np.mean(lum * ww) / (np.mean(ww) + 1e-12)
        reg = np.diag(_RIDGE) * lam * sw
        # pull towards uniform light (DC only) rather than towards zero for the DC term
        L = np.linalg.solve(G + reg, (Yd * ww[:, None]).T @ ld)
        pred = Yd @ L
        plum = pred @ np.array([0.2126, 0.7152, 0.0722]) if ld.shape[1] == 3 else pred[:, 0]
        r = np.log(np.maximum(lum, 1e-4)) - np.log(np.maximum(plum, 0.05 * max(scale, 1e-4)))
        med = np.median(r[w > 0.3]) if (w > 0.3).sum() > 20 else 0.0
        mad = np.median(np.abs(r[w > 0.3] - med)) * 1.4826 + 0.02 if (w > 0.3).sum() > 20 else 0.1
        rw = 1.0 / (1.0 + ((r - med) / (tv("sh_robust_k", 2.0) * mad)) ** 2)
    return L, rw


def eval_sh(L: np.ndarray, Y: np.ndarray, floor_frac: float = 0.15, weight: np.ndarray | None = None) -> np.ndarray:
    """Shading image Y @ L with a positive floor relative to the skin mean."""
    E = Y @ L.astype(np.float32)
    if weight is not None and weight.sum() > 0:
        m = (E * weight[..., None]).reshape(-1, E.shape[-1]).sum(0) / weight.sum()
    else:
        m = E.reshape(-1, E.shape[-1]).mean(0)
    return np.maximum(E, floor_frac * np.maximum(m, 1e-4)).astype(np.float32)


# --------------------------------------------------------------------------- skin / parts
def skin_likelihood(img: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Likelihood of matching the skin colour model learned (robustly) from ``seed`` pixels."""
    lab = color.rgb_to_lab(img)
    sel = seed > 0.5
    if sel.sum() < 30:
        return np.ones(img.shape[:2], np.float32)
    X = lab[sel]
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0) * 1.4826 + 1e-3
    keep = (np.abs(X - med) < 2.5 * mad).all(1)
    if keep.sum() > 20:
        X = X[keep]
    mu = X.mean(0)
    cov = np.cov(X.T) + np.diag([25.0, 3.0, 3.0])
    d = lab - mu
    m2 = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d)
    return np.exp(-0.5 * np.maximum(m2 - 7.8, 0) / 3.0).astype(np.float32)


def part_masks(shape, pts: np.ndarray, off, fs: float) -> dict:
    """Soft masks (ROI frame) of the eyes, brows, lips and inner mouth from mesh points."""
    t = topology()
    o = np.asarray(off, np.float64)

    def poly(idx, grow=0.0, convex=False):
        m = np.zeros(shape, np.uint8)
        p = np.round((pts[idx, :2] - o) * 16).astype(np.int32)
        if convex:
            p = cv2.convexHull(p)
        cv2.fillPoly(m, [p], 1, lineType=cv2.LINE_AA, shift=4)
        m = m.astype(np.float32)
        if grow > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * grow + 1),) * 2)
            m = cv2.dilate(m, k)
        return m

    eyes = np.maximum(poly(t["left_eye"]), poly(t["right_eye"]))
    brows = np.maximum(poly(t["left_brow"], fs * 0.012, True), poly(t["right_brow"], fs * 0.012, True))
    lips = poly(t["outer_lip"])
    inner = poly(t["inner_lip"])
    return {"eyes": eyes, "brows": brows, "lips": np.clip(lips - inner, 0, 1), "mouth": inner}


def bright_parts(img: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Eye whites / teeth inside ``region``: bright, low-chroma pixels (soft weight)."""
    sel = region > 0.5
    if sel.sum() < 12:
        return np.zeros(img.shape[:2], np.float32)
    lab = color.rgb_to_lab(img)
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    Lr = L[sel]
    q50, q95 = np.quantile(Lr, 0.5), np.quantile(Lr, 0.95)
    thr = max(np.quantile(Lr, 0.75), 0.5 * (q50 + q95))  # strictly the bright part (not iris / lashes)
    spread = max(q95 - np.quantile(Lr, 0.1), 5.0)
    wl = np.clip((L - thr) / (0.2 * spread) + 0.5, 0, 1)
    cthr = np.quantile(C[sel], 0.5) + 4.0
    wc = np.clip((cthr - C) / 6.0 + 0.5, 0, 1)
    return (wl * wc * region).astype(np.float32)


# --------------------------------------------------------------------------- specular
def specular_layer(lin: np.ndarray, w: np.ndarray, region: np.ndarray, fs: float, illum: np.ndarray) -> np.ndarray:
    """Specular amount (per pixel, along the light colour ``illum``) by dichromatic decomposition.

    Each pixel is explained as ``a * skin_chromaticity + b * light_colour``; the
    skin chromaticity is a smooth local estimate from diffuse skin samples, so
    natural variations (redness) are not mistaken for highlights.
    """
    eps = 1e-5
    nrm = np.linalg.norm(lin, axis=-1, keepdims=True) + eps
    chrom = lin / nrm
    cd = normalized_blur(chrom, w, max(2.0, fs * 0.08))
    cd /= np.linalg.norm(cd, axis=-1, keepdims=True) + eps
    cl = (illum / (np.linalg.norm(illum) + eps)).astype(np.float32)
    g11 = (cd * cd).sum(-1)
    g12 = (cd * cl).sum(-1)
    g22 = float(cl @ cl)
    r1 = (lin * cd).sum(-1)
    r2 = (lin * cl).sum(-1)
    det = g11 * g22 - g12 * g12
    ok = det > 0.02  # skin chromaticity distinguishable from the light colour
    b = np.where(ok, (g11 * r2 - g12 * r1) / np.maximum(det, 1e-6), 0.0)
    # baseline from (mostly diffuse) skin: removes the bias of the chromaticity estimate
    sel = w > 0.3
    if sel.sum() < 30:
        return np.zeros(lin.shape[:2], np.float32)
    q = np.quantile(b[sel], 0.7)
    base = normalized_blur(b, w * (b <= q), max(2.0, fs * 0.1))
    dev = b - base
    mad = np.median(np.abs(dev[sel])) * 1.4826 + 1e-4
    spec = np.clip(dev - tv("spec_noise_k", 2.0) * mad, 0, None)
    spec = gaussian(spec.astype(np.float32), max(0.6, fs * 0.004))
    return (spec * region).astype(np.float32)


# --------------------------------------------------------------------------- main transfer
def _wmean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = w.sum() + 1e-9
    return (x * w[..., None]).reshape(-1, x.shape[-1]).sum(0) / sw


def chroma_skin(lin: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Skin likelihood from lightness-invariant chromaticity (shadows stay skin, beards/brows/hair do not)."""
    s = lin.sum(-1, keepdims=True) + 1e-5
    rg = (lin[..., :2] / s).astype(np.float32)
    Y = color.luminance(lin)
    sel = (seed > 0.5) & (Y > 0.003)
    if sel.sum() < 30:
        return np.ones(lin.shape[:2], np.float32)
    X = rg[sel]
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0) * 1.4826 + 1e-4
    keep = (np.abs(X - med) < 2.5 * mad).all(1)
    if keep.sum() > 20:
        X = X[keep]
    mu = X.mean(0)
    cov = np.cov(X.T) + np.eye(2) * tv("chroma_skin_reg", 2e-5)
    d = rg - mu
    m2 = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d)
    p = np.exp(-0.5 * np.maximum(m2 - 6.0, 0) / 3.0)
    # very dark pixels have unreliable chromaticity
    p *= np.clip(Y / 0.004, 0, 1)
    return p.astype(np.float32)


def residual_shading(lin: np.ndarray, E: np.ndarray, w: np.ndarray, fs: float) -> np.ndarray:
    """Mid-scale shading not explained by the SH model (hard shadow edges, cast shadows, coloured bounce light).

    log(lin / E) is filled from nearby skin where ``w`` is low (features, hair,
    beard), then smoothed edge-aware so shadow terminators stay sharp while skin
    texture is removed. Returned as a multiplicative map (h, w, 3) with skin mean 1.
    """
    from ..imaging.filters import guided_filter

    r = np.log(np.maximum(lin, 1e-4)) - np.log(np.maximum(E, 1e-4))
    wf = np.clip(w, 0, 1)
    fill = normalized_blur(r, wf, max(2.0, fs * 0.04))
    a = np.clip(wf * 2.0, 0, 1)[..., None]
    filled = (r * a + fill * (1 - a)).astype(np.float32)
    guide = filled.mean(-1)
    rad = max(2, int(fs * tv("resid_radius", 0.05)))
    eps_g = tv("resid_eps", 0.02)
    sm = np.stack([guided_filter(guide, filled[..., c], rad, eps_g) for c in range(3)], -1)
    sm = gaussian(sm, max(1.0, fs * 0.012))
    sm -= _wmean(sm, wf + 1e-6)
    return np.exp(sm).astype(np.float32)


def appearance_transfer(
    warped: np.ndarray,
    target: np.ndarray,
    n_src: np.ndarray,
    n_dst: np.ndarray,
    skin_region: np.ndarray,
    parts: dict,
    fs: float,
    lighting: float,
    color_match: float,
    src_illum: np.ndarray | None = None,
    dst_illum: np.ndarray | None = None,
    debug: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Re-render the warped source face under the target's lighting and colour.

    ``n_src``/``n_dst`` are per-pixel unit normals (target frame) of the source
    surface point shown at each pixel and of the target surface. ``skin_region``
    marks the reliable inner skin (features excluded). Returns (sRGB image, report).
    """
    report: dict = {"method": "sh"}
    eps = 1e-4
    lin_s = color.srgb_to_linear(warped)
    lin_t = color.srgb_to_linear(target)
    seed = erode((skin_region > 0.5).astype(np.float32), max(1, int(fs * 0.02)))
    if seed.sum() < 60:
        seed = (skin_region > 0.5).astype(np.float32)
    unclipped_s = ((warped.max(-1) < 0.985) & (warped.min(-1) > 0.008)).astype(np.float32)
    unclipped_t = ((target.max(-1) < 0.985) & (target.min(-1) > 0.008)).astype(np.float32)
    cs_s = chroma_skin(lin_s, seed)
    w_s = skin_region * cs_s * unclipped_s
    small = fs < tv("small_face_px", 56)  # tiny faces: only the robust global steps
    w_t = skin_region * chroma_skin(lin_t, seed) * unclipped_t
    if w_s.sum() < 80 or w_t.sum() < 80:
        raise ValueError("too little visible skin for the lighting model")

    Ys, Yt = sh_basis(n_src), sh_basis(n_dst)
    rng = np.random.default_rng(3)

    def fit(lin, Y, w):  # robust SH fit on a pixel subsample
        idx = np.flatnonzero(w.ravel() > 0.05)
        if len(idx) > 9000:
            idx = rng.choice(idx, 9000, replace=False)
        L, _ = fit_sh(lin.reshape(-1, 3)[idx], Y.reshape(-1, 9)[idx], w.ravel()[idx])
        return L

    L_s = fit(lin_s, Ys, w_s)
    L_t = fit(lin_t, Yt, w_t)
    E_s = eval_sh(L_s, Ys, weight=w_s)
    E_t = eval_sh(L_t, Yt, weight=w_t)

    # ---- specular highlights (removed from the source; the target's are added back at the end)
    face_region = np.clip(gaussian(skin_region, fs * 0.01), 0, 1)
    no_eyes = 1 - np.clip(gaussian(parts["eyes"] + parts["mouth"], fs * 0.01) * 2, 0, 1)
    s_ill = np.asarray(src_illum if src_illum is not None else np.ones(3), np.float32)
    t_ill = np.asarray(dst_illum if dst_illum is not None else np.ones(3), np.float32)
    white = np.ones(3, np.float32)
    spec_s = specular_layer(lin_s, w_s, face_region * no_eyes, fs, white)
    spec_t = specular_layer(lin_t, w_t, face_region * no_eyes, fs, white)
    cl_s = cl_t = np.full(3, 1 / np.sqrt(3), np.float32)
    lum_s = color.luminance(lin_s)
    spec_strength = 0.0 if small else tv("spec_remove", 0.85) * float(np.clip(lighting, 0, 1))
    spec_rm = np.minimum(spec_s * spec_strength, 0.6 * lum_s / max(float(cl_s @ np.array([0.2126, 0.7152, 0.0722])), 0.1))
    # Fit diffuse albedo on BOTH images. Comparing source diffuse to target
    # diffuse+specular brightened the skin, then added highlights a second time.
    spec_restore = np.minimum(spec_t * spec_strength, 0.6 * color.luminance(lin_t) / max(float(cl_t @ np.array([0.2126, 0.7152, 0.0722])), 0.1))
    diffuse_t = np.maximum(lin_t - spec_restore[..., None] * cl_t, eps)
    diffuse_s = np.maximum(lin_s - spec_rm[..., None] * cl_s, eps)
    report["specular_removed"] = round(float((spec_rm * face_region).sum() / max(face_region.sum(), 1)) * 255, 2)

    # ---- albedo = image / SH shading (smooth by construction: cannot create blotches)
    m_s, m_t = _wmean(E_s, w_s), _wmean(E_t, w_t)
    alb_s = diffuse_s / E_s
    alb_t = diffuse_t / E_t

    # ---- shading pattern: target's (strength alpha), limited chroma (coloured light) and range
    alpha = float(np.clip(tv("light_gain", 1.3) * lighting, 0, 1))
    pat = ((E_t / m_t) / (E_s / m_s)) ** alpha
    pY = color.luminance(pat)[..., None]
    pC = np.clip(pat / np.maximum(pY, 1e-4), tv("light_chroma_lo", 0.85), tv("light_chroma_hi", 1.18)) ** tv("light_chroma_k", 0.8)
    pat = np.clip(pY, tv("light_ratio_lo", 0.3), tv("light_ratio_hi", 3.0)) * pC
    report["shading_change"] = round(float((np.abs(np.log(pY[..., 0])) * w_s).sum() / w_s.sum()), 3)

    # ---- global albedo colour transfer in log-RGB with robust (median / MAD) skin statistics
    la_s = np.log(np.maximum(alb_s, eps))
    la_t = np.log(np.maximum(alb_t, eps))
    sel_s, sel_t = w_s > 0.3, w_t > 0.3
    qs = np.array([0.25, 0.5, 0.75])
    st_s = np.stack([color.weighted_quantile(la_s[..., c][sel_s], w_s[sel_s], qs) for c in range(3)], 1)
    st_t = np.stack([color.weighted_quantile(la_t[..., c][sel_t], w_t[sel_t], qs) for c in range(3)], 1)
    ms_, mt_ = st_s[1], st_t[1]
    sc = np.clip((st_t[2] - st_t[0]) / np.maximum(st_s[2] - st_s[0], 1e-4), tv("std_clamp_lo", 0.8), tv("std_clamp_hi", 1.25))
    k = float(np.clip(color_match, 0, 1))
    keep = np.clip(gaussian(np.maximum(parts["lips"], parts["brows"]), max(1.0, fs * 0.01)), 0, 1)[..., None]
    shift = mt_ - ms_ + np.log(m_t) - np.log(m_s)  # albedo + overall exposure / light colour
    shiftY = float(shift @ np.array([0.2126, 0.7152, 0.0722]))
    # lips and brows keep part of their own colour (person-specific); luminance still follows
    # hair (beard, stubble, brows): non-skin chromaticity AND darker than the surrounding skin
    Yd = color.luminance(diffuse_s)
    Yloc = normalized_blur(Yd, w_s, max(2.0, fs * 0.1))
    dark = np.clip(1.6 - 2.0 * Yd / np.maximum(Yloc, 1e-4), 0, 1)
    hair = ((1 - np.clip(gaussian(cs_s, max(0.8, fs * 0.006)), 0, 1)) * dark)[..., None]
    # lips keep part of their own colour; hair (brows, beard, stubble) keeps its own hue entirely
    shift_map = shiftY + (shift - shiftY) * (1 - tv("feature_keep", 0.45) * keep) * (1 - np.clip(hair * tv("hair_keep", 1.0), 0, 1))
    log_out = ms_ + (la_s - ms_) * (1 + (sc - 1) * k) + k * shift_map + np.log(E_s) + np.log(np.maximum(pat, 1e-4))
    report["global_shift"] = np.round(np.exp(shift), 3).tolist()

    # ---- gentle local correction: regional tone (forehead/cheeks/chin), cast shadows, coloured light
    sig = max(2.0, fs * tv("local_sigma", 0.12))
    log_t = np.log(np.maximum(diffuse_t, eps))
    mu_t = normalized_blur(log_t, w_t, sig)
    mu_o = normalized_blur(log_out, w_s, sig)
    field = mu_t - mu_o
    field -= _wmean(field, np.minimum(w_s, w_t) + 1e-6)
    lim = tv("local_clip", 0.22)
    fY = np.clip(field @ np.array([0.2126, 0.7152, 0.0722], np.float32), -lim, lim)[..., None]
    fC = np.clip(field - fY, -0.5 * lim, 0.5 * lim)
    kl = 0.0 if small else tv("local_strength", 0.6)
    conf = np.minimum(gaussian(w_t, sig), gaussian(w_s, sig)) / max(float(gaussian(np.ones_like(w_t), sig).max()), 1e-6)
    conf = np.clip(conf * 3, 0, 1)[..., None]
    corr = (fY * kl * min(1.0, alpha) + fC * kl * k) * conf
    log_out = log_out + corr * (1 - 0.5 * keep)
    out_lin = np.exp(log_out)
    R_t = np.exp(corr)
    R_s = np.ones_like(R_t)

    # ---- target highlights back at their own positions (sparingly)
    if not small:
        out_lin = out_lin + spec_restore[..., None] * cl_t

    # ---- eye whites & teeth: lighting change only (no skin-albedo shift), matched to the target's when visible
    illum_ratio = t_ill / np.maximum(s_ill, 1e-4)
    illum_ratio /= float(illum_ratio @ np.array([0.2126, 0.7152, 0.0722]))
    illum_ratio = np.clip(illum_ratio, 0.8, 1.25)
    for part in ("eyes", "mouth"):
        reg = parts[part]
        if small or reg.sum() < 12:
            continue
        b_s = bright_parts(warped, reg)
        if part == "mouth" and b_s.sum() < 0.15 * reg.sum():
            continue  # no teeth visible: lips/tongue keep the normal transfer
        b_t = bright_parts(target, reg)
        neutral = lin_s * pat * illum_ratio
        g_exp = float(np.exp(np.clip(0.5 * shiftY, -0.5, 0.5)))
        if b_t.sum() >= 15 and b_s.sum() >= 15:
            g = _wmean(lin_t, b_t) / np.maximum(_wmean(neutral, b_s), 1e-4)
            gY = float(np.clip(g @ np.array([0.2126, 0.7152, 0.0722]), 0.6, 1.35))
            g = gY * np.clip(g / max(gY, 1e-4), 0.94, 1.06)
        else:
            g = np.full(3, g_exp)
        wpart = np.clip(gaussian(b_s, 0.6) * 1.3, 0, 1)[..., None]  # eye whites / teeth only (eyelid skin keeps the skin transfer)
        out_lin = out_lin * (1 - wpart) + neutral * g * wpart
        report[f"{part}_gain"] = round(float(np.mean(g)), 3)

    out = color.linear_to_srgb(np.clip(out_lin, 0, 1))
    if debug is not None:
        debug.update(E_s=E_s, E_t=E_t, spec_s=spec_s, spec_t=spec_t, w_s=w_s, w_t=w_t, field=np.log(R_t / R_s),
                     lit=color.linear_to_srgb(np.clip(diffuse_s * pat, 0, 1)), ratio=pY[..., 0])
    report["sh_source"] = np.round(L_s[:, 1] / max(L_s[0, 1], 1e-6), 3).tolist()
    report["sh_target"] = np.round(L_t[:, 1] / max(L_t[0, 1], 1e-6), 3).tolist()
    return out.astype(np.float32), report


# --------------------------------------------------------------------------- warp quality / symmetric completion
def warp_quality(raster: MeshRaster, src_pts: np.ndarray, dst_pts: np.ndarray, n_src_v: np.ndarray | None) -> np.ndarray:
    """Per-pixel sampling quality in [0, 1] of the source texture under the mesh warp.

    Low where triangles are folded (self-occluded in the source), strongly
    stretched (the source shows them foreshortened) or facing away from the
    source camera.
    """
    tris = topology()["triangles"]
    s = src_pts[:468, :2].astype(np.float64)[tris]
    d = dst_pts[:468, :2].astype(np.float64)[tris]
    cs = (s[:, 1, 0] - s[:, 0, 0]) * (s[:, 2, 1] - s[:, 0, 1]) - (s[:, 1, 1] - s[:, 0, 1]) * (s[:, 2, 0] - s[:, 0, 0])
    cd = (d[:, 1, 0] - d[:, 0, 0]) * (d[:, 2, 1] - d[:, 0, 1]) - (d[:, 1, 1] - d[:, 0, 1]) * (d[:, 2, 0] - d[:, 0, 0])
    sign = np.sign((cs * cd).sum())
    ratio = cs * cd * sign / np.maximum(cd * cd, 1e-9)  # signed source/target area ratio
    ref = np.median(ratio[np.abs(cd) > np.quantile(np.abs(cd), 0.3)])
    rel = ratio / max(ref, 1e-9)
    lo, hi = tv("stretch_bad", 0.2), tv("stretch_ok", 0.55)
    q_tri = np.clip((rel - lo) / (hi - lo), 0, 1)
    if n_src_v is not None:
        # facing: n_z < 0 is visible; grazing angles are poorly sampled
        nz = -n_src_v[:, 2]
        fz = nz[tris].mean(1)
        q_tri = np.minimum(q_tri, np.clip((fz - 0.05) / 0.25, 0, 1))
    return q_tri[raster.ids].astype(np.float32)
