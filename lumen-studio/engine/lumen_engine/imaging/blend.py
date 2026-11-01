"""Gradient-domain and multi-scale blending.

* :func:`poisson_blend` - Poisson image editing (Pérez et al. 2003) with
  Dirichlet boundary conditions, optional mixed gradients and a *robust
  boundary* that prevents colour bleeding when the boundary crosses
  dissimilar content (hair, background, shadows).
* :func:`multiband_blend` - Laplacian-pyramid blending (Burt & Adelson 1983)
  which blends low frequencies over wide transitions and high frequencies over
  narrow ones, so texture does not get blurred or ghosted at the seam.

All functions operate on the bounding box of the mask only; pixels outside
the mask's support are returned bit-exact.
"""

from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp

from .filters import bbox_of, dilate, gaussian, normalized_blur

try:  # algebraic multigrid makes megapixel solves interactive
    import pyamg
except Exception:  # pragma: no cover
    pyamg = None


def _laplacian_matrix(idx: np.ndarray, inside: np.ndarray) -> sp.csr_matrix:
    """4-neighbour Laplacian restricted to ``inside`` pixels (Dirichlet outside)."""
    h, w = inside.shape
    n = int(inside.sum())
    ys, xs = np.nonzero(inside)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ok[ok] = inside[ny[ok], nx[ok]]
        rows.append(np.nonzero(ok)[0])
        cols.append(idx[ny[ok], nx[ok]])
        vals.append(np.full(ok.sum(), -1.0))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def _solve(A: sp.csr_matrix, B: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    if pyamg is not None and A.shape[0] > 2000:
        ml = pyamg.ruge_stuben_solver(A)
        return np.stack(
            [ml.solve(B[:, c], x0=None if x0 is None else x0[:, c], tol=1e-5, maxiter=60) for c in range(B.shape[1])], 1
        )
    lu = sp.linalg.splu(A.tocsc())
    return lu.solve(B)


def robust_boundary(diff: np.ndarray, boundary: np.ndarray, sigma: float, k: float = 2.5, prior: np.ndarray | None = None) -> np.ndarray:
    """Replace outlying boundary differences by a smooth robust estimate.

    ``diff`` is (target - source) per pixel; only pixels where ``boundary`` is
    set are meaningful. Differences far from the local robust trend (e.g. a
    boundary pixel that lands on dark hair while the rest sit on skin) are
    down-weighted, which is what stops the Poisson membrane from smearing
    foreign colours into the pasted region.
    """
    b = boundary.astype(bool)
    if b.sum() < 8:
        return diff
    pw = np.ones(diff.shape[:2], np.float32) if prior is None else prior.astype(np.float32)
    trusted = b & (pw > 0.5)
    if trusted.sum() < max(8, 0.1 * b.sum()):
        trusted = b  # not enough trusted samples: fall back to all of them
    vals = diff[trusted]
    med = np.median(vals, axis=0)
    mad = np.median(np.abs(vals - med), axis=0) * 1.4826 + 1e-3
    w = np.exp(-0.5 * (((diff - med) / (k * mad)) ** 2).sum(-1)) * b * np.clip(pw, 0.02, 1.0)
    trend = normalized_blur(diff, w, sigma)
    # local agreement with the trend decides how much we trust each sample
    dev = np.sqrt(((diff - trend) ** 2).sum(-1))
    scale = np.median(dev[b]) * 1.4826 + 1e-3
    trust = np.exp(-0.5 * (dev / (k * scale)) ** 2)
    trust = trust * np.clip(pw, 0.0, 1.0)  # samples on non-skin (hair, shadows) follow the trusted trend
    out = diff * trust[..., None] + trend * (1 - trust[..., None])
    return np.where(b[..., None], out, diff)


def poisson_blend(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    mode: str = "normal",
    robust: bool = True,
    max_unknowns: int = 700_000,
    boundary_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Seamless cloning of ``src`` into ``dst`` inside ``mask`` (> 0.5).

    ``src`` and ``dst`` are aligned images of identical size. ``mode`` is
    ``"normal"`` (source gradients) or ``"mixed"`` (strongest gradient of the
    two, preserving destination texture through the inserted region).

    Returns the full-size result; outside the mask the result equals ``dst``.
    """
    inside_full = mask > 0.5
    if inside_full.sum() == 0:
        return dst.copy()
    y0, y1, x0, x1 = bbox_of(inside_full, pad=2)
    S = src[y0:y1, x0:x1].astype(np.float32)
    D = dst[y0:y1, x0:x1].astype(np.float32)
    inside = inside_full[y0:y1, x0:x1].copy()
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
    n = int(inside.sum())

    scale = 1.0
    if n > max_unknowns and mode == "normal":
        # The correction membrane is smooth: solve it at reduced resolution.
        scale = np.sqrt(max_unknowns / n)

    if mode == "normal":
        # result = src + harmonic membrane interpolating (dst - src) on the boundary
        boundary = dilate(inside.astype(np.uint8), 1).astype(bool) & ~inside
        diff = D - S
        if robust:
            bw = None if boundary_weight is None else boundary_weight[y0:y1, x0:x1]
            diff = robust_boundary(diff, boundary, sigma=max(3.0, 0.02 * max(S.shape[:2])), prior=bw)
        if scale < 1.0:
            h, w = inside.shape
            sh, sw = max(8, int(h * scale)), max(8, int(w * scale))
            ins_s = cv2.resize(inside.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
            bnd_s = dilate(ins_s.astype(np.uint8), 1).astype(bool) & ~ins_s
            # boundary values at low resolution via normalised convolution of the boundary samples
            diff_s = cv2.resize(normalized_blur(diff, boundary.astype(np.float32), 1.0 / scale), (sw, sh), interpolation=cv2.INTER_AREA)
            memb_s = _membrane(diff_s, ins_s, bnd_s)
            memb = cv2.resize(memb_s, (w, h), interpolation=cv2.INTER_CUBIC)
            out = S + memb
        else:
            out = S + _membrane(diff, inside, boundary)
    else:
        idx = -np.ones(inside.shape, np.int64)
        idx[inside] = np.arange(n)
        A = _laplacian_matrix(idx, inside)
        # guidance field: per-edge choice of the stronger gradient
        b = np.zeros((n, 3))
        ys, xs = np.nonzero(inside)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ys + dy, xs + dx
            gs = S[ys, xs] - S[ny, nx]
            gd = D[ys, xs] - D[ny, nx]
            g = np.where(np.abs(gd) > np.abs(gs), gd, gs)
            b += g
            outside = ~inside[ny, nx]
            b[outside] += D[ny[outside], nx[outside]]
        sol = _solve(A, b, x0=S[inside].astype(np.float64))
        out = D.copy()
        out[inside] = sol
    res = dst.copy()
    region = res[y0:y1, x0:x1]
    region[inside] = out[inside]
    return res


def _membrane(diff: np.ndarray, inside: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    n = int(inside.sum())
    out = np.zeros_like(diff)
    if n == 0:
        return out
    idx = -np.ones(inside.shape, np.int64)
    idx[inside] = np.arange(n)
    A = _laplacian_matrix(idx, inside)
    ys, xs = np.nonzero(inside)
    h, w = inside.shape
    b = np.zeros((n, diff.shape[2]))
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
        outside = ~inside[ny, nx]
        b[outside] += diff[ny[outside], nx[outside]]
    # initial guess: smooth extension of boundary values speeds up convergence
    x0 = normalized_blur(diff, boundary.astype(np.float32), max(2.0, 0.1 * np.sqrt(n)))[inside]
    out[inside] = _solve(A, b, x0=x0.astype(np.float64))
    return out


# --------------------------------------------------------------------------- multiband
def _levels_for(size: int, finest_transition: float) -> int:
    return int(np.clip(np.log2(max(size, 2) / 8.0), 1, 7))


def multiband_blend(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray, levels: int | None = None) -> np.ndarray:
    """Laplacian pyramid blending of ``fg`` over ``bg`` with soft ``mask``.

    Only the mask's bounding box (plus a margin) is processed; outside it the
    background is returned untouched.
    """
    support = mask > 1e-3
    if not support.any():
        return bg.copy()
    h, w = mask.shape
    pad = 32
    y0, y1, x0, x1 = bbox_of(support, pad=pad)
    F, B, M = fg[y0:y1, x0:x1].astype(np.float32), bg[y0:y1, x0:x1].astype(np.float32), mask[y0:y1, x0:x1].astype(np.float32)
    if levels is None:
        levels = _levels_for(min(F.shape[:2]), 1)
    # pad to a multiple of 2**levels (reflect avoids edge artefacts)
    m = 2**levels
    ph, pw = (-F.shape[0]) % m, (-F.shape[1]) % m
    Fp = cv2.copyMakeBorder(F, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    Bp = cv2.copyMakeBorder(B, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    Mp = cv2.copyMakeBorder(M, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    gf, gb, gm = [Fp], [Bp], [Mp]
    for _ in range(levels):
        gf.append(cv2.pyrDown(gf[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))
    out = gf[-1] * gm[-1][..., None] + gb[-1] * (1 - gm[-1][..., None])
    for i in range(levels - 1, -1, -1):
        size = (gf[i].shape[1], gf[i].shape[0])
        lf = gf[i] - cv2.pyrUp(gf[i + 1], dstsize=size)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size)
        mm = gm[i][..., None]
        out = cv2.pyrUp(out, dstsize=size) + lf * mm + lb * (1 - mm)
    out = out[: F.shape[0], : F.shape[1]]
    # Outside the mask's support the pyramid may have leaked tiny amounts; keep bg exact there.
    keep = gaussian((M > 1e-3).astype(np.float32), 2.0)
    keep = np.clip(keep * 4, 0, 1)
    keep[keep < 1e-3] = 0.0
    keep = keep[..., None]
    out = out * keep + B * (1 - keep)
    res = bg.copy()
    res[y0:y1, x0:x1] = np.clip(out, 0, 1)
    return res


def alpha_composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    return fg * a + bg * (1 - a)
