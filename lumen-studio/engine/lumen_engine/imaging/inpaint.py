"""Image completion (inpainting).

Three local back-ends, used in this order of preference by :func:`inpaint`:

* **LaMa** (optional model, ``lama``) - a learned large-mask inpainting
  network (Fourier convolutions) that continues structure and texture of
  the surroundings into big holes; run on square context windows at 512 px.
* **MI-GAN** (optional model, ``migan``) - a much lighter learned inpainting
  network (pipeline export with its own pre/post-processing, any input size).
* **PatchMatch exemplar completion** (always available) - multi-scale
  expectation-maximisation completion (Wexler, Shechtman & Irani 2007) using
  PatchMatch nearest-neighbour fields (Barnes et al. 2009). It synthesises
  missing areas from real texture found in the surrounding context, so the
  fill keeps the grain, texture and structure of the photo instead of a blur.
  Extra channels can be completed jointly, *guide* channels that are known
  everywhere (e.g. the depth of the hidden surface) steer which texture is
  copied where, and the source area can be restricted (e.g. "only copy from
  the background").

The learned back-ends are optional downloads. Their ONNX signature is
inspected at run time (input names, NCHW or NHWC layout, uint8 or float
tensors, fixed or free spatial size, 0..1 or 0..255 output), every output is
validated, and any failure falls back to PatchMatch - the method that was
actually used is returned so callers can report it.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numba import njit, prange

from .filters import bbox_of, dilate

# --------------------------------------------------------------------------- numba kernels


@njit(cache=True, fastmath=True)
def _pdist(img, py, px, qy, qx, r, best):
    d = 0.0
    C = img.shape[2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            for c in range(C):
                t = img[py + dy, px + dx, c] - img[qy + dy, qx + dx, c]
                d += t * t
        if d > best:
            return d
    return d


@njit(cache=True, fastmath=True)
def _pdist_known(img, known, py, px, qy, qx, r, best):
    d = 0.0
    cnt = 0
    C = img.shape[2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if known[py + dy, px + dx]:
                cnt += 1
                for c in range(C):
                    t = img[py + dy, px + dx, c] - img[qy + dy, qx + dx, c]
                    d += t * t
    if cnt == 0:
        return 1e30
    return d / cnt


@njit(cache=True)
def _onion_init(img, known, hole, valid_ys, valid_xs, r):
    """Onion-peel initialisation (Newson et al. 2014) by exhaustive search.

    Fills the hole layer by layer from its boundary inwards; each pixel takes
    the centre of the best-matching source patch, comparing only already
    known pixels. Used at the coarsest pyramid level where it is cheap.
    """
    H, W = known.shape
    C = img.shape[2]
    nv = valid_ys.shape[0]
    remaining = 0
    for y in range(H):
        for x in range(W):
            if hole[y, x] and not known[y, x]:
                remaining += 1
    while remaining > 0:
        # collect the current front (unknown pixels with a known 4-neighbour)
        fy = np.empty(remaining, np.int64)
        fx = np.empty(remaining, np.int64)
        nf = 0
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if hole[y, x] and not known[y, x]:
                    if known[y - 1, x] or known[y + 1, x] or known[y, x - 1] or known[y, x + 1]:
                        fy[nf] = y
                        fx[nf] = x
                        nf += 1
        if nf == 0:
            break
        vals = np.empty((nf, C), img.dtype)
        for k in range(nf):
            best = 1e30
            bi = 0
            for i in range(nv):
                d = _pdist_known(img, known, fy[k], fx[k], valid_ys[i], valid_xs[i], r, best)
                if d < best:
                    best = d
                    bi = i
            for c in range(C):
                vals[k, c] = img[valid_ys[bi], valid_xs[bi], c]
        for k in range(nf):
            for c in range(C):
                img[fy[k], fx[k], c] = vals[k, c]
            known[fy[k], fx[k]] = True
        remaining -= nf


@njit(cache=True)
def _random_valid(valid_ys, valid_xs):
    i = np.random.randint(0, valid_ys.shape[0])
    return valid_ys[i], valid_xs[i]


@njit(cache=True)
def _patchmatch(img, valid, tys, txs, nnf, dist, r, iters, seed):
    np.random.seed(seed)
    H, W = valid.shape
    n = tys.shape[0]
    maxr = max(H, W)
    # recompute distances for the current image estimate
    for i in range(n):
        dist[i] = _pdist(img, tys[i], txs[i], nnf[i, 0], nnf[i, 1], r, 1e30)
    # index map for propagation
    tindex = -np.ones((H, W), np.int64)
    for i in range(n):
        tindex[tys[i], txs[i]] = i
    for it in range(iters):
        if it % 2 == 0:
            start, stop, step = 0, n, 1
        else:
            start, stop, step = n - 1, -1, -1
        for k in range(start, stop, step):
            py, px = tys[k], txs[k]
            best = dist[k]
            by, bx = nnf[k, 0], nnf[k, 1]
            # propagation from the already-visited neighbours
            for nb in range(2):
                if step == 1:
                    ny, nx = (py, px - 1) if nb == 0 else (py - 1, px)
                    oy, ox = (0, 1) if nb == 0 else (1, 0)
                else:
                    ny, nx = (py, px + 1) if nb == 0 else (py + 1, px)
                    oy, ox = (0, -1) if nb == 0 else (-1, 0)
                if ny < 0 or nx < 0 or ny >= H or nx >= W:
                    continue
                j = tindex[ny, nx]
                if j < 0:
                    continue
                cy, cx = nnf[j, 0] + oy, nnf[j, 1] + ox
                if cy < 0 or cx < 0 or cy >= H or cx >= W or not valid[cy, cx]:
                    continue
                d = _pdist(img, py, px, cy, cx, r, best)
                if d < best:
                    best, by, bx = d, cy, cx
            # random search with exponentially shrinking radius
            rad = maxr
            while rad >= 1:
                cy = by + np.random.randint(-rad, rad + 1)
                cx = bx + np.random.randint(-rad, rad + 1)
                if 0 <= cy < H and 0 <= cx < W and valid[cy, cx]:
                    d = _pdist(img, py, px, cy, cx, r, best)
                    if d < best:
                        best, by, bx = d, cy, cx
                rad //= 2
            dist[k] = best
            nnf[k, 0] = by
            nnf[k, 1] = bx


@njit(cache=True, parallel=True)
def _patchmatch_par(img, valid, tys, txs, nnf, dist, r, iters, seed, max_radius):
    """Parallel (Jacobi-style) PatchMatch: every target reads its neighbours'
    matches from the previous iteration, so all targets update concurrently."""
    H, W = valid.shape
    n = tys.shape[0]
    tindex = -np.ones((H, W), np.int64)
    for i in range(n):
        tindex[tys[i], txs[i]] = i
    for i in prange(n):
        dist[i] = _pdist(img, tys[i], txs[i], nnf[i, 0], nnf[i, 1], r, 1e30)
    for it in range(iters):
        old = nnf.copy()
        for k in prange(n):
            np.random.seed(seed + it * 7919 + k)
            py, px = tys[k], txs[k]
            best = dist[k]
            by, bx = old[k, 0], old[k, 1]
            for nb in range(8):
                if nb == 0:
                    oy, ox = 0, -1
                elif nb == 1:
                    oy, ox = 0, 1
                elif nb == 2:
                    oy, ox = -1, 0
                elif nb == 3:
                    oy, ox = 1, 0
                elif nb == 4:
                    oy, ox = 0, -2
                elif nb == 5:
                    oy, ox = 0, 2
                elif nb == 6:
                    oy, ox = -2, 0
                else:
                    oy, ox = 2, 0
                ny, nx = py + oy, px + ox
                if ny < 0 or nx < 0 or ny >= H or nx >= W:
                    continue
                j = tindex[ny, nx]
                if j < 0:
                    continue
                cy, cx = old[j, 0] - oy, old[j, 1] - ox
                if cy < 0 or cx < 0 or cy >= H or cx >= W or not valid[cy, cx]:
                    continue
                d = _pdist(img, py, px, cy, cx, r, best)
                if d < best:
                    best, by, bx = d, cy, cx
            rad = max_radius
            while rad >= 1:
                cy = by + np.random.randint(-rad, rad + 1)
                cx = bx + np.random.randint(-rad, rad + 1)
                if 0 <= cy < H and 0 <= cx < W and valid[cy, cx]:
                    d = _pdist(img, py, px, cy, cx, r, best)
                    if d < best:
                        best, by, bx = d, cy, cx
                rad //= 2
            dist[k] = best
            nnf[k, 0] = by
            nnf[k, 1] = bx


@njit(cache=True, fastmath=True)
def _vote(img, hole, tys, txs, nnf, dist, r, sigma2, out):
    H, W, C = img.shape
    acc = np.zeros((H, W, C), np.float64)
    wsum = np.zeros((H, W), np.float64)
    for k in range(tys.shape[0]):
        py, px = tys[k], txs[k]
        qy, qx = nnf[k, 0], nnf[k, 1]
        w = math.exp(-dist[k] / (2.0 * sigma2))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y, x = py + dy, px + dx
                if hole[y, x]:
                    wsum[y, x] += w
                    for c in range(C):
                        acc[y, x, c] += w * img[qy + dy, qx + dx, c]
    for y in range(H):
        for x in range(W):
            if hole[y, x]:
                if wsum[y, x] > 0:
                    for c in range(C):
                        out[y, x, c] = acc[y, x, c] / wsum[y, x]
                else:
                    for c in range(C):
                        out[y, x, c] = img[y, x, c]
            else:
                for c in range(C):
                    out[y, x, c] = img[y, x, c]


@njit(cache=True)
def _vote_best(img, hole, tys, txs, nnf, dist, r, out):
    """Each hole pixel copies from the covering patch with the lowest error."""
    H, W, C = img.shape
    bestd = np.full((H, W), 1e30)
    for k in range(tys.shape[0]):
        py, px = tys[k], txs[k]
        qy, qx = nnf[k, 0], nnf[k, 1]
        d = dist[k]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y, x = py + dy, px + dx
                if hole[y, x] and d < bestd[y, x]:
                    bestd[y, x] = d
                    for c in range(C):
                        out[y, x, c] = img[qy + dy, qx + dx, c]
    for y in range(H):
        for x in range(W):
            if not hole[y, x] or bestd[y, x] >= 1e30:
                for c in range(C):
                    out[y, x, c] = img[y, x, c]


@njit(cache=True)
def _init_nnf(tys, txs, valid_ys, valid_xs, seed):
    np.random.seed(seed)
    n = tys.shape[0]
    nnf = np.empty((n, 2), np.int64)
    for i in range(n):
        y, x = _random_valid(valid_ys, valid_xs)
        nnf[i, 0] = y
        nnf[i, 1] = x
    return nnf


@njit(cache=True)
def _upsample_nnf(tys, txs, prev_map_y, prev_map_x, valid, valid_ys, valid_xs, r, seed):
    """prev_map_*: coarse-level NNF stored as images (-1 where none)."""
    np.random.seed(seed)
    H, W = valid.shape
    ch, cw = prev_map_y.shape
    n = tys.shape[0]
    nnf = np.empty((n, 2), np.int64)
    for i in range(n):
        yu, xu = tys[i] - r, txs[i] - r  # unpadded fine coordinates
        cy, cx = min(yu // 2, ch - 1), min(xu // 2, cw - 1)
        my, mx = prev_map_y[cy, cx], prev_map_x[cy, cx]
        ok = False
        if my >= 0:
            qy, qx = my * 2 + (yu - cy * 2) + r, mx * 2 + (xu - cx * 2) + r
            if 0 <= qy < H and 0 <= qx < W and valid[qy, qx]:
                nnf[i, 0], nnf[i, 1] = qy, qx
                ok = True
        if not ok:
            a, b = _random_valid(valid_ys, valid_xs)
            nnf[i, 0], nnf[i, 1] = a, b
    return nnf


# --------------------------------------------------------------------------- python driver
def _resize(img: np.ndarray, size: tuple[int, int], interp: int) -> np.ndarray:
    """cv2.resize for any channel count (OpenCV limits some modes to 4)."""
    if img.ndim == 2 or img.shape[2] <= 4:
        out = cv2.resize(img, size, interpolation=interp)
        return out if img.ndim == 2 or out.ndim == 3 else out[..., None]
    return np.concatenate([_resize(img[..., i : i + 4], size, interp) for i in range(0, img.shape[2], 4)], -1)


def _down(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return _resize(img, (max(1, (w + 1) // 2), max(1, (h + 1) // 2)), cv2.INTER_AREA)


def _down_mask(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    ph, pw = h % 2, w % 2
    mp = np.pad(m, ((0, ph), (0, pw)), mode="edge")
    return mp.reshape(mp.shape[0] // 2, 2, mp.shape[1] // 2, 2).max(axis=(1, 3))


def _smooth_fill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    out = img.copy()
    hm = hole.astype(np.uint8)
    for c in range(img.shape[2]):
        ch = np.ascontiguousarray(img[..., c]).astype(np.float32)
        out[..., c] = cv2.inpaint(ch, hm, 3, cv2.INPAINT_TELEA)
    return out


def _prefill_gray(g: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Smooth guess for the hole of a grey image (normalised convolution at growing scales).

    Texture features are computed before the completion starts; pre-filling
    stops the content that is being replaced (e.g. a person) from leaking into
    the features of the pixels around the hole.
    """
    known = (~hole).astype(np.float32)
    if not known.any():
        return np.full_like(g, 0.5)
    out = g.copy()
    todo = hole.copy()
    gk = (g * known).astype(np.float32)
    for sig in (2.0, 5.0, 12.0, 30.0, 80.0):
        den = cv2.GaussianBlur(known, (0, 0), sig)
        num = cv2.GaussianBlur(gk, (0, 0), sig)
        ok = todo & (den > 0.02)
        out[ok] = num[ok] / den[ok]
        todo &= ~ok
        if not todo.any():
            break
    if todo.any():
        out[todo] = float(g[~hole].mean())
    return out


def _valid_source_count(hole: np.ndarray, source: np.ndarray, r: int) -> int:
    h, w = hole.shape
    if h <= 2 * r or w <= 2 * r:
        return 0
    bad = cv2.dilate(((~source) | hole).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    return int((~bad[r : h - r, r : w - r]).sum())


def patchmatch_inpaint(
    image: np.ndarray,
    hole: np.ndarray,
    source: np.ndarray | None = None,
    patch_size: int = 7,
    seed: int = 1,
    progress=None,
    quality: str = "high",
    final_vote: str = "weighted",
    guide: np.ndarray | None = None,
    layout: float = 0.0,
) -> np.ndarray:
    """Complete ``hole`` in ``image`` (H x W x C float) from ``source`` pixels.

    ``layout`` > 0 adds a colour-layout guide: a smooth extrapolation of the
    trusted surroundings (``source`` only) into the hole, weighted by
    ``layout``; patches whose low-frequency colour does not fit the
    surroundings (a dark smear on a light wall) are penalised.

    ``source`` (bool) restricts which pixels may be copied from (default:
    everything that is not hole). ``guide`` (H x W or H x W x G) holds extra
    channels that are *known everywhere*, including inside the hole (for
    example the depth of the surface hidden behind a foreground object):
    they take part in patch matching but are never synthesised, so they steer
    which texture is copied where. Returns a full-size array where only hole
    pixels changed.
    """
    hole = hole.astype(bool)
    if not hole.any():
        return image.copy()
    img = image.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    H0, W0, C0 = img.shape
    r = patch_size // 2
    # Texture features (Newson et al. 2014): local mean absolute gradients.
    # They make the patch distance prefer textured candidates over smooth
    # ones, which prevents the classic "blurry mush" of EM completion.
    gray = img[..., :3].mean(-1) if C0 >= 3 else img[..., 0]
    gray = _prefill_gray(gray, hole)
    tex = np.stack([np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))], -1)
    tex = cv2.blur(tex, (patch_size, patch_size)) * 0.75
    parts = [img, tex.astype(np.float32)]
    G = 0
    if layout > 0 and C0 >= 3:
        trusted = ~hole if source is None else (source.astype(bool) & ~hole)
        lay = np.dstack([_prefill_gray(img[..., c].copy(), ~trusted) for c in range(3)]).astype(np.float32)
        sig = max(2.0, 0.02 * max(H0, W0), 1.5 * patch_size)
        lay = cv2.GaussianBlur(lay, (0, 0), sig) * float(layout)
        guide = lay if guide is None else np.concatenate([guide.astype(np.float32) if guide.ndim == 3 else guide.astype(np.float32)[..., None], lay], -1)
    if guide is not None:
        gd = guide.astype(np.float32)
        if gd.ndim == 2:
            gd = gd[..., None]
        G = gd.shape[2]
        parts.append(gd)
    img = np.concatenate(parts, -1)
    C = img.shape[2]
    src_allowed = ~hole if source is None else (source.astype(bool) & ~hole)
    if src_allowed.sum() < (patch_size * 4) ** 2 // 4:
        # not enough context: fall back to smooth fill rather than garbage
        return _smooth_fill(image.reshape(H0, W0, C0).astype(np.float32), hole).reshape(image.shape)

    # work on a context window around the hole for locality and speed
    hy0, hy1, hx0, hx1 = bbox_of(hole)
    ext = int(max(hy1 - hy0, hx1 - hx0) * 1.6 + 64)
    y0, y1, x0, x1 = bbox_of(hole, pad=ext)
    I = img[y0:y1, x0:x1].copy()
    Hm = hole[y0:y1, x0:x1]
    S = src_allowed[y0:y1, x0:x1]

    # pyramid depth: coarsest hole extent ~ 2-3 patches
    extent = max(hy1 - hy0, hx1 - hx0)
    levels = 0
    while extent / (2 ** (levels + 1)) >= patch_size * 3 and min(I.shape[:2]) / (2 ** (levels + 1)) > patch_size * 10:
        levels += 1
    pyr_I, pyr_H, pyr_S = [I], [Hm], [S]
    for _ in range(levels):
        pyr_I.append(_down(pyr_I[-1]))
        pyr_H.append(_down_mask(pyr_H[-1]))
        pyr_S.append(_down_mask((~pyr_S[-1]).astype(np.uint8)).astype(bool) == False)  # noqa: E712
    # a restricted source (e.g. thin strips of background between people) vanishes at
    # coarse levels: stop the pyramid where too few complete source patches remain
    while levels > 0:
        hh = pyr_H[levels].astype(bool)
        if _valid_source_count(hh, pyr_S[levels].astype(bool) & ~hh, r) >= max(64, 0.05 * hh.sum()):
            break
        levels -= 1
    pyr_I, pyr_H, pyr_S = pyr_I[: levels + 1], pyr_H[: levels + 1], pyr_S[: levels + 1]

    em_coarse, em_fine = (8, 3) if quality == "high" else (5, 2)
    prev = None
    total_levels = levels + 1
    for li in range(levels, -1, -1):
        L = pyr_I[li].copy()
        Hh = pyr_H[li].astype(bool)
        Sv = pyr_S[li].astype(bool) & ~Hh
        h, w = Hh.shape
        # pad by r so every target patch is inside the array
        Lp = np.pad(L, ((r, r), (r, r), (0, 0)), mode="reflect")
        Gp = Lp[..., C - G :].copy() if G else None

        def fix(a, Gp=Gp):
            if Gp is not None:
                a[..., C - G :] = Gp  # guide channels are known: never synthesised
            return a

        Hp = np.pad(Hh, r, mode="constant")
        valid = np.zeros_like(Hp)
        # a source patch is valid when it lies inside the original window and contains no hole
        hole_or_bad = (~Sv) | Hh
        bad = cv2.dilate(hole_or_bad.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        inner = np.zeros_like(Hh)
        inner[r : h - r, r : w - r] = True if h > 2 * r and w > 2 * r else False
        valid[r : r + h, r : r + w] = (~bad) & inner
        vys, vxs = np.nonzero(valid)
        if len(vys) == 0:
            return _smooth_fill(image.reshape(H0, W0, C0).astype(np.float32), hole).reshape(image.shape)
        target = cv2.dilate(Hh.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        tys, txs = np.nonzero(target)
        tys, txs = tys + r, txs + r
        if prev is None:
            Lp[Hp] = _smooth_fill(Lp, Hp)[Hp]
            fix(Lp)
            if len(vys) * len(tys) < 2.5e7:
                known = ~Hp
                known[:r, :] = known[-r:, :] = False
                known[:, :r] = known[:, -r:] = False
                known &= ~Hp
                known[r : r + h, r : r + w] |= ~Hh
                _onion_init(Lp, known.copy(), Hp, vys.astype(np.int64), vxs.astype(np.int64), r)
                fix(Lp)
            nnf = _init_nnf(tys, txs, vys, vxs, seed)
        else:
            pmy, pmx = prev
            nnf = _upsample_nnf(tys, txs, pmy, pmx, valid, vys.astype(np.int64), vxs.astype(np.int64), r, seed + li)
            # initial estimate from upsampled coarse result
            up = _resize(prev_img, (w, h), cv2.INTER_LINEAR).reshape(h, w, C)
            Lp[r : r + h, r : r + w][Hh] = up[Hh]
            fix(Lp)
            dist0 = np.zeros(len(tys))
            out = np.empty_like(Lp)
            _patchmatch_par(Lp, valid, tys, txs, nnf, dist0, r, 2, seed + 100 + li, max(16, max(h, w) // 4))
            _vote(Lp, Hp, tys, txs, nnf, dist0, r, max(np.percentile(dist0, 75), 1e-6), out)
            Lp = fix(out)
        dist = np.zeros(len(tys))
        n_em = em_coarse if li == levels else max(em_fine, int(em_coarse * (li + 1) / total_levels))
        for em in range(n_em):
            if li == levels:
                _patchmatch(Lp, valid, tys, txs, nnf, dist, r, 4, seed + em * 7 + li * 131)
            else:
                # finer levels start from an upsampled field: parallel updates with a local search radius
                _patchmatch_par(Lp, valid, tys, txs, nnf, dist, r, 4 if li > 0 else 3, seed + em * 7 + li * 131, max(16, max(h, w) // 4))
            out = np.empty_like(Lp)
            if li == 0 and em == n_em - 1 and final_vote == "best":
                # final reconstruction: copy real pixels (no averaging blur)
                _vote_best(Lp, Hp, tys, txs, nnf, dist, r, out)
            else:
                q = 75 if em < n_em - 1 else (40 if li > 0 else 15)
                _vote(Lp, Hp, tys, txs, nnf, dist, r, max(np.percentile(dist, q), 1e-6), out)
            Lp = fix(out)
        if progress:
            progress((levels - li + 1) / total_levels)
        # store NNF as image for the next level
        pmy = -np.ones((h, w), np.int64)
        pmx = -np.ones((h, w), np.int64)
        pmy[tys - r, txs - r] = nnf[:, 0] - r
        pmx[tys - r, txs - r] = nnf[:, 1] - r
        prev = (pmy, pmx)
        prev_img = Lp[r : r + h, r : r + w].copy()

    result = img[..., :C0].copy()
    win = result[y0:y1, x0:x1]
    win[Hm] = prev_img[..., :C0][Hm]
    return result if image.ndim == 3 else result[..., 0]


def _add_grain(out: np.ndarray, reference: np.ndarray, hole: np.ndarray, extent: int, channels: int = 3, seed: int = 3) -> np.ndarray:
    """Give a (resampled, hence too smooth) fill the sensor grain measured around the hole."""
    from .analysis import noise_sigma, synth_grain

    if channels < 3 or reference.ndim != 3 or reference.shape[2] < 3:
        return out
    ring = dilate(hole.astype(np.uint8), max(4, int(extent * 0.05))).astype(bool) & ~hole
    sig = noise_sigma(np.clip(reference[..., :3], 0, 1), ring.astype(np.float32))
    if sig > 0.4 / 255:
        g = synth_grain(hole.shape, sig, np.random.default_rng(seed))
        for c0 in range(0, channels - 2, 3):
            sl = out[..., c0 : c0 + 3]
            sl[hole] = np.clip(sl[hole] + g[hole], 0, 1)
    return out


def scaled_patchmatch(
    image: np.ndarray,
    hole: np.ndarray,
    source: np.ndarray | None = None,
    max_extent: int = 640,
    progress=None,
    guide: np.ndarray | None = None,
    grain: bool = True,
    grain_channels: int | None = None,
    layout: float = 0.0,
) -> np.ndarray:
    """PatchMatch completion with a bounded working size.

    Very large holes are completed on a downscaled copy (hole extent about
    ``max_extent`` px), upsampled, and given back sensor grain measured
    around the hole so the fill does not look smoother than the photo
    (``grain_channels``: number of leading colour channels, default 3).
    ``guide`` channels (known everywhere) steer the matching, see
    :func:`patchmatch_inpaint`. Known pixels are never modified.
    """
    hole = hole.astype(bool)
    if not hole.any():
        return image.copy()
    y0, y1, x0, x1 = bbox_of(hole)
    extent = max(y1 - y0, x1 - x0)
    if extent <= max_extent * 1.25:
        return patchmatch_inpaint(image, hole, source=source, progress=progress, guide=guide, layout=layout)
    s = max_extent / extent
    H, W = hole.shape
    sw, sh = max(8, int(W * s)), max(8, int(H * s))
    small = _resize(image.astype(np.float32), (sw, sh), cv2.INTER_AREA).reshape(sh, sw, -1)
    hs = cv2.resize(hole.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.01
    ss = None if source is None else cv2.resize(source.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.99
    gs = None
    if guide is not None:
        g = guide.astype(np.float32)
        gs = _resize(g if g.ndim == 3 else g[..., None], (sw, sh), cv2.INTER_AREA).reshape(sh, sw, -1)
    filled = patchmatch_inpaint(small if image.ndim == 3 else small[..., 0], hs, source=ss, progress=progress, guide=gs, layout=layout)
    up = _resize(filled.astype(np.float32), (W, H), cv2.INTER_CUBIC).reshape(image.shape)
    out = image.copy()
    out[hole] = up[hole]
    if grain and image.ndim == 3:
        nch = grain_channels if grain_channels is not None else (3 if image.shape[2] >= 3 else 0)
        out = _add_grain(out, image, hole, extent, channels=nch)
    return out


# --------------------------------------------------------------------------- learned models (LaMa, MI-GAN)
LEARNED_MODELS = ("lama", "migan")  # order of preference

# Mask conventions of the supported exports:
#   Carve/LaMa-ONNX     image float [1,3,512,512] in 0..1, mask float [1,1,512,512], 1 = hole
#   MI-GAN pipeline v2  image uint8 RGB, mask uint8 grey, 255 = known / 0 = hole, any size
_HOLE_IS_ONE = {"lama": True, "migan": False}
_TYPES = {"tensor(uint8)": np.uint8, "tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}


class InpaintModelError(RuntimeError):
    """A learned inpainting model could not be run or returned unusable output."""


def available_learned() -> list[str]:
    """Installed learned inpainting models, in order of preference."""
    from ..models import manager

    out = []
    for mid in LEARNED_MODELS:
        try:
            if manager.is_installed(mid):
                out.append(mid)
        except Exception:  # noqa: BLE001 - an unknown/broken entry is simply unavailable
            pass
    return out


def _lama_available() -> bool:
    return "lama" in available_learned()


class _Port:
    """One model input: name, tensor layout, element type and fixed spatial size (if any)."""

    def __init__(self, inp):
        shp = list(inp.shape)
        self.name = inp.name

        def ch(v):
            return isinstance(v, int) and v in (1, 3)

        rank = len(shp)
        if rank == 4:
            if ch(shp[3]) and not ch(shp[1]):
                self.layout, hw, c = "NHWC", shp[1:3], shp[3]
            else:
                self.layout, hw, c = "NCHW", shp[2:4], shp[1]
        elif rank == 3:
            if ch(shp[2]) and not ch(shp[0]):
                self.layout, hw, c = "HWC", shp[0:2], shp[2]
            else:
                self.layout, hw, c = "CHW", shp[1:3], shp[0]
        elif rank == 2:
            self.layout, hw, c = "HW", shp, 1
        else:
            raise InpaintModelError(f"unsupported rank {rank} of input '{inp.name}'")
        self.channels = c if isinstance(c, int) else None
        self.size = tuple(hw) if all(isinstance(v, int) and v > 0 for v in hw) else None
        self.dtype = _TYPES.get(inp.type)
        if self.dtype is None:
            raise InpaintModelError(f"unsupported element type {inp.type} of input '{inp.name}'")

    def pack(self, hwc: np.ndarray) -> np.ndarray:
        a = hwc
        if self.layout in ("NCHW", "CHW"):
            a = a.transpose(2, 0, 1)
        elif self.layout == "HW":
            a = a[..., 0]
        if self.layout in ("NCHW", "NHWC"):
            a = a[None]
        return np.ascontiguousarray(a.astype(self.dtype))


def _model_ports(sess) -> tuple[_Port, _Port]:
    inputs = sess.get_inputs()
    if len(inputs) != 2:
        raise InpaintModelError(f"expected two inputs (image, mask), the model has {len(inputs)}")
    ports = [_Port(i) for i in inputs]
    masks = [p for p in ports if "mask" in p.name.lower()] or [p for p in ports if p.channels == 1]
    images = [p for p in ports if p not in masks]
    if len(masks) != 1 or len(images) != 1:
        images, masks = [ports[0]], [ports[1]]
    if images[0].channels not in (None, 3):
        raise InpaintModelError(f"image input '{images[0].name}' has {images[0].channels} channels")
    return images[0], masks[0]


def _decode_output(out) -> np.ndarray:
    a = np.asarray(out)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        raise InpaintModelError(f"unexpected output shape {np.shape(out)}")
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    elif a.shape[-1] != 3:
        raise InpaintModelError(f"unexpected output shape {np.shape(out)}")
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    f = a.astype(np.float32)
    if not np.isfinite(f).all():
        raise InpaintModelError("output contains NaN or Inf")
    if float(f.max()) > 2.0:  # 0..255 export
        f = f / 255.0
    elif float(f.min()) < -0.25:  # -1..1 export
        f = (f + 1.0) * 0.5
    return np.clip(f, 0.0, 1.0)


def _run_model(sess, model_id: str, crop: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Inpaint one crop (h x w x 3 in 0..1, hole pixels already neutral); returns h x w x 3."""
    ip, mp = _model_ports(sess)
    h, w = hole.shape
    size = ip.size or mp.size
    if size is not None:
        th, tw = int(size[0]), int(size[1])
    else:  # free-size model: keep the crop, aligned to the network stride
        th, tw = max(16, int(round(h / 8)) * 8), max(16, int(round(w / 8)) * 8)
    if (th, tw) != (h, w):
        crop = crop.copy()
        crop[hole] = 0.0  # what is being replaced must not leak into the resampled context
        x = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA if th < h else cv2.INTER_CUBIC)
        # coverage, not nearest-neighbour sampling: thin hole strips must survive the downscale
        m = cv2.resize(hole.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA) > 0.01
        m = dilate(m.astype(np.uint8), 1) > 0  # resampling must never shrink the hole
    else:
        x, m = crop, hole
    x = np.clip(x, 0, 1).astype(np.float32)
    x[m] = 0.0
    xi = (x * 255 + 0.5).astype(np.uint8) if ip.dtype == np.uint8 else x
    mv = m if _HOLE_IS_ONE.get(model_id, True) else ~m
    mi = (mv.astype(np.float32) * (255.0 if mp.dtype == np.uint8 else 1.0))[..., None]
    out = sess.run(None, {ip.name: ip.pack(xi), mp.name: mp.pack(mi)})[0]
    y = _decode_output(out)
    if y.shape[:2] != (th, tw):
        if y.shape[:2] == (h, w):  # pipelines that resize back themselves
            th, tw = h, w
            x, m = np.clip(crop, 0, 1).astype(np.float32), hole
        else:
            raise InpaintModelError(f"output size {y.shape[:2]} does not match the input size {(th, tw)}")
    # sanity checks: the known area must be reproduced, the hole must not come back black
    known = ~m
    if known.any():
        err = float(np.abs(y[known] - x[known]).mean())
        if err > 0.2:
            raise InpaintModelError(f"output does not reproduce the known image area (mean error {err:.2f})")
        if m.any() and float(y[m].mean()) < 0.03 and float(x[known].mean()) > 0.12:
            raise InpaintModelError("output is black inside the hole")
    if (th, tw) != (h, w):
        y = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC if th < h else cv2.INTER_AREA)
    return np.clip(y, 0, 1)


def _windows(hole: np.ndarray, min_side: int = 256) -> list[tuple[int, int, int]]:
    """Square context windows (y, x, side) that together cover every hole pixel.

    A compact hole gets one window about 1.6x its extent; a hole spread over
    the whole image (e.g. the out-of-frame margins) gets overlapping squares
    as tall as the image along its long axis.
    """
    H, W = hole.shape
    y0, y1, x0, x1 = bbox_of(hole)
    E = max(y1 - y0, x1 - x0)
    short, long_ = min(H, W), max(H, W)
    S = int(min(max(E * 1.6 + 32, min_side), long_))
    if S <= short:
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        ya = int(np.clip(cy - S // 2, 0, H - S))
        xa = int(np.clip(cx - S // 2, 0, W - S))
        return [(ya, xa, S)]
    S = short
    n = 1 if long_ <= S else int(math.ceil((long_ - S) / (0.6 * S))) + 1
    starts = np.linspace(0, long_ - S, n).astype(int) if n > 1 else np.array([0])
    wins = [(0, int(st), S) if W >= H else (int(st), 0, S) for st in starts]
    return [wd for wd in wins if hole[wd[0] : wd[0] + wd[2], wd[1] : wd[1] + wd[2]].any()]


LEARNED_MAX_PIXELS = 640 * 640  # pixel budget of one model run (any-size exports); larger crops are downscaled
_OUT_SCALE = {"lama": 255.0}  # known output ranges (LaMa exports return 0..255 floats)


def _crops(hole: np.ndarray, ctx: float = 0.6, min_ctx: int = 24, max_crops: int = 6, budget: int | None = None) -> list[tuple[int, int, int, int]]:
    """Context rectangles (y0, y1, x0, x1) that together cover every hole pixel.

    Each connected hole region gets its bounding box grown by ``ctx`` times its
    extent on every side (at least ``min_ctx`` px), so the model sees enough of
    the surroundings to continue structure; overlapping rectangles are merged.
    Many scattered regions, or rectangles covering most of the image, become one
    rectangle around everything.
    """
    H, W = hole.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    boxes = [[int(s[1]), int(s[1] + s[3]), int(s[0]), int(s[0] + s[2])] for s in stats[1:n]]

    def grow(b):
        e = max(b[1] - b[0], b[3] - b[2])
        c = int(max(min_ctx, ctx * e))
        return [max(0, b[0] - c), min(H, b[1] + c), max(0, b[2] - c), min(W, b[3] + c)]

    merged = True
    while merged and len(boxes) > 1:
        merged = False
        rects = [grow(b) for b in boxes]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = rects[i], rects[j]
                if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]:
                    bi, bj = boxes[i], boxes[j]
                    boxes[i] = [min(bi[0], bj[0]), max(bi[1], bj[1]), min(bi[2], bj[2]), max(bi[3], bj[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    rects = [grow(b) for b in boxes]
    areas = [(r[1] - r[0]) * (r[3] - r[2]) for r in rects]
    # cost of separate runs (each capped by the pixel budget) vs. one (downscaled) run around everything
    costly = budget is not None and len(rects) > 1 and sum(min(a, budget) for a in areas) > 1.6 * budget
    if len(rects) > max_crops or sum(areas) > 0.6 * H * W or costly:
        return [tuple(grow([min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes)]))]
    return [tuple(r) for r in rects]


def _run_any_size(sess, model_id: str, crop: np.ndarray, hole: np.ndarray, max_pixels: int) -> np.ndarray:
    """Inpaint one crop with an any-size model: at its own resolution when it fits the pixel
    budget (no resampling at all), else downscaled; padded to the network stride."""
    ip, mp = _model_ports(sess)
    h, w = hole.shape
    sc = min(1.0, math.sqrt(max_pixels / float(h * w)))
    if sc < 0.98:
        tw, th = max(16, int(round(w * sc))), max(16, int(round(h * sc)))
        x = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        m = cv2.resize(hole.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA) > 0.01
        m = dilate(m.astype(np.uint8), 1) > 0  # resampling must never shrink the hole
    else:
        th, tw, x, m = h, w, crop, hole
    # stride 16: LaMa's Fourier units need an even feature width (W/8) - other sizes fail in ONNX Runtime
    ph, pw = (-th) % 16, (-tw) % 16
    x = np.clip(x, 0, 1).astype(np.float32)
    x[m] = 0.0
    xp = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    mpad = np.pad(m, ((0, ph), (0, pw)), mode="symmetric")
    xi = (xp * 255 + 0.5).astype(np.uint8) if ip.dtype == np.uint8 else xp
    mv = mpad if _HOLE_IS_ONE.get(model_id, True) else ~mpad
    mi = (mv.astype(np.float32) * (255.0 if mp.dtype == np.uint8 else 1.0))[..., None]
    out = sess.run(None, {ip.name: ip.pack(xi), mp.name: mp.pack(mi)})[0]
    scale = _OUT_SCALE.get(model_id)
    if scale is not None and np.asarray(out).dtype != np.uint8:
        a = np.asarray(out, np.float32)
        while a.ndim > 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
            a = a.transpose(1, 2, 0)
        if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
            raise InpaintModelError(f"unexpected output {np.shape(out)}")
        y = np.clip(a / scale, 0.0, 1.0)
    else:
        y = _decode_output(out)
    if y.shape[:2] != xp.shape[:2]:
        raise InpaintModelError(f"output size {y.shape[:2]} does not match the input size {xp.shape[:2]}")
    y = y[:th, :tw]
    known = ~m
    if known.any():
        err = float(np.abs(y[known] - x[known]).mean())
        if err > 0.2:
            raise InpaintModelError(f"output does not reproduce the known image area (mean error {err:.2f})")
        if m.any() and float(y[m].mean()) < 0.03 and float(x[known].mean()) > 0.12:
            raise InpaintModelError("output is black inside the hole")
    if (th, tw) != (h, w):
        y = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(y, 0, 1)


def learned_inpaint(image: np.ndarray, hole: np.ndarray, model_id: str, progress=None, max_pixels: int | None = None,
                    model_hole: np.ndarray | None = None) -> np.ndarray:
    """Complete ``hole`` with an installed learned model (``lama`` or ``migan``).

    Any-size exports (LaMa from ailia-models, MI-GAN pipeline) get one context
    rectangle per group of hole regions (the hole plus ~0.6x its extent of
    surroundings on every side), run at native resolution when the rectangle
    fits ``max_pixels`` (default :data:`LEARNED_MAX_PIXELS`) and downscaled
    otherwise; fixed-size exports (e.g. LaMa 512) get square windows. Crops are
    processed one after the other, so later ones see earlier fills as context.
    Fills of downscaled crops get the photo's grain back. Raises
    :class:`InpaintModelError` (or the runtime's own exception) on any failure;
    callers fall back to PatchMatch. ``model_hole`` (optional, a superset of
    ``hole``) is what the model must not see as context (e.g. the fringe of a
    removed person); only ``hole`` pixels are replaced.
    """
    from ..hardware import onnx_session
    from ..models import manager

    hole = hole.astype(bool)
    mhole = hole if model_hole is None else (model_hole.astype(bool) | hole)
    if image.ndim != 3 or image.shape[2] < 3:
        raise InpaintModelError("learned inpainting needs an RGB image")
    sess = onnx_session(str(manager.require(model_id)))
    ip, mp = _model_ports(sess)
    fixed = ip.size or mp.size
    out = np.clip(image[..., :3].astype(np.float32), 0, 1)
    budget = int(max_pixels or LEARNED_MAX_PIXELS)
    if not fixed:
        rects = _crops(mhole, budget=budget)
        downscaled = False
        for k, (y0, y1, x0, x1) in enumerate(rects):
            sub = hole[y0:y1, x0:x1]
            if not sub.any():
                continue
            crop = out[y0:y1, x0:x1].copy()
            crop[mhole[y0:y1, x0:x1]] = 0.0
            y = _run_any_size(sess, model_id, crop, mhole[y0:y1, x0:x1], budget)
            region = out[y0:y1, x0:x1]
            region[sub] = y[sub]
            downscaled |= (y1 - y0) * (x1 - x0) > budget * 1.1
            if progress:
                progress((k + 1) / len(rects))
        if downscaled:
            y0, y1, x0, x1 = bbox_of(hole)
            out = _add_grain(out, image, hole, max(y1 - y0, x1 - x0))
        res = image.astype(np.float32).copy()
        res[..., :3][hole] = out[hole]
        return res
    # fixed-size export (e.g. Carve/LaMa-ONNX 512): one square context crop per group of hole regions
    # (as many runs as the pixel budget allows), reflect-padded where it leaves the image
    side = int(fixed[0])
    max_runs = int(np.clip(round(budget / float(side * side)), 1, 4))
    rects = _crops(mhole, max_crops=max_runs)
    H, W = hole.shape
    big = False
    todo = hole.copy()
    squares = [sq for r in rects for sq in _squares(r, H, W, side)]
    for k, (sy, sx, S) in enumerate(squares):
        ya, yb, xa, xb = max(sy, 0), min(sy + S, H), max(sx, 0), min(sx + S, W)
        sub = todo[ya:yb, xa:xb]
        if not sub.any():
            continue
        # pixels filled by an earlier square are context now; the untrusted fringe stays hidden
        keep_hidden = mhole & ~hole
        crop = _take(out, sy, sx, S)
        sub_m = _take(todo | keep_hidden, sy, sx, S)
        crop[sub_m] = 0.0
        y = _run_model(sess, model_id, crop, sub_m)
        region = out[ya:yb, xa:xb]
        region[sub] = y[ya - sy : yb - sy, xa - sx : xb - sx][sub]
        todo[ya:yb, xa:xb] = False
        big |= S > side * 1.25
        if progress:
            progress((k + 1) / len(squares))
    if big:
        # the model worked on a downscaled crop: give the fill back the photo's grain
        y0, y1, x0, x1 = bbox_of(hole)
        out = _add_grain(out, image, hole, max(y1 - y0, x1 - x0))
    res = image.astype(np.float32).copy()
    res[..., :3][hole] = out[hole]
    return res


def _square(rect: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int]:
    """Square (top, left, side) around ``rect``, shifted inside the image where it fits (it may
    reach beyond the image along a side shorter than the square: :func:`_take` pads there)."""
    y0, y1, x0, x1 = rect
    S = max(y1 - y0, x1 - x0, 16)

    def place(a0, a1, n):
        s0 = (a0 + a1) // 2 - S // 2
        return int(min(max(s0, 0), n - S)) if S <= n else int(s0)

    return place(y0, y1, H), place(x0, x1, W), int(S)


def _squares(rect: tuple[int, int, int, int], H: int, W: int, side: int) -> list[tuple[int, int, int]]:
    """Squares for a fixed-size model covering ``rect``: one square when the rectangle is roughly
    square or small, else overlapping squares of its short side along the long axis (less
    downscaling than one big square around an elongated region such as a frame margin)."""
    y0, y1, x0, x1 = rect
    h, w = y1 - y0, x1 - x0
    lo, hi = min(h, w), max(h, w)
    if hi <= 1.25 * max(lo, side):
        return [_square(rect, H, W)]
    S = max(lo, min(side, hi))
    n = int(math.ceil((hi - S) / (0.6 * S))) + 1
    out = []
    for st in np.linspace(0, hi - S, n).astype(int):
        r = (y0, y1, x0 + st, x0 + st + S) if w >= h else (y0 + st, y0 + st + S, x0, x1)
        out.append(_square(r, H, W))
    return out


def _take(a: np.ndarray, sy: int, sx: int, S: int) -> np.ndarray:
    """S x S window of ``a`` at (sy, sx); parts outside the array are mirrored in."""
    H, W = a.shape[:2]
    ya, yb, xa, xb = max(sy, 0), min(sy + S, H), max(sx, 0), min(sx + S, W)
    win = a[ya:yb, xa:xb]
    pads = ((ya - sy, sy + S - yb), (xa - sx, sx + S - xb)) + ((0, 0),) * (a.ndim - 2)
    return np.pad(win, pads, mode="symmetric") if any(p for pr in pads for p in pr) else win.copy()


def _untrusted_context(hole: np.ndarray, source: np.ndarray | None, frac: float = 0.03) -> np.ndarray | None:
    """Pixels near the hole that the caller excluded from ``source`` (e.g. the fringe of a removed
    person, near surfaces of depth edges): a learned model must not continue them into the fill."""
    if source is None:
        return None
    r = max(3, int(round(frac * max(hole.shape))))
    near = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return hole | (near & ~source.astype(bool))


def lama_inpaint(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Run the optional LaMa ONNX model (kept for compatibility)."""
    return learned_inpaint(image, hole, "lama")


def inpaint(
    image: np.ndarray,
    hole: np.ndarray,
    source: np.ndarray | None = None,
    method: str = "auto",
    progress=None,
    guide: np.ndarray | None = None,
    max_extent: int = 640,
    info: dict | None = None,
) -> tuple[np.ndarray, str]:
    """Inpaint and return (result, method_used).

    ``method='auto'`` uses the best installed learned model for RGB images
    (LaMa, then MI-GAN) and falls back to PatchMatch when none is installed
    or the model fails or returns unusable output. ``source`` and ``guide``
    steer the PatchMatch back-end (learned models see the whole context).
    ``method`` may also name a back-end explicitly ('lama', 'migan',
    'patchmatch'); an explicitly requested learned model that fails raises
    :class:`~lumen_engine.errors.ProcessingError`. ``info`` (optional dict)
    receives the reasons of any fallback under ``'fallback'``.
    """
    hole = hole.astype(bool)
    if not hole.any():
        return image.copy(), "none"
    is_rgb = image.ndim == 3 and image.shape[2] == 3
    if method in LEARNED_MODELS:
        candidates = [method]
    elif method == "auto" and is_rgb:
        candidates = available_learned()
    else:
        candidates = []
    errors: list[str] = []
    for mid in candidates:
        try:
            # the learned model's pixel budget follows the caller's size budget (PatchMatch hole extent):
            # a context crop of ~1.6x that extent at native resolution, within 384^2 .. 896^2 pixels
            budget = int(np.clip((1.6 * max_extent) ** 2, 384**2, 896**2))
            res = learned_inpaint(image, hole, mid, progress, max_pixels=budget, model_hole=_untrusted_context(hole, source))
            if info is not None and errors:
                info["fallback"] = errors
            return res, mid
        except Exception as exc:  # noqa: BLE001 - any model problem falls back to PatchMatch
            errors.append(f"{mid}: {exc}")
    if info is not None and errors:
        info["fallback"] = errors
    if method in LEARNED_MODELS and errors:
        from ..errors import ProcessingError

        raise ProcessingError(f"{method} inpainting failed: {errors[0]}")
    # callers that steer the fill with guide channels (3D background plate) also get the colour-layout prior
    lay = 0.5 if guide is not None else 0.0
    return scaled_patchmatch(image, hole, source=source, max_extent=max_extent, progress=progress, guide=guide, layout=lay), "patchmatch"
