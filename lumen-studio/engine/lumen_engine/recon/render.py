"""CPU (server-side) renderer for the 3D scene and the 3D body model.

Browsers without WebGL (for example virtual machines without GPU
acceleration) cannot run the interactive three.js viewers, so the engine can
render the same geometry itself and return plain JPEG frames:

* **Scene** ('LSC1' layered depth mesh): the main surface, the hidden
  background layer and the coarse backdrop are unprojected exactly like the
  viewer's vertex shader and rasterised with a z-buffer. Colours follow the
  fragment shader (photo / depth / layer tint, low-confidence margin fade).
  Textures are mip-mapped (trilinear for ``full``, bilinear on the nearest level
  for ``preview``), so minified views do not shimmer.
* **Body model** ('LBD1' vertex grid): wireframe, scan lines, points or a
  Lambert-shaded solid; lines can be hidden behind the body surface.

The rasteriser is written with numba (compiled once, cached on disk) and runs
in parallel over horizontal screen bands - every band owns its rows, so the
threads never write the same pixel. The geometry is a regular grid, so
coarse blocks of quads are culled against each band before any triangle is
set up. Parsed scenes, decoded textures (with their mip chains) and
unprojected vertex grids are kept in small LRU caches: moving the camera only
costs projection, rasterisation and shading.
"""

from __future__ import annotations

import io
import json
import math
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .. import config  # noqa: F401  (sets NUMBA_CACHE_DIR before numba is imported)

import cv2  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402
from PIL import Image  # noqa: E402

from ..errors import InvalidInputError  # noqa: E402

SCENE_CLEAR = (10, 12, 16)
BODY_CLEAR = (7, 9, 13)
NEAR_SCENE = 0.02
NEAR_BODY = 0.01
BLOCK = 16  # quads per side of a culling block
MIN_SIZE, MAX_SIZE = 64, 2048
JPEG_QUALITY = 88
_EMPTY = 255  # "no surface" marker in the layer buffer
_TIE_EPS = 1e-3  # relative depth margin: on near-ties main wins over bg over backdrop
_LIGHT = np.array([1.2, 1.5, 2.0]) / np.linalg.norm([1.2, 1.5, 2.0])


def _srgb_to_linear_lut() -> np.ndarray:
    c = np.arange(256, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


# Texel decoding tables (byte -> 0..1). The viewer's textures are sRGB: the GPU decodes them to linear
# light before filtering, the fragment shader mixes (tint, dim, periphery fade, depth palette) in
# linear light and three.js encodes the result back to sRGB - the scene shading does the same.
_LUT_SRGB = _srgb_to_linear_lut()
_LUT_UNIT = (np.arange(256, dtype=np.float32) / 255.0).astype(np.float32)  # plain 0..1 (body renderer)

# --------------------------------------------------------------------------- numba kernels
# Screen convention: pixel (px, py) has its centre at (px + 0.5, py + 0.5), y grows downwards.
# Depth is stored as iw = 1 / (distance along the view axis): larger = nearer, and it is
# exactly linear in screen space, so it can be interpolated with plain barycentrics.


@njit(cache=True, parallel=True, fastmath=True)
def _unproject_grid(disp, rows, cols, gw1, gh1, tanx, aspect, ex, ey, zfar, X, Y, Z):
    """World positions of a disparity grid, identical to the viewer's vertex shader."""
    inv = 1.0 / zfar
    lo = 1.0 / (3.0 * zfar)
    for j in prange(rows.shape[0]):
        r = rows[j]
        yn = (2.0 * r / gh1 - 1.0) * ey
        for i in range(cols.shape[0]):
            c = cols[i]
            den = disp[r, c] * (1.0 - inv) + inv
            if den < lo:
                den = lo
            z = 1.0 / den
            xn = (2.0 * c / gw1 - 1.0) * ex
            X[j, i] = xn * z * tanx
            Y[j, i] = -yn * z * tanx / aspect
            Z[j, i] = -z


@njit(cache=True, parallel=True, fastmath=True)
def _project_grid(X, Y, Z, V, R, P, s, cx, cy, near, SX, SY, IW):
    """World -> screen: p_cam = R^T (p - P); IW = 1 / depth, or -1 when the vertex is in front of
    the near plane or absent (V < -0.5, e.g. background-layer disparity -1)."""
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    p0, p1, p2 = P[0], P[1], P[2]
    for j in prange(X.shape[0]):
        for i in range(X.shape[1]):
            dx = X[j, i] - p0
            dy = Y[j, i] - p1
            dz = Z[j, i] - p2
            xc = r00 * dx + r10 * dy + r20 * dz
            yc = r01 * dx + r11 * dy + r21 * dz
            w = -(r02 * dx + r12 * dy + r22 * dz)
            if w >= near and V[j, i] > -0.5:
                iw = 1.0 / w
                SX[j, i] = cx + xc * iw * s
                SY[j, i] = cy - yc * iw * s
                IW[j, i] = iw
            else:
                SX[j, i] = 0.0
                SY[j, i] = 0.0
                IW[j, i] = -1.0


@njit(cache=True, parallel=True, fastmath=True)
def _drop_stretched(X, Y, Z, keep, min_cos, out):
    """out = keep & not "rubber sheet": a quad is dropped when one of its triangles was seen nearly
    edge-on from the reference camera (|cos| between its normal and the reference view ray below
    ``min_cos``). Such triangles bridge an uncut depth ramp (an object's silhouette, depth noise on
    glass): they look like a thin line from the photo's viewpoint and smear into long streaks as soon
    as the camera moves. Floors and walls, even at grazing angles, stay well above the threshold.
    Identical to the WebGL fragment test in scene.ts."""
    gh, gw = keep.shape
    for yi in prange(gh):
        y = np.int64(yi)
        for xi in range(gw):
            x = np.int64(xi)
            if keep[y, x] == 0:
                out[y, x] = 0
                continue
            ok = 1
            for t in range(2):
                # triangle t=0: (a, c, b); t=1: (b, c, d) of quad a=(x,y) b=(x+1,y) c=(x,y+1) d=(x+1,y+1)
                ja = y
                ia = x + t
                jb = y + 1
                ib = x
                jc = y + t
                ic = x + 1
                ax, ay, az = X[ja, ia], Y[ja, ia], Z[ja, ia]
                e1x, e1y, e1z = X[jb, ib] - ax, Y[jb, ib] - ay, Z[jb, ib] - az
                e2x, e2y, e2z = X[jc, ic] - ax, Y[jc, ic] - ay, Z[jc, ic] - az
                nx = e1y * e2z - e1z * e2y
                ny = e1z * e2x - e1x * e2z
                nz = e1x * e2y - e1y * e2x
                cx = (ax + X[jb, ib] + X[jc, ic]) / 3.0
                cy = (ay + Y[jb, ib] + Y[jc, ic]) / 3.0
                cz = (az + Z[jb, ib] + Z[jc, ic]) / 3.0
                nn = math.sqrt(nx * nx + ny * ny + nz * nz) * math.sqrt(cx * cx + cy * cy + cz * cz)
                if nn > 0.0 and abs(nx * cx + ny * cy + nz * cz) < min_cos * nn:
                    ok = 0
            out[y, x] = ok


@njit(cache=True, parallel=True)
def _block_bounds(SX, SY, IW, keep, bq, bnd):
    """Screen bounding box (minx, maxx, miny, maxy) of the drawable quads of every block."""
    gh, gw = SX.shape
    for by in prange(bnd.shape[0]):
        for bx in range(bnd.shape[1]):
            mnx = 1e30
            mxx = -1e30
            mny = 1e30
            mxy = -1e30
            qy0 = by * bq
            qy1 = min(qy0 + bq, gh - 1)
            qx0 = bx * bq
            qx1 = min(qx0 + bq, gw - 1)
            for qy in range(qy0, qy1):
                for qx in range(qx0, qx1):
                    if keep[qy, qx] == 0:
                        continue
                    for k in range(4):
                        yy = qy + (k >> 1)
                        xx = qx + (k & 1)
                        if IW[yy, xx] > 0.0:
                            sx = SX[yy, xx]
                            sy = SY[yy, xx]
                            mnx = min(mnx, sx)
                            mxx = max(mxx, sx)
                            mny = min(mny, sy)
                            mxy = max(mxy, sy)
            bnd[by, bx, 0] = mnx
            bnd[by, bx, 1] = mxx
            bnd[by, bx, 2] = mny
            bnd[by, bx, 3] = mxy


@njit(cache=True, parallel=True, fastmath=True)
def _raster_grid(SX, SY, IW, A, SF, SB, keep, ucol, vrow, bnd, bq, W, H, nbands,
                 layer, tie_eps, tex_area, slope_bias, const_bias, mode, cull,
                 gdepth, glayer, gu, gv, ga, gs, gr):
    """Rasterise the kept quads of a vertex grid into a G-buffer, parallel over screen bands.

    Quad (x, y) uses vertices a=(x,y) b=(x+1,y) c=(x,y+1) d=(x+1,y+1) as triangles (a,c,b), (b,c,d).
    Per pixel: depth (1/w), layer id, perspective-correct (u, v), attribute A (disparity or model
    depth), two-sided shading (SF front / SB back, only when mode == 2) and texels-per-pixel.
    mode: 0 depth only (occluder), 1 attributes, 2 attributes + shading.
    cull: 1 skips back faces like the viewer's single-sided (THREE.FrontSide) materials, 0 draws both.
    """
    gh, gw = SX.shape
    nby = bnd.shape[0]
    nbx = bnd.shape[1]
    bh = (H + nbands - 1) // nbands
    eps = -1e-7
    use_bias = slope_bias > 0.0 or const_bias > 0.0
    for band in prange(nbands):
        ylo = band * bh
        yhi = min(H, ylo + bh)
        if ylo >= yhi:
            continue
        flo = ylo - 0.5
        fhi = yhi + 0.5
        for by in range(nby):
            for bx in range(nbx):
                if bnd[by, bx, 0] > bnd[by, bx, 1]:
                    continue  # nothing drawable in this block
                if bnd[by, bx, 3] < flo or bnd[by, bx, 2] > fhi:
                    continue
                if bnd[by, bx, 1] < -1.0 or bnd[by, bx, 0] > W + 1.0:
                    continue
                qy0 = by * bq
                qy1 = min(qy0 + bq, gh - 1)
                qx0 = bx * bq
                qx1 = min(qx0 + bq, gw - 1)
                for qy in range(qy0, qy1):
                    for qx in range(qx0, qx1):
                        if keep[qy, qx] == 0:
                            continue
                        qya = SY[qy, qx]
                        qyb = SY[qy, qx + 1]
                        qyc = SY[qy + 1, qx]
                        qyd = SY[qy + 1, qx + 1]
                        if max(max(qya, qyb), max(qyc, qyd)) < flo or min(min(qya, qyb), min(qyc, qyd)) > fhi:
                            continue
                        qtex = (ucol[qx + 1] - ucol[qx]) * (vrow[qy + 1] - vrow[qy]) * tex_area
                        for t in range(2):
                            # triangle t=0: (a, c, b); t=1: (b, c, d) -> vertices (ia, ja), (ib, jb), (ic, jc)
                            if t == 0:
                                ja, ia, jb, ib, jc, ic = qy, qx, qy + 1, qx, qy, qx + 1
                            else:
                                ja, ia, jb, ib, jc, ic = qy, qx + 1, qy + 1, qx, qy + 1, qx + 1
                            # float64 set-up: float32 products here would leave sub-pixel cracks on shared edges
                            wa = np.float64(IW[ja, ia])
                            wb = np.float64(IW[jb, ib])
                            wc = np.float64(IW[jc, ic])
                            if wa <= 0.0 or wb <= 0.0 or wc <= 0.0:
                                continue  # crosses the near plane
                            xa = np.float64(SX[ja, ia])
                            ya = np.float64(SY[ja, ia])
                            xb = np.float64(SX[jb, ib])
                            yb = np.float64(SY[jb, ib])
                            xc = np.float64(SX[jc, ic])
                            yc = np.float64(SY[jc, ic])
                            # pixel-centre bounding box, clipped to the band (cheap reject first)
                            minx = max(min(xa, min(xb, xc)), -2.0)
                            maxx = min(max(xa, max(xb, xc)), W + 2.0)
                            miny = max(min(ya, min(yb, yc)), ylo - 2.0)
                            maxy = min(max(ya, max(yb, yc)), yhi + 2.0)
                            px0 = max(int(math.ceil(minx - 0.5)), 0)
                            px1 = min(int(math.floor(maxx - 0.5)), W - 1)
                            if px0 > px1:
                                continue
                            py0 = max(int(math.ceil(miny - 0.5)), ylo)
                            py1 = min(int(math.floor(maxy - 0.5)), yhi - 1)
                            if py0 > py1:
                                continue
                            area = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
                            if not (area > 1e-12 or area < -1e-12):
                                continue
                            if cull != 0 and area > 0.0:
                                continue  # back face: front faces have negative area on the y-down screen
                            inv = 1.0 / area
                            # barycentric weights of a and b are affine in the pixel centre; c = 1 - a - b
                            Aa = (yb - yc) * inv
                            Ba = (xc - xb) * inv
                            Ca = ((yc - yb) * xb - (xc - xb) * yb) * inv
                            Ab = (yc - ya) * inv
                            Bb = (xa - xc) * inv
                            Cb = ((ya - yc) * xc - (xa - xc) * yc) * inv
                            dwa = wa - wc
                            dwb = wb - wc
                            bias = 0.0
                            if use_bias:
                                # polygon offset: push the surface back by its depth slope
                                gx = Aa * dwa + Ab * dwb
                                gy = Ba * dwa + Bb * dwb
                                bias = slope_bias * (abs(gx) + abs(gy)) + const_bias * max(wa, max(wb, wc))
                            ratio = qtex / abs(area)  # texels per pixel (area): selects the mip level
                            for py in range(py0, py1 + 1):
                                cy = py + 0.5
                                rowa = Ba * cy + Ca
                                rowb = Bb * cy + Cb
                                base = py * W
                                for px in range(px0, px1 + 1):
                                    cx = px + 0.5
                                    la = Aa * cx + rowa
                                    lb = Ab * cx + rowb
                                    lc = 1.0 - la - lb
                                    if la < eps or lb < eps or lc < eps:
                                        continue
                                    iw0 = wc + la * dwa + lb * dwb
                                    iw = iw0 - bias
                                    i = base + px
                                    ol = glayer[i]
                                    if ol != 255:
                                        od = gdepth[i]
                                        if ol < layer:
                                            if iw <= od * (1.0 + tie_eps):
                                                continue
                                        elif iw <= od:
                                            continue
                                    gdepth[i] = iw
                                    glayer[i] = layer
                                    if mode == 0:
                                        continue
                                    k = 1.0 / iw0
                                    fa = la * wa * k
                                    fb = lb * wb * k
                                    fc = lc * wc * k
                                    gu[i] = fa * ucol[ia] + fb * ucol[ib] + fc * ucol[ic]
                                    gv[i] = fa * vrow[ja] + fb * vrow[jb] + fc * vrow[jc]
                                    ga[i] = fa * A[ja, ia] + fb * A[jb, ib] + fc * A[jc, ic]
                                    gr[i] = ratio
                                    if mode == 2:
                                        # two-sided lighting: front faces have negative area (y-down screen)
                                        if area < 0.0:
                                            gs[i] = fa * SF[ja, ia] + fb * SF[jb, ib] + fc * SF[jc, ic]
                                        else:
                                            gs[i] = fa * SB[ja, ia] + fb * SB[jb, ib] + fc * SB[jc, ic]


@njit(cache=True, inline="always", fastmath=True)
def _bilinear(buf, off, w, h, u, v, lut):
    """Bilinear texel lookup; every texel is decoded through ``lut`` (byte -> 0..1) before filtering."""
    x = u * w - 0.5
    y = v * h - 0.5
    fx0 = math.floor(x)
    fy0 = math.floor(y)
    fx = x - fx0
    fy = y - fy0
    x0 = min(max(int(fx0), 0), w - 1)
    y0 = min(max(int(fy0), 0), h - 1)
    x1 = min(max(int(fx0) + 1, 0), w - 1)
    y1 = min(max(int(fy0) + 1, 0), h - 1)
    i00 = off + (y0 * w + x0) * 3
    i01 = off + (y0 * w + x1) * 3
    i10 = off + (y1 * w + x0) * 3
    i11 = off + (y1 * w + x1) * 3
    w00 = (1.0 - fx) * (1.0 - fy)
    w01 = fx * (1.0 - fy)
    w10 = (1.0 - fx) * fy
    w11 = fx * fy
    r = lut[buf[i00]] * w00 + lut[buf[i01]] * w01 + lut[buf[i10]] * w10 + lut[buf[i11]] * w11
    g = lut[buf[i00 + 1]] * w00 + lut[buf[i01 + 1]] * w01 + lut[buf[i10 + 1]] * w10 + lut[buf[i11 + 1]] * w11
    b = lut[buf[i00 + 2]] * w00 + lut[buf[i01 + 2]] * w01 + lut[buf[i10 + 2]] * w10 + lut[buf[i11 + 2]] * w11
    return r, g, b


@njit(cache=True, inline="always", fastmath=True)
def _nearest(buf, off, w, h, u, v, lut):
    x = min(max(int(math.floor(u * w)), 0), w - 1)
    y = min(max(int(math.floor(v * h)), 0), h - 1)
    i = off + (y * w + x) * 3
    return np.float64(lut[buf[i]]), np.float64(lut[buf[i + 1]]), np.float64(lut[buf[i + 2]])


@njit(cache=True, inline="always", fastmath=True)
def _sample(buf, offs, ws, hs, u, v, ratio, filt, lut):
    """Mip-mapped texture lookup (texels decoded through ``lut``); ratio = texels per pixel (area).

    filt: 0 nearest level + nearest texel, 1 nearest level + bilinear, 2 trilinear."""
    n = offs.shape[0]
    lod = 0.5 * math.log2(ratio) if ratio > 1.0 else 0.0
    if lod > n - 1:
        lod = n - 1.0
    if filt == 2:
        l0 = int(lod)
        f = lod - l0
        r, g, b = _bilinear(buf, offs[l0], ws[l0], hs[l0], u, v, lut)
        if f > 0.01 and l0 + 1 < n:
            r2, g2, b2 = _bilinear(buf, offs[l0 + 1], ws[l0 + 1], hs[l0 + 1], u, v, lut)
            r += (r2 - r) * f
            g += (g2 - g) * f
            b += (b2 - b) * f
        return r, g, b
    l0 = min(int(lod + 0.5), n - 1)
    if filt == 1:
        return _bilinear(buf, offs[l0], ws[l0], hs[l0], u, v, lut)
    return _nearest(buf, offs[l0], ws[l0], hs[l0], u, v, lut)


@njit(cache=True, inline="always", fastmath=True)
def _inferno(t):
    t = min(max(t, 0.0), 1.0)
    if t < 0.6:
        k = t / 0.6
        return 0.0015 + (0.87 - 0.0015) * k, 0.0005 + (0.24 - 0.0005) * k, 0.014 + (0.32 - 0.014) * k
    k = (t - 0.6) / 0.4
    return 0.87 + (0.99 - 0.87) * k, 0.24 + (0.99 - 0.24) * k, 0.32 + (0.64 - 0.32) * k


@njit(cache=True, inline="always", fastmath=True)
def _turbo(t):
    t = min(max(t, 0.0), 1.0)
    r = 1.6 * t - 0.4 + (0.6 * math.sin(math.pi * t) if t > 0.5 else 0.0)
    g = math.sin(math.pi * min(1.0, t * 1.05))
    b = 1.0 - 1.8 * max(0.0, t - 0.15) + 0.4 * math.sin(math.pi * t * 0.5)
    return min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0)


@njit(cache=True, inline="always", fastmath=True)
def _u8(x):
    x = x * 255.0 + 0.5
    if x < 0.0:
        return np.uint8(0)
    if x > 255.0:
        return np.uint8(255)
    return np.uint8(int(x))


@njit(cache=True, inline="always", fastmath=True)
def _srgb_u8(x):
    """Linear light (0..1) -> 8-bit sRGB, as three.js encodes the viewer's output (sRGBTransferOETF)."""
    if x <= 0.0031308:
        return _u8(12.92 * x)
    return _u8(1.055 * math.pow(x, 0.41666) - 0.055)


@njit(cache=True, inline="always", fastmath=True)
def _env_color(ebuf, ew, eh, R, s, cx, cy, px, py, lut):
    """Far environment (equirectangular panorama at infinity) seen through pixel (px, py): the view
    ray is rotated to world space and looked up with environment.env_uv's convention (bilinear,
    wrapping around in longitude)."""
    xc = (px + 0.5 - cx) / s
    yc = -(py + 0.5 - cy) / s
    dx = R[0, 0] * xc + R[0, 1] * yc - R[0, 2]
    dy = R[1, 0] * xc + R[1, 1] * yc - R[1, 2]
    dz = R[2, 0] * xc + R[2, 1] * yc - R[2, 2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    dy = min(max(dy / n, -1.0), 1.0)
    u = 0.5 + math.atan2(dx, -dz) / (2.0 * math.pi)
    v = 0.5 - math.asin(dy) / math.pi
    x = u * ew - 0.5
    y = v * eh - 0.5
    fx0 = math.floor(x)
    fy0 = math.floor(y)
    fx = x - fx0
    fy = y - fy0
    x0 = int(fx0) % ew
    x1 = (x0 + 1) % ew
    y0 = min(max(int(fy0), 0), eh - 1)
    y1 = min(max(int(fy0) + 1, 0), eh - 1)
    i00 = (y0 * ew + x0) * 3
    i01 = (y0 * ew + x1) * 3
    i10 = (y1 * ew + x0) * 3
    i11 = (y1 * ew + x1) * 3
    w00 = (1.0 - fx) * (1.0 - fy)
    w01 = fx * (1.0 - fy)
    w10 = (1.0 - fx) * fy
    w11 = fx * fy
    r = lut[ebuf[i00]] * w00 + lut[ebuf[i01]] * w01 + lut[ebuf[i10]] * w10 + lut[ebuf[i11]] * w11
    g = lut[ebuf[i00 + 1]] * w00 + lut[ebuf[i01 + 1]] * w01 + lut[ebuf[i10 + 1]] * w10 + lut[ebuf[i11 + 1]] * w11
    b = lut[ebuf[i00 + 2]] * w00 + lut[ebuf[i01 + 2]] * w01 + lut[ebuf[i10 + 2]] * w10 + lut[ebuf[i11 + 2]] * w11
    return r, g, b


@njit(cache=True, inline="always", fastmath=True)
def _margin_weight(m, margin_fade, has_env):
    """How much of the environment colour replaces a surface colour at margin coordinate m (0 at the
    photo's border, 1 at the outer edge of the outpainted margin) - identical in scene.ts.
    With an environment the outermost part of the margin always dissolves into it (no hard edge)."""
    m = min(max(m, 0.0), 1.0)
    f = m * margin_fade
    if has_env != 0:
        t = min(max((m - 0.72) / 0.28, 0.0), 1.0)
        f = max(f, t * t * (3.0 - 2.0 * t))
    return f


@njit(cache=True, parallel=True, fastmath=True)
def _shade_scene(glayer, gu, gv, ga, gr, W, H,
                 buf0, off0, w0, h0, buf1, off1, w1, h1, buf2, off2, w2, h2,
                 mode, filt, margin_fade, ex, ey, lut, ebuf, ew, eh, has_env, R, s, cx, cy, out):
    """Deferred shading of the scene G-buffer, following the viewer's fragment shader: the texture is
    decoded to linear light (``lut``), tinted / dimmed / faded there and encoded back to sRGB.
    Empty pixels show the far environment (``has_env``) or the clear colour."""
    mx = max(ex - 1.0, 1e-4)
    my = max(ey - 1.0, 1e-4)
    if mode == 1:
        has_env = 0  # depth view: plain background
    for y in prange(H):
        for x in range(W):
            i = y * W + x
            L = glayer[i]
            if L == 255:
                if has_env != 0:
                    glayer[i] = 3  # environment (tells G-buffer readers this pixel is not a void)
                    r, g, b = _env_color(ebuf, ew, eh, R, s, cx, cy, x, y, lut)
                    if mode == 2:
                        r += (1.0 - r) * 0.85
                        g += (0.25 - g) * 0.85
                        b += (0.6 - b) * 0.85
                    out[y, x, 0] = _srgb_u8(r)
                    out[y, x, 1] = _srgb_u8(g)
                    out[y, x, 2] = _srgb_u8(b)
                else:
                    out[y, x, 0] = 10
                    out[y, x, 1] = 12
                    out[y, x, 2] = 16
                continue
            u = gu[i]
            v = gv[i]
            if mode == 1:
                r, g, b = _inferno(max(ga[i], 0.0))
            elif L == 0:
                r, g, b = _sample(buf0, off0, w0, h0, u, v, gr[i], filt, lut)
            elif L == 1:
                r, g, b = _sample(buf1, off1, w1, h1, u, v, gr[i], filt, lut)
            else:
                r, g, b = _sample(buf2, off2, w2, h2, u, v, gr[i], filt, lut)
            if mode == 2:
                t = 0.0 if L == 0 else (0.55 if L == 1 else 0.85)
                r += (1.0 - r) * t
                g += (0.25 - g) * t
                b += (0.6 - b) * t
            if L == 2:
                r *= 0.96
                g *= 0.96
                b *= 0.96
            if margin_fade > 0.0 or has_env != 0:
                xn = abs((2.0 * u - 1.0) * ex)
                yn = abs((2.0 * v - 1.0) * ey)
                m = max(max(xn - 1.0, 0.0) / mx, max(yn - 1.0, 0.0) / my)
                f = _margin_weight(m, margin_fade, has_env)
                if f > 0.0:
                    if has_env != 0:
                        er, eg, eb = _env_color(ebuf, ew, eh, R, s, cx, cy, x, y, lut)
                        if mode == 2:
                            er += (1.0 - er) * 0.85
                            eg += (0.25 - eg) * 0.85
                            eb += (0.6 - eb) * 0.85
                    else:
                        er, eg, eb = 0.04, 0.045, 0.06
                    r += (er - r) * f
                    g += (eg - g) * f
                    b += (eb - b) * f
            out[y, x, 0] = _srgb_u8(r)
            out[y, x, 1] = _srgb_u8(g)
            out[y, x, 2] = _srgb_u8(b)


@njit(cache=True, parallel=True, fastmath=True)
def _body_vertices(z, keep, aspect, zs, X, Y, Z, N, SF, SB, light, amb, kd, hemi):
    """Body model positions, smooth (area-weighted) vertex normals and two-sided Lambert terms."""
    rows, cols = z.shape
    for y in prange(rows):
        for x in range(cols):
            X[y, x] = (x / (cols - 1) - 0.5) * aspect
            Y[y, x] = 0.5 - y / (rows - 1)
            Z[y, x] = (z[y, x] - 0.5) * zs
    for y in prange(rows):
        for x in range(cols):
            nx = 0.0
            ny = 0.0
            nz = 0.0
            # quads touching this vertex; a quad is drawn when its four corners are kept
            for qy in range(y - 1, y + 1):
                for qx in range(x - 1, x + 1):
                    if qy < 0 or qx < 0 or qy >= rows - 1 or qx >= cols - 1:
                        continue
                    if keep[qy, qx] == 0 or keep[qy, qx + 1] == 0 or keep[qy + 1, qx] == 0 or keep[qy + 1, qx + 1] == 0:
                        continue
                    ax, ay, az = X[qy, qx], Y[qy, qx], Z[qy, qx]
                    bx, by, bz = X[qy, qx + 1], Y[qy, qx + 1], Z[qy, qx + 1]
                    cx, cy, cz = X[qy + 1, qx], Y[qy + 1, qx], Z[qy + 1, qx]
                    dx, dy, dz = X[qy + 1, qx + 1], Y[qy + 1, qx + 1], Z[qy + 1, qx + 1]
                    is_a = qy == y and qx == x
                    is_d = qy + 1 == y and qx + 1 == x
                    if not is_d:  # triangle (a, c, b): normal (c - a) x (b - a)
                        e1x, e1y, e1z = cx - ax, cy - ay, cz - az
                        e2x, e2y, e2z = bx - ax, by - ay, bz - az
                        nx += e1y * e2z - e1z * e2y
                        ny += e1z * e2x - e1x * e2z
                        nz += e1x * e2y - e1y * e2x
                    if not is_a:  # triangle (b, c, d): normal (c - b) x (d - b)
                        e1x, e1y, e1z = cx - bx, cy - by, cz - bz
                        e2x, e2y, e2z = dx - bx, dy - by, dz - bz
                        nx += e1y * e2z - e1z * e2y
                        ny += e1z * e2x - e1x * e2z
                        nz += e1x * e2y - e1y * e2x
            ln = math.sqrt(nx * nx + ny * ny + nz * nz)
            if ln > 0.0:
                nx /= ln
                ny /= ln
                nz /= ln
            else:
                nx, ny, nz = 0.0, 0.0, 1.0
            N[y, x, 0] = nx
            N[y, x, 1] = ny
            N[y, x, 2] = nz
            d = nx * light[0] + ny * light[1] + nz * light[2]
            SF[y, x] = amb + kd * max(d, 0.0) + hemi * ny
            SB[y, x] = amb + kd * max(-d, 0.0) - hemi * ny


@njit(cache=True, parallel=True, fastmath=True)
def _shade_body(glayer, gu, gv, ga, gs, gr, W, H, colors, buf, offs, ws, hs, filt, lut, out):
    """Solid style: base colour (depth / photo / hologram) times the interpolated Lambert term."""
    for y in prange(H):
        for x in range(W):
            i = y * W + x
            if glayer[i] == 255:
                out[y, x, 0] = 7
                out[y, x, 1] = 9
                out[y, x, 2] = 13
                continue
            t = ga[i]
            if colors == 0:
                r, g, b = _turbo(t)
            elif colors == 1:
                r, g, b = _sample(buf, offs, ws, hs, gu[i], gv[i], gr[i], filt, lut)
            else:
                t = min(max(t, 0.0), 1.0)
                r, g, b = 0.25 + 0.45 * t, 0.75 + 0.25 * t, 1.0
            s = gs[i]
            out[y, x, 0] = _u8(r * s)
            out[y, x, 1] = _u8(g * s)
            out[y, x, 2] = _u8(b * s)


@njit(cache=True)
def _plot(fb, px, py, cov, iw, r, g, b, opacity, hide, odepth, olayer, ylo, yhi, W):
    if py < ylo or py >= yhi or px < 0 or px >= W or cov <= 0.0:
        return
    i = py * W + px
    if hide and olayer[i] != 255 and iw < odepth[i]:
        return  # behind the body surface
    a = cov * opacity
    fb[py, px, 0] += (r - fb[py, px, 0]) * a
    fb[py, px, 1] += (g - fb[py, px, 1]) * a
    fb[py, px, 2] += (b - fb[py, px, 2]) * a


@njit(cache=True)
def _line(fb, xa, ya, wa, ra, ga, ba, xb, yb, wb, rb, gb, bb, opacity, hide, odepth, olayer, ylo, yhi, W):
    """Anti-aliased (Wu) line, half-open at its end so joined segments do not double-blend."""
    dx = xb - xa
    dy = yb - ya
    if abs(dx) >= abs(dy):
        if dx == 0.0:
            return
        if dx < 0.0:
            xa, ya, wa, ra, ga, ba, xb, yb, wb, rb, gb, bb = xb, yb, wb, rb, gb, bb, xa, ya, wa, ra, ga, ba
            dx = -dx
            dy = -dy
        p0 = max(int(math.ceil(max(xa, -2.0) - 0.5)), 0)
        p1 = min(int(math.ceil(min(xb, W + 2.0) - 0.5)) - 1, W - 1)
        for px in range(p0, p1 + 1):
            t = (px + 0.5 - xa) / dx
            yy = ya + t * dy - 0.5
            if yy < ylo - 2.0 or yy > yhi + 1.0:
                continue
            fy = math.floor(yy)
            f = yy - fy
            py = int(fy)
            iw = wa + t * (wb - wa)
            r = ra + t * (rb - ra)
            g = ga + t * (gb - ga)
            b = ba + t * (bb - ba)
            _plot(fb, px, py, 1.0 - f, iw, r, g, b, opacity, hide, odepth, olayer, ylo, yhi, W)
            _plot(fb, px, py + 1, f, iw, r, g, b, opacity, hide, odepth, olayer, ylo, yhi, W)
    else:
        if dy < 0.0:
            xa, ya, wa, ra, ga, ba, xb, yb, wb, rb, gb, bb = xb, yb, wb, rb, gb, bb, xa, ya, wa, ra, ga, ba
            dx = -dx
            dy = -dy
        p0 = max(int(math.ceil(max(ya, ylo - 2.0) - 0.5)), ylo)
        p1 = min(int(math.ceil(min(yb, yhi + 2.0) - 0.5)) - 1, yhi - 1)
        for py in range(p0, p1 + 1):
            t = (py + 0.5 - ya) / dy
            xx = xa + t * dx - 0.5
            if xx < -2.0 or xx > W + 1.0:
                continue
            fx = math.floor(xx)
            f = xx - fx
            px = int(fx)
            iw = wa + t * (wb - wa)
            r = ra + t * (rb - ra)
            g = ga + t * (gb - ga)
            b = ba + t * (bb - ba)
            _plot(fb, px, py, 1.0 - f, iw, r, g, b, opacity, hide, odepth, olayer, ylo, yhi, W)
            _plot(fb, px + 1, py, f, iw, r, g, b, opacity, hide, odepth, olayer, ylo, yhi, W)


@njit(cache=True, parallel=True)
def _draw_lines(SX, SY, IW, C, keep, hrows, vcols, W, H, nbands, opacity, hide, odepth, olayer, fb):
    """Grid lines between kept neighbours: along the rows in ``hrows`` and the columns in ``vcols``."""
    rows, cols = SX.shape
    bh = (H + nbands - 1) // nbands
    for band in prange(nbands):
        ylo = band * bh
        yhi = min(H, ylo + bh)
        if ylo >= yhi:
            continue
        flo = ylo - 2.0
        fhi = yhi + 2.0
        for k in range(hrows.shape[0]):
            y = hrows[k]
            for x in range(cols - 1):
                if keep[y, x] == 0 or keep[y, x + 1] == 0 or IW[y, x] <= 0.0 or IW[y, x + 1] <= 0.0:
                    continue
                ya = SY[y, x]
                yb = SY[y, x + 1]
                if (ya < flo and yb < flo) or (ya > fhi and yb > fhi):
                    continue
                _line(fb, SX[y, x], ya, IW[y, x], C[y, x, 0], C[y, x, 1], C[y, x, 2],
                      SX[y, x + 1], yb, IW[y, x + 1], C[y, x + 1, 0], C[y, x + 1, 1], C[y, x + 1, 2],
                      opacity, hide, odepth, olayer, ylo, yhi, W)
        for k in range(vcols.shape[0]):
            x = vcols[k]
            for y in range(rows - 1):
                if keep[y, x] == 0 or keep[y + 1, x] == 0 or IW[y, x] <= 0.0 or IW[y + 1, x] <= 0.0:
                    continue
                ya = SY[y, x]
                yb = SY[y + 1, x]
                if (ya < flo and yb < flo) or (ya > fhi and yb > fhi):
                    continue
                _line(fb, SX[y, x], ya, IW[y, x], C[y, x, 0], C[y, x, 1], C[y, x, 2],
                      SX[y + 1, x], yb, IW[y + 1, x], C[y + 1, x, 0], C[y + 1, x, 1], C[y + 1, x, 2],
                      opacity, hide, odepth, olayer, ylo, yhi, W)


@njit(cache=True, parallel=True)
def _draw_points(SX, SY, IW, C, keep, size_k, W, H, nbands, pdepth, fb):
    """One z-buffered square dot per kept vertex; the side follows the projected grid spacing."""
    rows, cols = SX.shape
    bh = (H + nbands - 1) // nbands
    for band in prange(nbands):
        ylo = band * bh
        yhi = min(H, ylo + bh)
        if ylo >= yhi:
            continue
        for y in range(rows):
            for x in range(cols):
                iw = IW[y, x]
                if keep[y, x] == 0 or iw <= 0.0:
                    continue
                sy = SY[y, x]
                if sy < ylo - 9.0 or sy > yhi + 9.0:
                    continue
                sx = SX[y, x]
                if sx < -9.0 or sx > W + 9.0:
                    continue
                side = min(max(int(size_k * iw + 0.5), 1), 8)
                hs = side * 0.5
                px0 = max(int(math.ceil(sx - hs - 0.5)), 0)
                px1 = min(px0 + side - 1, W - 1)
                py0 = int(math.ceil(sy - hs - 0.5))
                py1 = min(py0 + side - 1, yhi - 1)
                py0 = max(py0, ylo)
                for py in range(py0, py1 + 1):
                    for px in range(px0, px1 + 1):
                        i = py * W + px
                        if iw > pdepth[i]:
                            pdepth[i] = iw
                            fb[py, px, 0] = C[y, x, 0]
                            fb[py, px, 1] = C[y, x, 1]
                            fb[py, px, 2] = C[y, x, 2]


@njit(cache=True, parallel=True, fastmath=True)
def _body_colors(z, colors, tex, C):
    """Per-vertex colours: 0 depth (turbo), 1 photo (texture resized to the grid), 2 hologram."""
    rows, cols = z.shape
    for y in prange(rows):
        for x in range(cols):
            t = z[y, x]
            if colors == 0:
                r, g, b = _turbo(t)
            elif colors == 1:
                r = tex[y, x, 0] / 255.0
                g = tex[y, x, 1] / 255.0
                b = tex[y, x, 2] / 255.0
            else:
                t = min(max(t, 0.0), 1.0)
                r, g, b = 0.25 + 0.45 * t, 0.75 + 0.25 * t, 1.0
            C[y, x, 0] = r
            C[y, x, 1] = g
            C[y, x, 2] = b


@njit(cache=True, parallel=True, fastmath=True)
def _to_u8(fb, out):
    for y in prange(fb.shape[0]):
        for x in range(fb.shape[1]):
            for c in range(3):
                out[y, x, c] = _u8(fb[y, x, c])


# --------------------------------------------------------------------------- data


@dataclass
class SceneData:
    header: dict
    disp: np.ndarray  # (gh, gw) float32
    keep: np.ndarray  # (gh-1, gw-1) uint8
    bg_disp: np.ndarray  # (gh, gw) float32, -1 where absent
    keep_bg: np.ndarray  # (gh-1, gw-1) uint8
    back: np.ndarray | None  # (bh, bw) float32
    grids: dict = field(default_factory=dict)  # stride -> _Grids
    verts: dict = field(default_factory=dict)  # (fov, zfar, stride) -> vertex arrays (bounded by _vert_lru)


@dataclass
class BodyData:
    header: dict
    z: np.ndarray  # (rows, cols) float32
    keep: np.ndarray  # (rows, cols) uint8
    qkeep: np.ndarray  # (rows-1, cols-1) uint8: quads whose four corners are kept


class MipTexture:
    """RGB uint8 texture with its mip chain packed in one buffer (for the numba samplers)."""

    def __init__(self, rgb: np.ndarray):
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        levels = [rgb]
        while min(levels[-1].shape[:2]) >= 8 and len(levels) < 12:
            h, w = levels[-1].shape[:2]
            levels.append(cv2.resize(levels[-1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA))
        self.buf = np.concatenate([lv.ravel() for lv in levels])
        self.rgb = self.buf[: rgb.size].reshape(rgb.shape)  # level 0 as a view (no second copy)
        self.w = np.array([lv.shape[1] for lv in levels], np.int64)
        self.h = np.array([lv.shape[0] for lv in levels], np.int64)
        sizes = np.array([lv.size for lv in levels], np.int64)
        self.off = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.area = float(rgb.shape[0] * rgb.shape[1])
        self._grid: dict = {}

    def args(self):
        return self.buf, self.off, self.w, self.h

    def resized(self, w: int, h: int) -> np.ndarray:
        """Texture averaged down to (h, w) - per-vertex colours of a grid."""
        key = (w, h)
        if key not in self._grid:
            self._grid[key] = np.ascontiguousarray(cv2.resize(self.rgb, (w, h), interpolation=cv2.INTER_AREA))
        return self._grid[key]


@dataclass
class _Grids:
    """Grid topology of one layer set at a given vertex stride."""

    rows: np.ndarray
    cols: np.ndarray
    u: np.ndarray
    v: np.ndarray
    keep: np.ndarray
    keep_bg: np.ndarray
    disp: np.ndarray
    bg_disp: np.ndarray
    triangles: int
    bg_triangles: int


def _unpack_bits(raw, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(raw, np.uint8), count=n).astype(np.uint8)


def parse_scene(blob: bytes) -> SceneData:
    """Parse the engine's 'LSC1' scene container (see scene.pack_scene)."""
    try:
        if blob[:4] != b"LSC1":
            raise ValueError("magic")
        (hlen,) = struct.unpack_from("<I", blob, 4)
        header = json.loads(blob[8 : 8 + hlen].decode())
        base = 8 + hlen
        gw, gh = int(header["grid_w"]), int(header["grid_h"])
        for k in ("aspect", "extent_x", "extent_y"):
            if not (math.isfinite(float(header[k])) and float(header[k]) > 0):
                raise ValueError(k)
        if gw < 2 or gh < 2:
            raise ValueError("grid")
        bufs = header["buffers"]
        mv = memoryview(blob)

        def raw(i):
            off, n = bufs[i]
            if base + off + n > len(blob):
                raise ValueError("truncated")
            return mv[base + off : base + off + n]

        nq = (gw - 1) * (gh - 1)
        disp = np.frombuffer(raw(0), "<f4").astype(np.float32).reshape(gh, gw)
        keep = _unpack_bits(raw(1), nq).reshape(gh - 1, gw - 1)
        bg = np.frombuffer(raw(2), "<f4").astype(np.float32).reshape(gh, gw)
        keep_bg = _unpack_bits(raw(3), nq).reshape(gh - 1, gw - 1)
        back = None
        bw, bh = int(header.get("back_w") or 0), int(header.get("back_h") or 0)
        if len(bufs) > 4 and bw > 1 and bh > 1:
            back = np.frombuffer(raw(4), "<f4").astype(np.float32).reshape(bh, bw)
    except Exception as exc:
        raise InvalidInputError("This file is not a valid Lumen 3D scene.", hint="Run the 3D reconstruction again.") from exc
    return SceneData(header, disp, keep, bg, keep_bg, back)


def parse_body(blob: bytes) -> BodyData:
    """Parse the engine's 'LBD1' body container (see body3d.build_body3d)."""
    try:
        if blob[:4] != b"LBD1":
            raise ValueError("magic")
        (hlen,) = struct.unpack_from("<I", blob, 4)
        header = json.loads(blob[8 : 8 + hlen].decode())
        cols, rows = int(header["cols"]), int(header["rows"])
        for k in ("aspect", "depth_scale"):
            if not math.isfinite(float(header[k])):
                raise ValueError(k)
        n = cols * rows
        off = 8 + hlen
        if off + n * 5 > len(blob) or cols < 2 or rows < 2:
            raise ValueError("truncated")
        z = np.frombuffer(blob, "<f4", count=n, offset=off).astype(np.float32).reshape(rows, cols)
        keep = (np.frombuffer(blob, np.uint8, count=n, offset=off + n * 4).reshape(rows, cols) > 0).astype(np.uint8)
    except Exception as exc:
        raise InvalidInputError("This file is not a valid Lumen body model.", hint="Build the 3D body model again.") from exc
    qkeep = (keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]).astype(np.uint8)
    return BodyData(header, z, keep, qkeep)


def decode_texture(blob: bytes) -> MipTexture:
    try:
        rgb = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise InvalidInputError("A texture of the 3D model could not be read.") from exc
    return MipTexture(rgb)


def _strided(n: int, stride: int) -> np.ndarray:
    idx = np.arange(0, n, stride, dtype=np.int64)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _scene_grids(sc: SceneData, stride: int) -> _Grids:
    g = sc.grids.get(stride)
    if g is not None:
        return g
    gh, gw = sc.disp.shape
    rows, cols = _strided(gh, stride), _strided(gw, stride)
    if stride == 1:
        keep, keep_bg = sc.keep, sc.keep_bg
        disp, bg = sc.disp, sc.bg_disp
    else:
        # a coarse quad is drawn only when every fine quad under it is (no bridging across cuts)
        def coarse(k):
            k = np.logical_and.reduceat(k.astype(bool), rows[:-1], axis=0)
            return np.ascontiguousarray(np.logical_and.reduceat(k, cols[:-1], axis=1).astype(np.uint8))

        keep, keep_bg = coarse(sc.keep), coarse(sc.keep_bg)
        disp = np.ascontiguousarray(sc.disp[np.ix_(rows, cols)])
        bg = np.ascontiguousarray(sc.bg_disp[np.ix_(rows, cols)])
    g = _Grids(rows, cols, cols / (gw - 1.0), rows / (gh - 1.0), np.ascontiguousarray(keep), np.ascontiguousarray(keep_bg),
               disp, bg, int(keep.sum()) * 2, int(keep_bg.sum()) * 2)
    sc.grids[stride] = g
    return g


def _preview_detail_keep(fine: np.ndarray, coarse: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Fine quads lost by conservative LOD cuts, retained around silhouettes.

    Uniformly coarsening a cut mesh erases narrow limbs/rails and widens every
    crack. Keep coarse interior geometry, but restore its fine boundary cells.
    """
    ri = np.searchsorted(rows, np.arange(fine.shape[0]), side="right")-1
    ci = np.searchsorted(cols, np.arange(fine.shape[1]), side="right")-1
    covered = coarse[np.ix_(ri, ci)] > 0
    return np.ascontiguousarray((fine.astype(bool) & ~covered).astype(np.uint8))


def _unproject(sc: SceneData, disp: np.ndarray, rows: np.ndarray, cols: np.ndarray, gw: int, gh: int, fov: float, zfar: float):
    h = sc.header
    shape = (rows.shape[0], cols.shape[0])
    X, Y, Z = np.empty(shape, np.float32), np.empty(shape, np.float32), np.empty(shape, np.float32)
    _run(_unproject_grid, disp, rows, cols, float(gw - 1), float(gh - 1), math.tan(math.radians(fov) / 2), float(h["aspect"]),
         float(h["extent_x"]), float(h["extent_y"]), float(zfar), X, Y, Z)
    return X, Y, Z


_vert_lru: OrderedDict = OrderedDict()  # (id(scene), key) -> scene: bounds the cached vertex grids globally
_VERT_MAX = 3  # ~17 MB each for a 'balanced' reconstruction


def _scene_vertices(sc: SceneData, fov: float, zfar: float, stride: int) -> dict:
    """Unprojected (world space) vertex grids of all layers for one (fov, zfar, stride), cached."""
    key = (round(fov, 5), round(zfar, 5), stride)
    v = sc.verts.get(key)
    if v is not None:
        if (id(sc), key) in _vert_lru:
            _vert_lru.move_to_end((id(sc), key))
        return v
    g = _scene_grids(sc, stride)
    gh, gw = sc.disp.shape
    v = {"main": _unproject(sc, sc.disp, g.rows, g.cols, gw, gh, fov, zfar),
         "bg": _unproject(sc, sc.bg_disp, g.rows, g.cols, gw, gh, fov, zfar)}
    for name, keep in (("main", g.keep), ("bg", g.keep_bg)):
        X, Y, Z = v[name]
        k = np.empty_like(keep)
        _run(_drop_stretched, X, Y, Z, keep, STRETCH_MIN_COS, k)
        v["keep_" + name] = k
    if sc.back is not None:
        bh, bw = sc.back.shape
        v["back"] = _unproject(sc, sc.back, np.arange(bh, dtype=np.int64), np.arange(bw, dtype=np.int64), bw, bh, fov, zfar)
    sc.verts[key] = v
    _vert_lru[(id(sc), key)] = sc  # holds the scene, so its id cannot be reused meanwhile
    while len(_vert_lru) > _VERT_MAX:
        (_, old_key), old = _vert_lru.popitem(last=False)
        old.verts.pop(old_key, None)
    return v


# --------------------------------------------------------------------------- camera


def _rot_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """three.js Euler(pitch, yaw, 0, 'YXZ'): R = Ry(yaw) Rx(pitch)."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


def _rot_look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera orientation looking at ``target`` (three.js Matrix4.lookAt semantics)."""
    up = np.asarray(up, float)
    z = eye - target
    if np.linalg.norm(z) < 1e-12:
        z = np.array([0.0, 0.0, 1.0])
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:
        z = z + (np.array([0.0001, 0, 0]) if abs(up[2]) == 1 else np.array([0, 0, 0.0001]))
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


@dataclass
class _Frame:
    W: int
    H: int
    nbands: int
    gdepth: np.ndarray
    glayer: np.ndarray
    gu: np.ndarray
    gv: np.ndarray
    ga: np.ndarray
    gs: np.ndarray
    gr: np.ndarray


def _frame(W: int, H: int, shading: bool = False) -> _Frame:
    """G-buffer for one frame (pixels with layer 255 are empty; the other channels are only read
    where a surface was written). ~25 bytes per pixel."""
    n = W * H
    nb = int(np.clip(H // 24, 1, 8 * max(1, numba.get_num_threads())))
    e = lambda k: np.empty(k, np.float32)  # noqa: E731
    return _Frame(W, H, nb, e(n), np.full(n, _EMPTY, np.uint8), e(n), e(n), e(n), e(n if shading else 1), e(n))


def _bounds(SX, SY, IW, keep) -> np.ndarray:
    gh, gw = SX.shape
    bnd = np.empty(((gh - 1 + BLOCK - 1) // BLOCK, (gw - 1 + BLOCK - 1) // BLOCK, 4), np.float32)
    _run(_block_bounds, SX, SY, IW, keep, BLOCK, bnd)
    return bnd


def _project(XYZ, valid_src, R, P, s, W, H, near):
    X, Y, Z = XYZ
    SX, SY, IW = np.empty_like(X), np.empty_like(X), np.empty_like(X)
    _run(_project_grid, X, Y, Z, valid_src, R, P, float(s), W / 2.0, H / 2.0, float(near), SX, SY, IW)
    return SX, SY, IW


@njit(parallel=True, cache=True)
def _probe(a):
    for i in prange(a.shape[0]):
        a[i] = i


_serial_twins: dict = {}
_parallel_ok: bool | None = None


def _parallel_is_threadsafe() -> bool:
    """numba's 'workqueue' threading layer (used when neither TBB nor OpenMP can be loaded)
    aborts the process if two threads launch parallel kernels at once - e.g. a frame while a
    reconstruction job runs. Then the renderer uses serial twins of its kernels instead."""
    global _parallel_ok
    if _parallel_ok is None:
        _probe(np.zeros(8))
        try:
            _parallel_ok = numba.threading_layer() != "workqueue"
        except ValueError:  # pragma: no cover
            _parallel_ok = True
    return _parallel_ok


def _run(fn, *args, dynamic: bool = False):
    """Run a kernel; band kernels use dynamic scheduling (bands differ a lot in cost)."""
    if not _parallel_is_threadsafe():
        twin = _serial_twins.get(fn)
        if twin is None:
            twin = _serial_twins[fn] = njit(fastmath=True)(fn.py_func)  # prange runs as range
        twin(*args)
    elif dynamic:
        with numba.parallel_chunksize(1):
            fn(*args)
    else:
        fn(*args)


# --------------------------------------------------------------------------- scene rendering


@dataclass
class SceneParams:
    position: tuple = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0
    look_at: tuple | None = None
    fov: float = 60.0
    zfar: float = 8.0
    width: int = 960
    height: int = 600
    quality: str = "preview"
    show_background: bool = True
    mode: str = "color"
    margin_fade: float = 0.35
    environment: bool = True  # far environment panorama behind everything (when one is given)


PREVIEW_STRIDE = 1  # minimum vertex stride of 'preview' frames
STRETCH_MIN_COS = float(__import__("os").environ.get("LUMEN_STRETCH_COS", 0.06))  # triangles seen more edge-on than this from the photo's viewpoint are not drawn (scene.ts: STRETCH_MIN_COS)
PREVIEW_MAX_VERTS = 160_000  # 'preview' frames use a coarser grid (stride 2, 3, ...) above this many vertices


def preview_stride(gw: int, gh: int) -> int:
    """Grid stride of 'preview' frames: large (wide-margin) grids are drawn at a coarser level of
    detail while the camera moves; the 'full' frame after it always uses every vertex."""
    st = PREVIEW_STRIDE
    while (gw / st) * (gh / st) > PREVIEW_MAX_VERTS and st < 4:
        st += 1
    return st


def render_scene(sc: SceneData, tex_main: MipTexture, tex_bg: MipTexture, tex_back: MipTexture | None, p: SceneParams,
                 env: MipTexture | None = None) -> tuple[np.ndarray, dict]:
    """Render the layered scene; returns (H x W x 3 uint8 RGB, info). ``env``: optional far
    environment panorama (see recon.environment), shown wherever no surface is drawn."""
    h = sc.header
    W, H = int(p.width), int(p.height)
    full = p.quality == "full"
    stride = 1 if full else preview_stride(sc.disp.shape[1], sc.disp.shape[0])
    fov = float(p.fov)
    zfar = float(p.zfar)
    verts = _scene_vertices(sc, fov, zfar, stride)
    g = _scene_grids(sc, stride)
    P = np.asarray(p.position, np.float64)
    if p.look_at is not None and np.linalg.norm(np.asarray(p.look_at, float) - P) > 1e-9:
        R = _rot_look_at(P, np.asarray(p.look_at, np.float64))
    else:
        R = _rot_yaw_pitch(float(p.yaw), float(p.pitch))
    R = np.ascontiguousarray(R, np.float64)
    aspect = float(h["aspect"])
    tanx = math.tan(math.radians(fov) / 2)
    A = W / H
    fit_tan_y = tanx / aspect if A > aspect else tanx / A
    s = H / (2.0 * fit_tan_y)
    f = _frame(W, H)
    tex_back = tex_back or tex_bg
    layers = [("main", verts["keep_main"], g.disp, g.u, g.v, tex_main, g.triangles)]
    if p.show_background:
        layers.append(("bg", verts["keep_bg"], g.bg_disp, g.u, g.v, tex_bg, g.bg_triangles))
        if sc.back is not None:
            bh, bw = sc.back.shape
            keep_all = _keep_all(bh, bw)
            layers.append(("back", keep_all, sc.back, np.arange(bw) / (bw - 1.0), np.arange(bh) / (bh - 1.0), tex_back, 2 * (bh - 1) * (bw - 1)))
    fine_verts = _scene_vertices(sc, fov, zfar, 1) if stride > 1 else None
    fine_grid = _scene_grids(sc, 1) if stride > 1 else None
    tris = 0
    for li, (name, keep, disp, u, v, tex, nt) in enumerate(layers):
        SX, SY, IW = _project(verts[name], disp, R, P, s, W, H, NEAR_SCENE)
        bnd = _bounds(SX, SY, IW, keep)
        _run(_raster_grid, SX, SY, IW, disp, disp, disp, keep, u, v, bnd, BLOCK, W, H, f.nbands,
             li, _TIE_EPS, tex.area, 0.0, 0.0, 1, 1, f.gdepth, f.glayer, f.gu, f.gv, f.ga, f.gs, f.gr, dynamic=True)
        tris += nt
        if fine_verts is not None and name in ("main", "bg"):
            detail_keep = _preview_detail_keep(fine_verts["keep_"+name], keep, g.rows, g.cols)
            if detail_keep.any():
                fine_disp = sc.disp if name == "main" else sc.bg_disp
                fx, fy, fw = _project(fine_verts[name], fine_disp, R, P, s, W, H, NEAR_SCENE)
                bounds = _bounds(fx, fy, fw, detail_keep)
                _run(_raster_grid, fx, fy, fw, fine_disp, fine_disp, fine_disp, detail_keep, fine_grid.u, fine_grid.v,
                     bounds, BLOCK, W, H, f.nbands, li, _TIE_EPS, tex.area, 0., 0., 1, 1,
                     f.gdepth, f.glayer, f.gu, f.gv, f.ga, f.gs, f.gr, dynamic=True)
                tris += int(detail_keep.sum())*2
    out = np.empty((H, W, 3), np.uint8)
    mode = {"color": 0, "depth": 1, "layers": 2}.get(p.mode, 0)
    use_env = env is not None and p.environment
    e = env if use_env else _DUMMY_TEX
    _run(_shade_scene, f.glayer, f.gu, f.gv, f.ga, f.gr, W, H, *tex_main.args(), *tex_bg.args(), *tex_back.args(),
         mode, 2 if full else 1, float(p.margin_fade), float(h["extent_x"]), float(h["extent_y"]), _LUT_SRGB,
         e.buf, int(e.w[0]), int(e.h[0]), 1 if use_env else 0, R, float(s), W / 2.0, H / 2.0, out)
    return out, {"triangles": tris, "stride": stride}


_keep_all_cache: dict = {}


def _keep_all(bh: int, bw: int) -> np.ndarray:
    k = _keep_all_cache.get((bh, bw))
    if k is None:
        k = np.ones((bh - 1, bw - 1), np.uint8)
        _keep_all_cache[(bh, bw)] = k
    return k


# --------------------------------------------------------------------------- body rendering


@dataclass
class BodyParams:
    position: tuple = (0.0, 0.05, 2.3)
    target: tuple = (0.0, 0.0, 0.0)
    fov: float = 32.0
    width: int = 960
    height: int = 600
    quality: str = "preview"
    style: str = "wireframe"
    colors: str = "depth"
    line_step: int = 4
    hide_hidden: bool = True
    relief: float = 1.0


def _js_round(x: float) -> int:
    return int(math.floor(x + 0.5))


def render_body(bd: BodyData, tex: MipTexture | None, p: BodyParams) -> tuple[np.ndarray, dict]:
    """Render the body model; returns (H x W x 3 uint8 RGB, info)."""
    hd = bd.header
    rows, cols = bd.z.shape
    W, H = int(p.width), int(p.height)
    full = p.quality == "full"
    colors = {"depth": 0, "photo": 1, "hologram": 2}.get(p.colors, 0)
    if colors == 1 and tex is None:
        colors = 0
    X, Y, Z = (np.empty((rows, cols), np.float32) for _ in range(3))
    N = np.empty((rows, cols, 3), np.float32)
    SF, SB = np.empty((rows, cols), np.float32), np.empty((rows, cols), np.float32)
    zs = float(hd["depth_scale"]) * float(p.relief)
    _run(_body_vertices, bd.z, bd.keep, float(hd["aspect"]), zs, X, Y, Z, N, SF, SB, _LIGHT, 0.34, 0.78, 0.10)
    P = np.asarray(p.position, np.float64)
    R = np.ascontiguousarray(_rot_look_at(P, np.asarray(p.target, np.float64)), np.float64)
    tany = math.tan(math.radians(float(p.fov)) / 2)
    s = H / (2.0 * tany)
    SX, SY, IW = _project((X, Y, Z), bd.z, R, P, s, W, H, NEAR_BODY)
    u = np.arange(cols) / (cols - 1.0)
    v = np.arange(rows) / (rows - 1.0)
    style = p.style if p.style in ("wireframe", "scanlines", "points", "solid") else "wireframe"
    f = _frame(W, H, shading=style == "solid")
    tris = int(bd.qkeep.sum()) * 2
    if style == "solid":
        bnd = _bounds(SX, SY, IW, bd.qkeep)
        tarea = tex.area if tex is not None else 1.0
        _run(_raster_grid, SX, SY, IW, bd.z, SF, SB, bd.qkeep, u, v, bnd, BLOCK, W, H, f.nbands,
             0, 0.0, tarea, 0.0, 0.0, 2, 0, f.gdepth, f.glayer, f.gu, f.gv, f.ga, f.gs, f.gr, dynamic=True)
        out = np.empty((H, W, 3), np.uint8)
        t = tex if tex is not None else _DUMMY_TEX
        _run(_shade_body, f.glayer, f.gu, f.gv, f.ga, f.gs, f.gr, W, H, colors, *t.args(), 2 if full else 1, _LUT_UNIT, out)
        return out, {"triangles": tris}
    C = np.empty((rows, cols, 3), np.float32)
    grid_tex = tex.resized(cols, rows) if (colors == 1 and tex is not None) else _DUMMY_GRID
    _run(_body_colors, bd.z, colors, grid_tex, C)
    fb = np.empty((H, W, 3), np.float32)
    fb[:] = np.array(BODY_CLEAR, np.float32) / 255.0
    if style == "points":
        pdepth = np.zeros(W * H, np.float32)
        spacing = 1.0 / (rows - 1)
        # dots about half the projected grid spacing: dense at a distance, a visible point cloud up close
        _run(_draw_points, SX, SY, IW, C, bd.keep, spacing * s * 0.55, W, H, f.nbands, pdepth, fb, dynamic=True)
        tris = 0
    else:
        step = max(1, _js_round(float(p.line_step)))
        if style == "scanlines":
            hrows = np.arange(0, rows, max(1, _js_round(step / 2)), dtype=np.int64)
            vcols = np.zeros(0, np.int64)
        else:
            hrows = np.arange(0, rows, step, dtype=np.int64)
            vcols = np.arange(0, cols, step, dtype=np.int64)
        hide = bool(p.hide_hidden)
        if hide:
            # occluder: the body surface pushed back by its depth slope (like glPolygonOffset)
            bnd = _bounds(SX, SY, IW, bd.qkeep)
            _run(_raster_grid, SX, SY, IW, bd.z, SF, SB, bd.qkeep, u, v, bnd, BLOCK, W, H, f.nbands,
                 0, 0.0, 1.0, 1.5, 2e-3, 0, 0, f.gdepth, f.glayer, f.gu, f.gv, f.ga, f.gs, f.gr, dynamic=True)
        else:
            tris = 0
        opacity = 0.85 if colors == 2 else 0.95
        _run(_draw_lines, SX, SY, IW, C, bd.keep, hrows, vcols, W, H, f.nbands, opacity, hide, f.gdepth, f.glayer, fb, dynamic=True)
    out = np.empty((H, W, 3), np.uint8)
    _run(_to_u8, fb, out)
    return out, {"triangles": tris}


_DUMMY_TEX = MipTexture(np.full((2, 2, 3), 128, np.uint8))
_DUMMY_GRID = np.zeros((1, 1, 3), np.uint8)


# --------------------------------------------------------------------------- caches


class _LRU:
    def __init__(self, cap: int):
        self.cap = cap
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key, make):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key]
        val = make()
        with self.lock:
            self.d[key] = val
            self.d.move_to_end(key)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)
        return val

    def clear(self):
        with self.lock:
            self.d.clear()


_scenes = _LRU(4)
_bodies = _LRU(4)
_textures = _LRU(10)
_envs = _LRU(6)  # (scene, main texture, fov) -> environment panorama
_render_lock = threading.Lock()  # one frame at a time: bounded memory, no concurrent parallel launches


def clear_caches() -> None:
    for c in (_scenes, _bodies, _textures, _envs):
        c.clear()


def scene_environment_texture(sc: SceneData, tex_main: MipTexture, fov: float, tex_bg: MipTexture | None = None) -> MipTexture:
    """Far environment panorama derived from the scene (outpainted main texture + sky), as a texture."""
    from .environment import scene_environment

    # Background disparity is sparse (-1 means absent, not sky). Use main
    # depth outside completed regions rather than treating missing data as sky.
    depth = np.where(sc.bg_disp >= -0.0801, sc.bg_disp, sc.disp) if tex_bg is not None else sc.disp
    pano = scene_environment(depth, (tex_bg or tex_main).rgb, sc.header, fov)
    return MipTexture((np.clip(pano, 0, 1) * 255 + 0.5).astype(np.uint8))


def _environment(names: dict, sc: SceneData, tm: MipTexture, fov: float, tb: MipTexture | None = None) -> MipTexture:
    key = (names["scene"], names["main_texture"], names.get("bg_texture"), round(float(fov), 1))
    return _envs.get(key, lambda: scene_environment_texture(sc, tm, fov, tb))


def _texture(name: str, path) -> MipTexture:
    return _textures.get(name, lambda: decode_texture(path.read_bytes()))


# --------------------------------------------------------------------------- request handling


def _num(v, name: str, default=None) -> float:
    if v is None:
        if default is None:
            raise InvalidInputError(f"Missing value '{name}'.")
        return float(default)
    if isinstance(v, bool):
        raise InvalidInputError(f"'{name}' must be a number.")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise InvalidInputError(f"'{name}' must be a number.") from None
    if not math.isfinite(f):
        raise InvalidInputError(f"'{name}' must be a finite number.")
    return f


def _vec3(v, name: str, default=None) -> tuple:
    if v is None and default is not None:
        return tuple(default)
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise InvalidInputError(f"'{name}' must be a list of three numbers.")
    return tuple(_num(x, name) for x in v)


def _size(v, name: str, default: int) -> int:
    return int(np.clip(round(_num(v, name, default)), MIN_SIZE, MAX_SIZE))


def _name(files: dict, key: str, required: bool = True) -> str | None:
    v = files.get(key)
    if v is None and not required:
        return None
    if not isinstance(v, str) or not v:
        raise InvalidInputError(f"Missing file name '{key}'.")
    return v


def _choice(v, allowed: tuple, default: str) -> str:
    return v if isinstance(v, str) and v in allowed else default


def scene_params(body: dict) -> tuple[dict, SceneParams]:
    if not isinstance(body, dict):
        raise InvalidInputError("Invalid request.")
    files = body.get("files")
    if not isinstance(files, dict):
        raise InvalidInputError("Missing 'files'.")
    names = {"scene": _name(files, "scene"), "main_texture": _name(files, "main_texture"), "bg_texture": _name(files, "bg_texture"),
             "back_texture": _name(files, "back_texture", required=False)}
    cam = body.get("camera") or {}
    if not isinstance(cam, dict):
        raise InvalidInputError("'camera' must be an object.")
    look = cam.get("look_at")
    p = SceneParams(
        position=_vec3(cam.get("position"), "camera.position", (0.0, 0.0, 0.0)),
        yaw=_num(cam.get("yaw"), "camera.yaw", 0.0),
        pitch=_num(cam.get("pitch"), "camera.pitch", 0.0),
        look_at=None if look is None else _vec3(look, "camera.look_at"),
        fov=float(np.clip(_num(body.get("fov"), "fov", 60.0), 5.0, 150.0)),
        zfar=float(np.clip(_num(body.get("zfar"), "zfar", 8.0), 1.05, 1000.0)),
        width=_size(body.get("width"), "width", 960),
        height=_size(body.get("height"), "height", 600),
        quality=_choice(body.get("quality"), ("preview", "full"), "preview"),
        show_background=bool(body.get("show_background", True)),
        mode=_choice(body.get("mode"), ("color", "depth", "layers"), "color"),
        margin_fade=float(np.clip(_num(body.get("margin_fade"), "margin_fade", 0.35), 0.0, 1.0)),
        environment=bool(body.get("environment", True)),
    )
    return names, p


def body_params(body: dict) -> tuple[dict, BodyParams]:
    if not isinstance(body, dict):
        raise InvalidInputError("Invalid request.")
    files = body.get("files")
    if not isinstance(files, dict):
        raise InvalidInputError("Missing 'files'.")
    names = {"body": _name(files, "body"), "texture": _name(files, "texture", required=False)}
    cam = body.get("camera") or {}
    if not isinstance(cam, dict):
        raise InvalidInputError("'camera' must be an object.")
    p = BodyParams(
        position=_vec3(cam.get("position"), "camera.position", (0.0, 0.05, 2.3)),
        target=_vec3(cam.get("target"), "camera.target", (0.0, 0.0, 0.0)),
        fov=float(np.clip(_num(cam.get("fov"), "camera.fov", 32.0), 5.0, 150.0)),
        width=_size(body.get("width"), "width", 960),
        height=_size(body.get("height"), "height", 600),
        quality=_choice(body.get("quality"), ("preview", "full"), "preview"),
        style=_choice(body.get("style"), ("wireframe", "scanlines", "points", "solid"), "wireframe"),
        colors=_choice(body.get("colors"), ("depth", "photo", "hologram"), "depth"),
        line_step=int(np.clip(round(_num(body.get("line_step"), "line_step", 4)), 1, 64)),
        hide_hidden=bool(body.get("hide_hidden", True)),
        relief=float(np.clip(_num(body.get("relief"), "relief", 1.0), 0.0, 5.0)),
    )
    return names, p


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:  # pragma: no cover
        raise InvalidInputError("Encoding the rendered frame failed.")
    return enc.tobytes()


def render_scene_request(body: dict, resolve) -> tuple[bytes, dict]:
    """Validate a POST /api/recon/render body, render and encode. ``resolve`` maps a file name to a path."""
    t0 = time.perf_counter()
    names, p = scene_params(body)
    paths = {k: resolve(v) for k, v in names.items() if v is not None}  # raises NotFoundError (404)
    with _render_lock:
        sc = _scenes.get(names["scene"], lambda: parse_scene(paths["scene"].read_bytes()))
        tm = _texture(names["main_texture"], paths["main_texture"])
        tb = _texture(names["bg_texture"], paths["bg_texture"])
        tk = _texture(names["back_texture"], paths["back_texture"]) if names["back_texture"] else None
        env = _environment(names, sc, tm, p.fov, tb) if p.environment else None
        img, info = render_scene(sc, tm, tb, tk, p, env=env)
    data = encode_jpeg(img)
    info["ms"] = (time.perf_counter() - t0) * 1000.0
    return data, info


def environment_request(body: dict, resolve) -> tuple[bytes, dict]:
    """POST /api/recon/environment: the far environment panorama of a scene as a JPEG (the WebGL
    viewer draws the same panorama the CPU renderer uses). Body: {files: {scene, main_texture}, fov}."""
    t0 = time.perf_counter()
    if not isinstance(body, dict) or not isinstance(body.get("files"), dict):
        raise InvalidInputError("Missing 'files'.")
    files = body["files"]
    names = {"scene": _name(files, "scene"), "main_texture": _name(files, "main_texture"), "bg_texture": _name(files, "bg_texture", required=False)}
    fov = float(np.clip(_num(body.get("fov"), "fov", 60.0), 5.0, 150.0))
    paths = {k: resolve(v) for k, v in names.items() if v is not None}
    with _render_lock:
        sc = _scenes.get(names["scene"], lambda: parse_scene(paths["scene"].read_bytes()))
        tm = _texture(names["main_texture"], paths["main_texture"])
        tb = _texture(names["bg_texture"], paths["bg_texture"]) if names["bg_texture"] else None
        env = _environment(names, sc, tm, fov, tb)
    data = encode_jpeg(env.rgb, quality=90)
    return data, {"ms": (time.perf_counter() - t0) * 1000.0, "triangles": 0}


def render_body_request(body: dict, resolve) -> tuple[bytes, dict]:
    """Validate a POST /api/body3d/render body, render and encode."""
    t0 = time.perf_counter()
    names, p = body_params(body)
    paths = {k: resolve(v) for k, v in names.items() if v is not None}
    with _render_lock:
        bd = _bodies.get(names["body"], lambda: parse_body(paths["body"].read_bytes()))
        tex = _texture(names["texture"], paths["texture"]) if names["texture"] else None
        img, info = render_body(bd, tex, p)
    data = encode_jpeg(img)
    info["ms"] = (time.perf_counter() - t0) * 1000.0
    return data, info


# --------------------------------------------------------------------------- JIT warm-up

_warm_lock = threading.Lock()
_warm = False


def _synthetic_scene() -> tuple[SceneData, MipTexture, MipTexture, MipTexture]:
    gh, gw = 24, 32
    yy, xx = np.mgrid[0:gh, 0:gw]
    disp = (0.3 + 0.4 * (xx > gw // 2)).astype(np.float32)
    keep = np.ones((gh - 1, gw - 1), np.uint8)
    keep[:, gw // 2 - 1] = 0
    bg = np.where(np.abs(xx - gw // 2) < 3, 0.3, -1.0).astype(np.float32)
    keep_bg = ((bg[:-1, :-1] > -0.5) & (bg[1:, 1:] > -0.5) & (bg[:-1, 1:] > -0.5) & (bg[1:, :-1] > -0.5)).astype(np.uint8)
    back = np.full((7, 9), 0.2, np.float32)
    header = {"grid_w": gw, "grid_h": gh, "extent_x": 1.1, "extent_y": 1.1, "aspect": 1.4, "default_fov": 60.0, "default_zfar": 8.0,
              "safe_radius": 0.35, "back_w": 9, "back_h": 7}
    rng = np.random.default_rng(0)
    tex = MipTexture((rng.random((48, 64, 3)) * 255).astype(np.uint8))
    return SceneData(header, disp, keep, bg, keep_bg, back), tex, tex, tex


def _synthetic_body() -> tuple[BodyData, MipTexture]:
    rows, cols = 40, 20
    yy, xx = np.mgrid[0:rows, 0:cols]
    keep = (((xx - cols / 2) / (cols / 2.2)) ** 2 + ((yy - rows / 2) / (rows / 2.2)) ** 2 < 1).astype(np.uint8)
    z = np.where(keep > 0, 0.5 + 0.3 * np.cos((xx - cols / 2) / cols * 3), 0).astype(np.float32)
    qkeep = (keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]).astype(np.uint8)
    header = {"cols": cols, "rows": rows, "aspect": 0.5, "depth_scale": 0.25}
    rng = np.random.default_rng(1)
    return BodyData(header, z, keep, qkeep), MipTexture((rng.random((80, 40, 3)) * 255).astype(np.uint8))


def warmup() -> float:
    """Compile (or load from the on-disk cache) every kernel; returns seconds spent."""
    global _warm
    t0 = time.perf_counter()
    with _warm_lock, _render_lock:
        if _warm:
            return 0.0
        sc, tm, tb, tk = _synthetic_scene()
        env = MipTexture(np.full((16, 32, 3), 90, np.uint8))
        for q in ("preview", "full"):
            for mode in ("color", "depth", "layers"):
                render_scene(sc, tm, tb, tk, SceneParams(width=64, height=48, quality=q, mode=mode, position=(0.05, 0.0, 0.0)), env=env)
        bd, tex = _synthetic_body()
        for style in ("wireframe", "scanlines", "points", "solid"):
            for colors in ("depth", "photo"):
                render_body(bd, tex, BodyParams(width=64, height=64, style=style, colors=colors, quality="full" if style == "solid" else "preview"))
        render_body(bd, tex, BodyParams(width=64, height=64, style="wireframe", hide_hidden=False))
        encode_jpeg(np.zeros((8, 8, 3), np.uint8))
        _warm = True
    return time.perf_counter() - t0


def warmup_in_background() -> None:
    def run():
        try:
            warmup()
        except Exception:  # pragma: no cover - a failed warm-up only means the first frame compiles
            pass

    threading.Thread(target=run, name="lumen-render-warmup", daemon=True).start()
