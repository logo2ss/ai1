"""Perspective-correct, z-buffered CPU rendering of a closed person volume."""
from __future__ import annotations

import io
import math
import time

import cv2
import numpy as np
from numba import njit, prange

from ..errors import InvalidInputError
from . import render as r

_persons = r._LRU(4)


@njit(cache=True, parallel=True)
def _triangles(projected, uv, faces, shade, texture, out):
    H, W = out.shape[:2]
    th, tw = texture.shape[:2]
    # Independent horizontal bands: no z-buffer races and bounded memory.
    for band in prange((H+23)//24):
        ylo, yhi = band*24, min(H, (band+1)*24)
        depth = np.zeros((yhi-ylo, W), np.float32)
        for ti in range(len(faces)):
            ia, ib, ic = faces[ti]
            ax, ay, aw = projected[ia]
            bx, by, bw = projected[ib]
            cx, cy, cw = projected[ic]
            if min(aw, bw, cw) <= 0:
                continue
            yl, yh = max(ylo, int(math.floor(min(ay, by, cy)))), min(yhi-1, int(math.ceil(max(ay, by, cy))))
            xl, xh = max(0, int(math.floor(min(ax, bx, cx)))), min(W-1, int(math.ceil(max(ax, bx, cx))))
            den = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
            if yl > yh or xl > xh or abs(den) < 1e-9:
                continue
            for y in range(yl, yh+1):
                for x in range(xl, xh+1):
                    a = ((by-cy)*(x+0.5-cx)+(cx-bx)*(y+0.5-cy))/den
                    b = ((cy-ay)*(x+0.5-cx)+(ax-cx)*(y+0.5-cy))/den
                    c = 1-a-b
                    if min(a,b,c) < -1e-6:
                        continue
                    iw = a*aw+b*bw+c*cw
                    if iw <= depth[y-ylo, x]:
                        continue
                    depth[y-ylo, x] = iw
                    u = (a*aw*uv[ia,0]+b*bw*uv[ib,0]+c*cw*uv[ic,0])/iw*(tw-1)
                    v = (a*aw*uv[ia,1]+b*bw*uv[ib,1]+c*cw*uv[ic,1])/iw*(th-1)
                    u, v = min(tw-1., max(0., u)), min(th-1., max(0., v))
                    u0, v0 = int(u), int(v)
                    u1, v1 = min(tw-1,u0+1), min(th-1,v0+1)
                    du, dv = u-u0, v-v0
                    for ch in range(3):
                        color = ((texture[v0,u0,ch]*(1-du)+texture[v0,u1,ch]*du)*(1-dv)+
                                 (texture[v1,u0,ch]*(1-du)+texture[v1,u1,ch]*du)*dv)
                        out[y,x,ch] = min(255., max(0., color*shade[ti]))


def load_person(blob):
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as d:
            v, uv, f = d["vertices"], d["uv"], d["faces"]
        if v.ndim != 2 or v.shape[1] != 3 or uv.shape != (len(v), 2) or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError
        if len(f) > 1500000 or len(v) > 1000000 or not len(f) or f.min() < 0 or f.max() >= len(v):
            raise ValueError
        if not np.isfinite(v).all() or not np.isfinite(uv).all():
            raise ValueError
        return v.astype(np.float32), uv.astype(np.float32), f.astype(np.int32)
    except (ValueError, KeyError, OSError):
        raise InvalidInputError("Invalid Person 3D model.") from None


def render_person(model, texture, position=(0, 0, 2.3), target=(0, 0, 0), width=800, height=600, fov=32):
    vertices, uv, faces = model
    eye = np.asarray(position, float)
    rot = r._rot_look_at(eye, np.asarray(target, float))
    camera = (vertices-eye) @ rot
    iw = 1 / np.maximum(-camera[:, 2], 0.01)
    scale = height/(2*math.tan(math.radians(fov)/2))
    projected = np.stack([width/2+camera[:, 0]*iw*scale, height/2-camera[:, 1]*iw*scale,
                          np.where(camera[:, 2] < -0.02, iw, 0)], 1).astype(np.float32)
    n = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    light = np.array([-0.25, 0.4, 1.0]); light /= np.linalg.norm(light)
    # Double-sided rim normals; subtle lighting preserves the photograph.
    shade = (0.88+0.12*np.abs(n @ light)).astype(np.float32)
    out = np.full((height, width, 3), (14, 17, 23), np.uint8)
    with r._render_lock:
        r._run(_triangles, projected, uv, faces, shade, texture, out)
    return out, {"triangles": len(faces)}


def render_person_request(body, resolve):
    start = time.perf_counter()
    if not isinstance(body, dict) or not isinstance(body.get("files"), dict):
        raise InvalidInputError("Missing person files.")
    files = body["files"]
    name, texname = r._name(files, "person"), r._name(files, "texture")
    model = _persons.get(name, lambda: load_person(resolve(name).read_bytes()))
    # Preserve the original aspect; cache the decoded texture with the mesh.
    texture = _persons.get("texture:"+texname, lambda: cv2.cvtColor(cv2.imdecode(np.frombuffer(resolve(texname).read_bytes(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    cam = body.get("camera") or {}
    if not isinstance(cam, dict):
        raise InvalidInputError("Invalid person camera.")
    pos = r._vec3(cam.get("position"), "camera.position", (0, 0, 2.3))
    target = r._vec3(cam.get("target"), "camera.target", (0, 0, 0))
    out, info = render_person(model, texture, pos, target,
        r._size(body.get("width"), "width", 800), r._size(body.get("height"), "height", 600),
        float(np.clip(r._num(cam.get("fov"), "camera.fov", 32), 10, 100)))
    info["ms"] = (time.perf_counter()-start)*1000
    return r.encode_jpeg(out), info
