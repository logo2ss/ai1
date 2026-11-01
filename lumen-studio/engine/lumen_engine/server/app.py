"""Local HTTP API (bound to 127.0.0.1 only) used by the desktop UI.

Every request must carry the per-session token generated at start-up
(header ``X-Lumen-Token`` or ``?token=`` for image URLs), so other local
programs or web pages cannot use the engine.
"""

from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import __version__, mpworker, store
from ..config import get_settings, update_settings
from ..errors import InvalidInputError, LumenError, NotFoundError
from ..hardware import clear_sessions, hardware_info
from ..imaging.io import decode_mask, encode_image, resize_max
from ..models import manager
from . import jobs

TOKEN = secrets.token_urlsafe(24)


def _u8(rgb: np.ndarray) -> np.ndarray:
    """Compact uint8 copy for the MediaPipe helper process (no large float temporaries)."""
    return cv2.convertScaleAbs(rgb, alpha=255.0)
_face_cache: dict[str, list] = {}


def create_app(static_dir: str | None = None, token: str | None = None) -> FastAPI:
    global TOKEN
    if token:
        TOKEN = token
    app = FastAPI(title="Lumen Engine", version=__version__, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            t = request.headers.get("x-lumen-token") or request.query_params.get("token")
            if not t or not secrets.compare_digest(t, TOKEN):
                return JSONResponse({"code": "unauthorized", "message": "Missing or invalid session token."}, status_code=401)
        resp = await call_next(request)
        # The UI must never load remote resources.
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; worker-src 'self' blob:; object-src 'none'"
        )
        return resp

    @app.exception_handler(LumenError)
    async def lumen_error(_: Request, exc: LumenError):
        return JSONResponse(exc.to_dict(), status_code=exc.status)

    # compile (or load from the disk cache) the CPU 3D renderer in the background at start-up,
    # so the first frame in a browser without WebGL is fast
    app.router.add_event_handler("startup", _warm_renderer)

    # ------------------------------------------------------------------ system
    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "offline": True}

    @app.get("/api/system")
    def system():
        from dataclasses import asdict

        from ..face import backend

        return {
            "hardware": hardware_info(),
            "settings": asdict(get_settings()),
            "storage_bytes": store.usage_bytes(),
            "version": __version__,
            "mediapipe": {"usable": backend.mediapipe_status() is None, "reason": backend.mediapipe_status(), "hint": backend.VM_HINT},
            "face_compat": backend.compat_info(),
        }

    @app.put("/api/settings")
    async def put_settings(request: Request):
        body = await request.json()
        s = update_settings(**body)
        clear_sessions()
        from dataclasses import asdict

        return asdict(s)

    @app.delete("/api/storage")
    def clear_storage():
        store.clear()
        _face_cache.clear()
        return {"ok": True}

    # ------------------------------------------------------------------ models
    @app.get("/api/models")
    def models():
        return manager.status()

    @app.post("/api/models/{model_id}/install")
    def install(model_id: str):
        manager.get_spec(model_id)
        threading.Thread(target=lambda: _safe(manager.download, model_id), daemon=True).start()
        return {"ok": True}

    @app.post("/api/models/{model_id}/import")
    async def import_model(model_id: str, file: UploadFile = File(...)):
        tmp = manager.model_path(model_id).with_suffix(".upload")
        with open(tmp, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        try:
            manager.import_file(model_id, tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return {"ok": True}

    @app.delete("/api/models/{model_id}")
    def remove_model(model_id: str):
        manager.remove(model_id)
        return {"ok": True}

    # ------------------------------------------------------------------ images
    @app.post("/api/images")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            raise InvalidInputError("The file is empty.")
        try:
            image_id = await run_in_threadpool(store.put_bytes, data, file.filename or "")
        except MemoryError:
            raise InvalidInputError("This image is too large for the available memory.", hint="Use a smaller image or give the computer/VM more RAM.") from None
        return store.meta(image_id)

    @app.get("/api/images/{image_id}")
    def image(image_id: str, max: int = 0, format: str = "jpg", quality: int = 90):
        rgb, alpha = store.get(image_id)
        if max > 0:
            rgb, _ = resize_max(rgb, max)
            if alpha is not None:
                alpha, _ = resize_max(alpha, max)
        fmt = "png" if format == "png" else "jpg"
        data = encode_image(rgb, fmt, quality=quality, alpha=alpha if fmt == "png" else None)
        return Response(data, media_type=f"image/{'png' if fmt == 'png' else 'jpeg'}", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/images/{image_id}/meta")
    def image_meta(image_id: str):
        return store.meta(image_id)

    @app.get("/api/images/{image_id}/export")
    def export(image_id: str, format: str = "png", quality: int = 95, bits: int = 8):
        rgb, alpha = store.get(image_id)
        meta = store.meta(image_id)
        fmt = format.lower()
        if fmt not in ("png", "jpg", "jpeg", "tiff", "webp"):
            raise InvalidInputError("Unsupported export format.")
        text = {"Software": f"Lumen Studio {__version__}"}
        if get_settings().tag_exports and meta.get("operation"):
            text["Comment"] = f"Edited locally with Lumen Studio ({meta['operation']})."
        data = encode_image(rgb, fmt, quality=quality, alpha=alpha, icc_profile=store.icc(image_id), text=text, bit_depth=bits)
        ext = "jpg" if fmt == "jpeg" else fmt
        name = (meta.get("name") or "image").rsplit(".", 1)[0] or "image"
        media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "tiff": "image/tiff", "webp": "image/webp"}[fmt]
        return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}-lumen.{ext}"'})

    # ------------------------------------------------------------------ face
    _face_info: dict[str, dict] = {}

    def _faces(image_id: str):
        if image_id not in _face_cache:
            from ..face import backend

            rgb, _ = store.get(image_id)
            faces, info = backend.detect(rgb)
            _face_cache[image_id] = faces
            _face_info[image_id] = info
        return _face_cache[image_id]

    def _face(ref: dict):
        faces = _faces(ref["image_id"])
        i = int(ref.get("face_index", 0))
        if not (0 <= i < len(faces)):
            raise InvalidInputError("Please select a face in each image.")
        return faces[i]

    @app.post("/api/face/detect")
    async def face_detect(request: Request):
        body = await request.json()
        faces = await run_in_threadpool(_faces, body["image_id"])
        return {"faces": [f.to_dict() for f in faces], **_face_info.get(body["image_id"], {})}

    @app.post("/api/face/detect-box")
    async def face_detect_box(request: Request):
        from ..face import backend

        body = await request.json()
        rgb, _ = store.get(body["image_id"])
        f, info = await run_in_threadpool(backend.detect, rgb, tuple(body["box"]))
        faces = await run_in_threadpool(_faces, body["image_id"])
        faces.append(f)
        return {"faces": [x.to_dict() for x in faces], "index": len(faces) - 1, **info}

    @app.post("/api/face/analyze")
    async def face_analyze(request: Request):
        from ..face.swap import analyze_pair

        body = await request.json()
        s_img, _ = store.get(body["source"]["image_id"])
        t_img, _ = store.get(body["target"]["image_id"])
        fs = await run_in_threadpool(_face, body["source"])
        ft = await run_in_threadpool(_face, body["target"])
        return await run_in_threadpool(analyze_pair, s_img, fs, t_img, ft)

    @app.post("/api/face/swap")
    async def face_swap(request: Request):
        from ..face.swap import FaceSwapSettings, swap_face

        body = await request.json()
        src_ref, dst_ref = body["source"], body["target"]
        s_img, _ = store.get(src_ref["image_id"])
        t_img, t_alpha = store.get(dst_ref["image_id"])
        fs = await run_in_threadpool(_face, src_ref)
        ft = await run_in_threadpool(_face, dst_ref)
        settings = FaceSwapSettings.from_dict(body.get("settings"))
        name = store.meta(dst_ref["image_id"]).get("name", "")

        def work(progress):
            out, report, dbg = swap_face(s_img, fs, t_img, ft, settings, progress)
            rid = store.put(out, t_alpha, {"name": name, "operation": "face replacement", "icc_profile": store.icc(dst_ref["image_id"])})
            mid = store.put(np.dstack([dbg["mask"]] * 3), None, {"name": "mask"})
            return {"image_id": rid, "mask_id": mid, "report": _jsonable(report)}

        return {"job_id": jobs.submit("face_swap", work)}

    # ------------------------------------------------------------------ region
    @app.post("/api/region/smart-select")
    async def smart(request: Request):
        """Click-to-select. MediaPipe's interactive segmenter when it can run and is
        installed; otherwise the MediaPipe-free selector (U2-Net silueta, or a graph cut)."""
        from ..face import backend
        from ..models import manager as _m
        from ..region.smart_select_fallback import smart_select as fallback_select

        body = await request.json()
        rgb, _ = store.get(body["image_id"])
        pts = [tuple(p) for p in body["points"]]
        method = None
        if backend.mediapipe_status() is None and _m.is_installed("magic-touch"):
            try:
                m = await run_in_threadpool(lambda: backend.run_mediapipe("lumen_engine.mptasks.smart_select", _u8(rgb), pts, what="Smart select"))
                method = "mediapipe"
            except LumenError:
                method = None  # crashed / unavailable: fall back below
        if method is None:
            m, method = await run_in_threadpool(fallback_select, rgb, pts)
        data = encode_image(np.dstack([m] * 3), "png")
        return Response(data, media_type="image/png", headers={"X-Lumen-Method": method})

    @app.post("/api/region/refine-mask")
    async def region_refine_mask(payload: str = Form(...), mask: UploadFile = File(...)):
        """Snap a drawn selection to the object edges, only near what was drawn (selection editor)."""
        from ..region.refine import refine_selection

        body = json.loads(payload)
        rgb, _ = store.get(body["image_id"])
        w, h = (int(v) for v in body["size"])
        m = decode_mask(await mask.read(), (h, w))
        small = rgb if rgb.shape[:2] == (h, w) else cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        roi = tuple(int(v) for v in body["roi"]) if body.get("roi") else None
        out = await run_in_threadpool(refine_selection, small, m, roi, float(body.get("band", 12)))
        return Response(encode_image(np.dstack([out.astype(np.float32)] * 3), "png"), media_type="image/png")

    @app.post("/api/region/swap")
    async def region_swap(
        payload: str = Form(...),
        mask_a: UploadFile = File(...),
        mask_b: UploadFile = File(...),
    ):
        from ..region.swap import RegionSwapSettings, swap_regions

        body = json.loads(payload)
        a_id, b_id = body["image_a"], body["image_b"]
        img_a, alpha_a = store.get(a_id)
        img_b, alpha_b = store.get(b_id)
        ma = decode_mask(await mask_a.read(), img_a.shape[:2])
        mb = decode_mask(await mask_b.read(), img_b.shape[:2])
        if (ma > 0.5).sum() < 16 or (mb > 0.5).sum() < 16:
            raise InvalidInputError("Please draw a selection on both images.")
        settings = RegionSwapSettings.from_dict(body.get("settings"))
        direction = body.get("direction", "both")
        names = store.meta(a_id).get("name", ""), store.meta(b_id).get("name", "")

        def work(progress):
            res = swap_regions(img_a, ma, img_b, mb, settings, direction, progress)
            out = {}
            for key, (img, rep, _dbg), alpha, nm, src in (
                ("a", res.get("a", (None,) * 3), alpha_a, names[0], a_id),
                ("b", res.get("b", (None,) * 3), alpha_b, names[1], b_id),
            ):
                if img is not None:
                    out[key] = {"image_id": store.put(img, alpha, {"name": nm, "operation": "region exchange", "icc_profile": store.icc(src)}), "report": _jsonable(rep)}
            return out

        return {"job_id": jobs.submit("region_swap", work)}

    # ------------------------------------------------------------------ body proportions
    _body_cache: dict[str, dict] = {}

    def _body(image_id: str) -> dict:
        """Person mask + lattice: MediaPipe pose when it can run, otherwise segmentation + proportions."""
        if image_id not in _body_cache:
            from ..body.reshape import detect_body_fallback
            from ..face import backend
            from ..models import manager as _m

            rgb, _ = store.get(image_id)
            res = None
            if backend.mediapipe_status() is None and _m.is_installed("pose-landmarker"):
                try:
                    res = backend.run_mediapipe("lumen_engine.mptasks.detect_body", _u8(rgb), what="The body editor")
                    res["method"] = "mediapipe"
                except LumenError as e:
                    if not (e.details.get("crashed") or e.details.get("mediapipe_unavailable")):
                        raise
            if res is None:
                _m.require("segment-human")
                res = detect_body_fallback(rgb)
            _body_cache[image_id] = res
        return _body_cache[image_id]

    @app.post("/api/body/detect")
    async def body_detect(request: Request):
        from ..body.reshape import ROW_NAMES

        body = await request.json()
        b = await run_in_threadpool(_body, body["image_id"])
        return {"bbox": b["bbox"], "lattice": b["lattice"].tolist(), "rows": ROW_NAMES, "people": b["people"], "full_body": b["full_body"], "method": b.get("method", "mediapipe")}

    @app.post("/api/body/preview")
    async def body_preview(request: Request):
        from ..body.reshape import BodySettings, reshape_body

        body = await request.json()
        rgb, _ = store.get(body["image_id"])
        b = await run_in_threadpool(_body, body["image_id"])
        small, s = resize_max(rgb, 1024)
        mask = cv2.resize(b["mask"], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        src, dst = np.asarray(body["src"], np.float64) * s, np.asarray(body["dst"], np.float64) * s
        out, _ = await run_in_threadpool(lambda: reshape_body(small, mask, src, dst, BodySettings.from_dict(body.get("settings")), preview=True))
        return Response(encode_image(out, "jpg", quality=88), media_type="image/jpeg")

    @app.post("/api/body/apply")
    async def body_apply(request: Request):
        from ..body.reshape import BodySettings, reshape_body

        body = await request.json()
        image_id = body["image_id"]
        rgb, alpha = store.get(image_id)
        b = await run_in_threadpool(_body, image_id)
        src, dst = np.asarray(body["src"], np.float64), np.asarray(body["dst"], np.float64)
        settings = BodySettings.from_dict(body.get("settings"))
        name = store.meta(image_id).get("name", "")

        def work(progress):
            out, rep = reshape_body(rgb, b["mask"], src, dst, settings, progress=progress)
            return {"image_id": store.put(out, alpha, {"name": name, "operation": "body proportions", "icc_profile": store.icc(image_id)}), "report": _jsonable(rep)}

        return {"job_id": jobs.submit("body", work)}

    # ------------------------------------------------------------------ body 3D model
    @app.post("/api/body3d")
    async def body3d(request: Request):
        from ..models import manager as _m
        from ..recon.body3d import Body3DSettings, build_body3d

        body = await request.json()
        image_id = body["image_id"]
        settings = Body3DSettings.from_dict(body.get("settings"))
        _m.require(settings.model)
        rgb, _ = store.get(image_id)

        def work(progress):
            progress(0.02, "Finding the person")
            b = _body(image_id)
            out = build_body3d(rgb, b["mask"], settings, progress)
            return {
                "body": store.put_file(out["body"], ".lbd"),
                "texture": store.put_file(out["texture"], ".jpg"),
                "depth_preview": store.put_file(out["depth_preview"], ".jpg"),
                "header": out["header"],
                "method": b.get("method", "mediapipe"),
            }

        return {"job_id": jobs.submit("body3d", work)}

    # ------------------------------------------------------------------ 3D
    @app.post("/api/recon")
    async def recon(request: Request):
        from ..recon.scene import ReconSettings, reconstruct

        body = await request.json()
        rgb, _ = store.get(body["image_id"])
        settings = ReconSettings.from_dict(body.get("settings"))
        if not manager.is_installed(settings.model):
            manager.require(settings.model)  # raises a clear "install the model" error

        def work(progress):
            out = reconstruct(rgb, settings, progress)
            return {
                "scene": store.put_file(out["scene"], ".lsc"),
                "main_texture": store.put_file(out["main_texture"], ".jpg"),
                "bg_texture": store.put_file(out["bg_texture"], ".jpg"),
                "back_texture": store.put_file(out["back_texture"], ".jpg"),
                "depth_png": store.put_file(out["depth_png"], ".png"),
                "depth_preview": store.put_file(out["depth_preview"], ".jpg"),
                "header": out["header"],
                "report": _jsonable(out["report"]),
            }

        return {"job_id": jobs.submit("recon", work)}

    # ------------------------------------------------------------------ CPU rendering (browsers without WebGL)
    @app.post("/api/recon/render")
    async def recon_render(request: Request):
        from ..recon import render

        body = await _json_body(request)
        data, info = await run_in_threadpool(render.render_scene_request, body, store.file_path)
        return _frame_response(data, info)

    @app.post("/api/recon/environment")
    async def recon_environment(request: Request):
        # far environment panorama (equirectangular JPEG) shared by the WebGL and CPU viewers
        from ..recon import render

        body = await _json_body(request)
        data, info = await run_in_threadpool(render.environment_request, body, store.file_path)
        return _frame_response(data, info)

    @app.post("/api/body3d/render")
    async def body3d_render(request: Request):
        from ..recon import render

        body = await _json_body(request)
        data, info = await run_in_threadpool(render.render_body_request, body, store.file_path)
        return _frame_response(data, info)

    @app.get("/api/files/{name}")
    def files(name: str, download: int = 0):
        p = store.file_path(name)
        headers = {"Content-Disposition": f'attachment; filename="{name}"'} if download else {}
        return FileResponse(p, headers=headers)

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = jobs.get(job_id)
        if not j:
            raise NotFoundError("Unknown job")
        return j

    if static_dir and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    return app


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass  # state is exposed through manager.status()


def _warm_renderer() -> None:
    from ..recon import render

    render.warmup_in_background()


async def _json_body(request: Request):
    try:
        return await request.json()
    except ValueError:
        raise InvalidInputError("The request body is not valid JSON.") from None


def _frame_response(data: bytes, info: dict) -> Response:
    """A rendered JPEG frame with its timing and triangle count."""
    headers = {"X-Render-Ms": f"{info['ms']:.1f}", "X-Triangles": str(int(info["triangles"])), "Cache-Control": "no-store"}
    return Response(data, media_type="image/jpeg", headers=headers)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
