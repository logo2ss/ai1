"""Reconstruct -> render the standard camera set -> metrics + contact sheets; and tag comparison."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import OUT_ROOT, STANDARD_SCENES
from .cameras import KEY_VIEWS, Cam, standard_cameras
from .metrics import SceneGeometry, overlay, psnr, quad_stretch, summarize, view_metrics
from .sheets import fit, grid, hrow, label, load_rgb, save_jpg, stack, text_panel

ROOT = Path(__file__).resolve().parents[3]
TESTDATA = ROOT / "testdata"
VIEW_W = 480
FILES = {"scene": "scene.lsc", "main_texture": "main.jpg", "bg_texture": "bg.jpg", "back_texture": "back.jpg",
         "depth_preview": "depth.jpg"}

# metrics shown in tables: (key, label, format, better)  better: -1 lower is better, +1 higher is better
TABLE = [
    ("small_guessed", "sm.guess%", "pct", -1),
    ("small_stretched", "sm.stretch%", "pct", -1),
    ("small_streak", "sm.streak", "f3", -1),
    ("guessed", "guess%", "pct", -1),
    ("empty", "void%", "pct", -1),
    ("env", "env%", "pct", -1),
    ("backdrop", "backdrop%", "pct", -1),
    ("stretched", "stretch%", "pct", -1),
    ("streak", "streak", "f3", -1),
    ("tex_bleed", "bleed%", "pct", -1),
    ("tex_streak_behind", "pl.streak", "f3", -1),
    ("tex_detail_behind", "pl.detail", "f2", +1),
    ("tex_streak_margin", "mg.streak", "f3", -1),
    ("tex_detail_margin", "mg.detail", "f2", +1),
    ("ref_psnr", "refPSNR", "f1", +1),
    ("ref_leak", "refLeak%", "pct", -1),
    ("time_total", "time s", "f1", -1),
]


def _fmt(v, f: str) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if f == "pct":
        return f"{100 * v:.2f}"
    return {"f1": f"{v:.1f}", "f2": f"{v:.2f}", "f3": f"{v:.3f}"}[f]


# ----------------------------------------------------------------------------- inputs


def resolve_scene(name: str) -> tuple[str, Path]:
    p = Path(name)
    if p.is_file():
        return p.stem, p
    for cand in (TESTDATA / "scenes" / name, TESTDATA / name):
        for ext in ("", ".jpg", ".png", ".jpeg"):
            q = Path(str(cand) + ext)
            if q.is_file():
                return name.replace("/", "_"), q
    raise FileNotFoundError(f"scene not found: {name}")


def scene_list(spec: str) -> list[str]:
    out: list[str] = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        out.extend(STANDARD_SCENES if s in ("std", "standard") else [s])
    return out


def load_photo(path: Path, max_side: int) -> np.ndarray:
    from lumen_engine.imaging.io import decode_image, resize_max

    rgb = decode_image(path.read_bytes())[0]
    rgb, _ = resize_max(rgb, max_side)
    return np.ascontiguousarray(rgb, np.float32)


# ----------------------------------------------------------------------------- reconstruction (cached)


class _StageTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.cur: str | None = None
        self.start = self.t0
        self.stages: dict[str, float] = {}

    def __call__(self, p=0.0, label: str = ""):
        now = time.perf_counter()
        if label != self.cur:
            if self.cur is not None:
                self.stages[self.cur] = self.stages.get(self.cur, 0.0) + now - self.start
            self.cur, self.start = label, now

    def finish(self) -> dict:
        self("", "__end__")
        self.stages.pop("__end__", None)
        return {k: round(v, 2) for k, v in self.stages.items()}


def reconstruct_cached(sdir: Path, rgb: np.ndarray, settings: dict, force: bool) -> dict:
    from lumen_engine.recon.scene import ReconSettings, reconstruct

    ab = os.environ.get("LUMEN_RECON_AB", "")  # scene.py development A/B switches are part of the cache key
    import lumen_engine
    engine_root = Path(lumen_engine.__file__).parent
    source_hash = hashlib.sha256()
    for source_file in sorted(engine_root.rglob("*.py")):
        source_hash.update(str(source_file.relative_to(engine_root)).encode())
        source_hash.update(source_file.read_bytes())
    key = hashlib.sha1(rgb.tobytes()[:: 97] + json.dumps(settings, sort_keys=True).encode() + ab.encode() + source_hash.digest()).hexdigest()[:16]
    meta_p = sdir / "recon.json"
    if not force and meta_p.is_file() and all((sdir / f).is_file() for f in FILES.values()):
        meta = json.loads(meta_p.read_text())
        if meta.get("key") == key:
            meta["cached"] = True
            return meta
    sdir.mkdir(parents=True, exist_ok=True)
    import lumen_engine.recon.scene as S

    timer = _StageTimer()
    S._DEBUG = {}  # intermediate masks (development aid of scene.py) -> decomposition overlay
    t0 = time.perf_counter()
    try:
        out = reconstruct(rgb, ReconSettings.from_dict(settings), timer)
        dbg = S._DEBUG or {}
    finally:
        S._DEBUG = None
    total = time.perf_counter() - t0
    stages = timer.finish()
    try:
        _decomposition_overlay(dbg, sdir / "layers.jpg")
    except Exception:  # noqa: BLE001 - debug output only
        traceback.print_exc()
    for k, f in FILES.items():
        if out.get(k) is not None:
            (sdir / f).write_bytes(out[k])
    rep = out.get("report", {})
    meta = {"key": key, "settings": settings, "time_total": round(total, 2), "stages": rep.get("timings") or stages,
            "report": json.loads(json.dumps(rep, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))),
            "header": out.get("header"), "cached": False, "size": [int(rgb.shape[1]), int(rgb.shape[0])]}
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def _decomposition_overlay(dbg: dict, path: Path) -> None:
    """Which region went to which layer: people red, other foreground orange, occlusion band blue,
    near surfaces of depth edges violet, sky cyan, inpainting hole outlined white."""
    work = dbg.get("work")
    if work is None:
        return
    img = (np.clip(work, 0, 1) * 255).astype(np.float32)
    h, w = img.shape[:2]

    def m(k):
        a = dbg.get(k)
        return a.astype(bool) if isinstance(a, np.ndarray) and a.shape[:2] == (h, w) else np.zeros((h, w), bool)

    people, fg, band, nsurf, sky, hole = m("people"), m("fg"), m("band"), m("nsurf"), m("sky"), m("hole_c")
    for mk, col, a in ((sky, (0, 220, 255), 0.35), (nsurf & ~band, (160, 80, 255), 0.35), (band, (40, 90, 255), 0.5),
                       (fg & ~people, (255, 150, 0), 0.5), (people, (255, 30, 30), 0.5)):
        img[mk] = img[mk] * (1 - a) + np.array(col, np.float32) * a
    edge = hole & ~cv2.erode(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    img[edge] = 255
    save_jpg(path, np.clip(img, 0, 255).astype(np.uint8), 88)


# ----------------------------------------------------------------------------- rendering with G-buffer


def _accepts_env(R) -> bool:
    try:
        return "env" in inspect.signature(R.render_scene).parameters
    except (TypeError, ValueError):
        return False


def scene_env(R, sc, tm, fov: float, tb=None):
    """Far-environment panorama (D3's recon.environment) when this renderer supports one."""
    if not _accepts_env(R) or not hasattr(R, "scene_environment_texture"):
        return None
    if "tex_bg" in inspect.signature(R.scene_environment_texture).parameters:
        return R.scene_environment_texture(sc, tm, fov, tb)
    return R.scene_environment_texture(sc, tm, fov)


def render_view(R, sc, tm, tb, tk, cam: Cam, W: int, H: int, fov: float, zfar: float, margin_fade: float = 0.35, env=None):
    """Render one view; returns (rgb uint8, gbuffer dict or None). The G-buffer is captured from the
    renderer's frame allocation (falls back to image-only metrics if the renderer changes)."""
    cap: dict = {}
    orig = getattr(R, "_frame", None)
    if orig is not None:
        def _cap(*a, **k):
            f = orig(*a, **k)
            cap["f"] = f
            return f
        R._frame = _cap
    try:
        p = R.SceneParams(position=tuple(cam.position), yaw=math.radians(cam.yaw), pitch=math.radians(cam.pitch),
                          look_at=cam.look_at, fov=fov, zfar=zfar, width=W, height=H, quality="full",
                          show_background=True, mode="color", margin_fade=margin_fade)
        img, info = R.render_scene(sc, tm, tb, tk, p, env=env) if env is not None else R.render_scene(sc, tm, tb, tk, p)
    finally:
        if orig is not None:
            R._frame = orig
    f = cap.get("f")
    gb = None
    if f is not None and getattr(f, "glayer", None) is not None and f.glayer.size == W * H:
        gb = {"layer": f.glayer.copy(), "u": f.gu.copy(), "v": f.gv.copy()}
    return img, gb


def _camera_rot(R, cam: Cam) -> np.ndarray:
    P = np.asarray(cam.position, np.float64)
    if cam.look_at is not None and np.linalg.norm(np.asarray(cam.look_at, float) - P) > 1e-9:
        return np.asarray(R._rot_look_at(P, np.asarray(cam.look_at, np.float64)), np.float64)
    return np.asarray(R._rot_yaw_pitch(math.radians(cam.yaw), math.radians(cam.pitch)), np.float64)


def _screen_scale(aspect: float, fov: float, W: int, H: int) -> float:
    tanx = math.tan(math.radians(fov) / 2)
    A = W / H
    fit_tan_y = tanx / aspect if A > aspect else tanx / A
    return H / (2.0 * fit_tan_y)


def _project_np(XYZ, Rm, P, s, W, H, valid=None, near=0.02):
    X, Y, Z = XYZ
    dx, dy, dz = X - P[0], Y - P[1], Z - P[2]
    xc = Rm[0, 0] * dx + Rm[1, 0] * dy + Rm[2, 0] * dz
    yc = Rm[0, 1] * dx + Rm[1, 1] * dy + Rm[2, 1] * dz
    w = -(Rm[0, 2] * dx + Rm[1, 2] * dy + Rm[2, 2] * dz)
    ok = w > near
    if valid is not None:
        ok &= valid
    wsafe = np.where(ok, w, 1.0)
    return W / 2.0 + xc / wsafe * s, H / 2.0 - yc / wsafe * s, ok


def stretch_flags(R, sc, cam: Cam, W: int, H: int, fov: float, zfar: float) -> tuple:
    try:
        verts = R._scene_vertices(sc, fov, zfar, 1)
    except Exception:  # renderer internals changed
        return None, None
    aspect = float(sc.header["aspect"])
    s = _screen_scale(aspect, fov, W, H)
    P = np.asarray(cam.position, np.float64)
    Rm = _camera_rot(R, cam)
    ref = Cam("ref")
    R0 = _camera_rot(R, ref)
    out = []
    for name, disp, keep in (("main", sc.disp, sc.keep), ("bg", sc.bg_disp, sc.keep_bg)):
        if name not in verts:
            out.append(None)
            continue
        valid = disp > -0.5
        sx, sy, ok = _project_np(verts[name], Rm, P, s, W, H, valid)
        sx0, sy0, _ = _project_np(verts[name], R0, np.zeros(3), s, W, H, valid)
        out.append(quad_stretch(sx, sy, ok, sx0, sy0, keep))
    return tuple(out)


def _median_depth(sc, zfar: float) -> float:
    gh, gw = sc.disp.shape
    fr = sc.header.get("frame") or {"x0": 0, "y0": 0, "x1": 1, "y1": 1}
    r0, r1 = int(fr["y0"] * (gh - 1)), int(fr["y1"] * (gh - 1))
    c0, c1 = int(fr["x0"] * (gw - 1)), int(fr["x1"] * (gw - 1))
    d = sc.disp[r0 : r1 + 1, c0 : c1 + 1]
    d = d[d > 0.0]
    if d.size == 0:
        return 3.0
    den = np.maximum(d * (1 - 1 / zfar) + 1 / zfar, 1 / (3 * zfar))
    return float(np.median(1.0 / den))


# ----------------------------------------------------------------------------- one scene


def eval_scene(name: str, tag_dir: Path, settings: dict, max_side: int, force: bool, sheets: bool = True, log=print) -> dict:
    from lumen_engine.recon import render as R

    key, path = resolve_scene(name)
    sdir = tag_dir / key
    rgb = load_photo(path, max_side)
    meta = reconstruct_cached(sdir, rgb, settings, force)
    log(f"  {key}: reconstruct {'(cached)' if meta.get('cached') else ''} {meta['time_total']:.1f}s  stages={meta['stages']}")
    blob = (sdir / FILES["scene"]).read_bytes()
    sc = R.parse_scene(blob)
    tex = {}
    for k in ("main_texture", "bg_texture", "back_texture"):
        p = sdir / FILES[k]
        tex[k] = R.decode_texture(p.read_bytes()) if p.is_file() else None
    tm, tb, tk = tex["main_texture"], tex["bg_texture"], tex["back_texture"]
    h = sc.header
    fov, zfar = float(h.get("default_fov", 60.0)), float(h.get("default_zfar", 8.0))
    aspect = float(h["aspect"])
    geo = SceneGeometry(sc, tm.rgb, tb.rgb)
    cm = geo.container_metrics()
    t_env = time.perf_counter()
    env = scene_env(R, sc, tm, fov, tb)
    t_env = time.perf_counter() - t_env
    if env is not None:
        save_jpg(sdir / "env.jpg", env.rgb, 85)
    zp = _median_depth(sc, zfar)
    cams = standard_cameras(zp)
    (sdir / "views").mkdir(exist_ok=True)
    (sdir / "diag").mkdir(exist_ok=True)
    views: dict = {}
    t_render = time.perf_counter()
    for cam in cams:
        W = VIEW_W if cam.aspect is None else int(round(VIEW_W * 1.33))
        Hh = int(round(W / (cam.aspect or aspect)))
        img, gb = render_view(R, sc, tm, tb, tk, cam, W, Hh, fov, zfar, env=env)
        if gb is not None:
            sm, sb = stretch_flags(R, sc, cam, W, Hh, fov, zfar)
            m, masks = view_metrics(img, gb["layer"], gb["u"], gb["v"], geo, sm, sb, env_shown=env is not None)
        else:
            empty = np.all(np.abs(img.astype(int) - np.array(R.SCENE_CLEAR)) <= 2, axis=2)
            m, masks = {"empty": float(empty.mean())}, {"empty": empty}
        views[cam.name] = m
        save_jpg(sdir / "views" / f"{cam.name}.jpg", img, 90)
        save_jpg(sdir / "diag" / f"{cam.name}.jpg", overlay(img, masks), 85)
    t_render = time.perf_counter() - t_render
    # reference-view fidelity (no margin fade): PSNR against the photo
    Wr = 640
    Hr = int(round(Wr / aspect))
    ref_img, ref_gb = render_view(R, sc, tm, tb, tk, Cam("ref"), Wr, Hr, fov, zfar, margin_fade=0.0, env=env)
    photo = cv2.resize((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), (Wr, Hr), interpolation=cv2.INTER_AREA)
    ref_psnr = psnr(ref_img, photo)
    ref_leak = float((ref_gb["layer"] != 0).mean()) if ref_gb is not None else float("nan")
    summary = summarize(views)
    res = {"scene": key, "source": str(path), **summary, **cm, "ref_psnr": ref_psnr, "ref_leak": ref_leak,
           "time_total": meta["time_total"], "stages": meta["stages"], "render_s": round(t_render, 2), "env_s": round(t_env, 2),
           "has_env": env is not None,
           "views": views, "report": meta.get("report", {}), "pivot_z": zp, "size": meta.get("size")}
    (sdir / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    if sheets:
        scene_sheet(sdir, res, photo, cams)
    log(f"  {key}: guess {100 * summary['guessed']:.1f}%  void {100 * summary['empty']:.2f}%  backdrop {100 * summary['backdrop']:.1f}%"
        f"  stretch {100 * summary['stretched']:.2f}%  streak {summary['streak']:.3f}  bleed {_fmt(cm['tex_bleed'], 'pct')}%"
        f"  refPSNR {ref_psnr:.1f}  render {t_render:.1f}s")
    return res


def _view_label(name: str, m: dict) -> list[str]:
    if "guessed" not in m:
        return [name, f"void {100 * m.get('empty', 0):.1f}%"]
    st = m.get("streak")
    st_s = "-" if st is None or not np.isfinite(st) else f"{st:+.2f}"
    env = f" E{100 * m['env']:.1f}" if m.get("env") else ""
    return [f"{name}  guess {100 * m['guessed']:.0f}%  streak {st_s}",
            f"V{100 * m['empty']:.1f}{env} B{100 * m['backdrop']:.1f} R{100 * m['revealed']:.1f} M{100 * m['margin']:.1f} S{100 * m['stretched']:.1f}"]


def scene_sheet(sdir: Path, res: dict, photo: np.ndarray, cams: list[Cam]) -> None:
    tw = 400
    head = [label(fit(photo, tw), ["photo"])]
    for f, lab in (("depth.jpg", "depth"), ("bg.jpg", "background plate (behind = full colour)"), ("main.jpg", "main texture + margin"),
                   ("back.jpg", "backdrop"), ("layers.jpg", "people red, objects orange, band blue, sky cyan, hole white")):
        im = load_rgb(sdir / f)
        if im is None:
            continue
        if f == "bg.jpg":
            try:
                from lumen_engine.recon import render as R

                sc = R.parse_scene((sdir / "scene.lsc").read_bytes())
                beh = (sc.bg_disp > -0.5) & (sc.bg_disp < sc.disp - 1e-4)
                beh = cv2.resize(beh.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                im = np.where(beh[..., None], im, (im * 0.3).astype(np.uint8))
            except Exception:
                pass
        head.append(label(fit(im, tw, int(tw / (photo.shape[1] / photo.shape[0]))), [lab]))
    rep = res.get("report", {})
    txt = [f"{res['scene']}  {res.get('size')}  recon {res['time_total']:.1f}s  render {res['render_s']:.1f}s",
           "  ".join(f"{k}:{v}" for k, v in list(res.get("stages", {}).items())[:4]),
           "  ".join(f"{k}:{v}" for k, v in list(res.get("stages", {}).items())[4:9]),
           f"novel-view mean: guess {_fmt(res['guessed'], 'pct')}%  void {_fmt(res['empty'], 'pct')}%  backdrop {_fmt(res['backdrop'], 'pct')}%",
           f"  revealed {_fmt(res['revealed'], 'pct')}%  margin {_fmt(res['margin'], 'pct')}%  stretched {_fmt(res['stretched'], 'pct')}%  streak {_fmt(res['streak'], 'f3')}",
           f"plate: bleed {_fmt(res['tex_bleed'], 'pct')}%  streak {_fmt(res['tex_streak_behind'], 'f3')}  detail {_fmt(res['tex_detail_behind'], 'f2')}  behind {_fmt(res['behind_fraction'], 'pct')}%",
           f"margin: streak {_fmt(res['tex_streak_margin'], 'f3')}  detail {_fmt(res['tex_detail_margin'], 'f2')}   ref PSNR {res['ref_psnr']:.2f}  leak {_fmt(res['ref_leak'], 'pct')}%",
           f"inpaint {rep.get('inpainting')}  outpaint {rep.get('outpainting')}  fg {rep.get('foreground')}  sky {rep.get('sky_fraction')}",
           "tint: magenta=revealed bg  cyan=backdrop  yellow=margin  green=stretched  red=void  blue=environment",
           f"environment: {'yes' if res.get('has_env') else 'no'} ({res.get('env_s', 0)} s)  env {_fmt(res.get('env'), 'pct')}%"]
    panel = text_panel(txt, tw * 3 + 16, head[0].shape[0], 0.45)
    tiles, dtiles = [], []
    for cam in cams:
        m = res["views"].get(cam.name, {})
        im = load_rgb(sdir / "views" / f"{cam.name}.jpg")
        dm = load_rgb(sdir / "diag" / f"{cam.name}.jpg")
        if im is None:
            continue
        hh = int(round(tw / (photo.shape[1] / photo.shape[0])))
        tiles.append(label(fit(im, tw, hh), _view_label(cam.name, m)))
        if dm is not None:
            dtiles.append(label(fit(dm, tw, hh), _view_label(cam.name, m)))
    top = [hrow(head[:4]), hrow(head[4:6]), hrow([panel])]
    save_jpg(sdir.parent / f"sheet_{res['scene']}.jpg", stack(top + [grid(tiles, 4)]), 84)
    save_jpg(sdir.parent / f"diag_{res['scene']}.jpg", stack([hrow([head[0], panel]), grid(dtiles, 4)]), 82)


# ----------------------------------------------------------------------------- whole run


def mean_row(results: list[dict]) -> dict:
    out = {}
    for k, *_ in TABLE:
        vals = [r[k] for r in results if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def table_md(results: list[dict], title: str) -> str:
    cols = [lab for _, lab, _, _ in TABLE]
    lines = [f"### {title}", "", "| scene | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for r in results + [dict(mean_row(results), scene="**mean**")]:
        lines.append(f"| {r['scene']} | " + " | ".join(_fmt(r.get(k), f) for k, _, f, _ in TABLE) + " |")
    return "\n".join(lines)


def run(scenes: list[str], tag: str, quality: str, max_side: int, extra: dict, force: bool, sheets: bool = True) -> list[dict]:
    tag_dir = Path(OUT_ROOT) / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    settings = {"quality": quality, **extra}
    results = []
    t0 = time.perf_counter()
    print(f"[recon_eval] tag={tag} quality={quality} max_side={max_side} settings={settings} -> {tag_dir}")
    for name in scenes:
        try:
            results.append(eval_scene(name, tag_dir, settings, max_side, force, sheets))
        except Exception as exc:  # keep going: one broken scene must not hide the others
            traceback.print_exc()
            print(f"  {name}: FAILED {exc}")
    if not results:
        return results
    # merge with earlier results of the same tag (runs on other scene subsets)
    summ_p = tag_dir / "summary.json"
    prev = json.loads(summ_p.read_text()) if summ_p.is_file() else {}
    allres = {r["scene"]: r for r in prev.get("scenes", [])}
    for r in results:
        allres[r["scene"]] = {k: v for k, v in r.items() if k not in ("views",)}
    meta = {"tag": tag, "quality": quality, "max_side": max_side, "settings": settings, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    ordered = sorted(allres.values(), key=lambda r: r["scene"])
    summ_p.write_text(json.dumps({"meta": meta, "scenes": ordered}, indent=1, default=float))
    md = table_md(ordered, f"{tag} ({quality}, max side {max_side}, {meta['updated']})")
    (tag_dir / "summary.md").write_text(md + "\n")
    if sheets:
        index_sheet(tag_dir, ordered)
    print(md)
    print(f"[recon_eval] done in {time.perf_counter() - t0:.0f}s; sheets: {tag_dir}/index.jpg, sheet_<scene>.jpg, diag_<scene>.jpg")
    return results


def index_sheet(tag_dir: Path, results: list[dict]) -> None:
    tw, rows = 220, []
    for r in results:
        sdir = tag_dir / r["scene"]
        mpath = sdir / "metrics.json"
        views = json.loads(mpath.read_text()).get("views", {}) if mpath.is_file() else {}
        tiles = []
        for v in KEY_VIEWS:
            im = load_rgb(sdir / "views" / f"{v}.jpg")
            if im is None:
                continue
            m = views.get(v, {})
            tiles.append(label(fit(im, tw, int(tw / 1.5)), [f"{v}" + (f" g{100 * m['guessed']:.0f}%" if "guessed" in m else "")], 0.38))
        txt = [r["scene"], f"guess {_fmt(r.get('guessed'), 'pct')}%  void {_fmt(r.get('empty'), 'pct')}%",
               f"env {_fmt(r.get('env'), 'pct')}%  backdrop {_fmt(r.get('backdrop'), 'pct')}%",
               f"stretch {_fmt(r.get('stretched'), 'pct')}%",
               f"streak {_fmt(r.get('streak'), 'f3')}  bleed {_fmt(r.get('tex_bleed'), 'pct')}%",
               f"plate detail {_fmt(r.get('tex_detail_behind'), 'f2')}  refPSNR {_fmt(r.get('ref_psnr'), 'f1')}",
               f"time {_fmt(r.get('time_total'), 'f1')}s"]
        tiles.append(text_panel(txt, 250, int(tw / 1.5), 0.38))
        rows.append(grid(tiles, len(tiles), 2))
    if rows:
        save_jpg(tag_dir / "index.jpg", stack(rows), 82)


# ----------------------------------------------------------------------------- comparison


def compare(tag_a: str, tag_b: str, scenes: list[str] | None = None) -> str:
    ra, rb = Path(OUT_ROOT) / tag_a, Path(OUT_ROOT) / tag_b
    sa = {r["scene"]: r for r in json.loads((ra / "summary.json").read_text())["scenes"]}
    sb = {r["scene"]: r for r in json.loads((rb / "summary.json").read_text())["scenes"]}
    keys = [k for k in sa if k in sb and (not scenes or k in {resolve_scene(s)[0] if not (TESTDATA / s).exists() else s.replace('/', '_') for s in scenes})]
    if not keys:
        raise SystemExit(f"no common scenes between {tag_a} and {tag_b}")
    out_dir = Path(OUT_ROOT) / f"compare_{tag_a}_vs_{tag_b}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [lab for _, lab, _, _ in TABLE]
    lines = [f"### {tag_a} -> {tag_b}", "", "| scene | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    A, B = [sa[k] for k in keys], [sb[k] for k in keys]
    for k, a, b in zip(keys + ["**mean**"], A + [mean_row(A)], B + [mean_row(B)]):
        cells = []
        for mk, _, f, better in TABLE:
            va, vb = a.get(mk), b.get(mk)
            s = f"{_fmt(va, f)}→{_fmt(vb, f)}"
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and math.isfinite(va) and math.isfinite(vb):
                d = (vb - va) * better
                tol = 0.002 if f == "pct" else (0.005 if f == "f3" else 0.02)
                if f == "f1":
                    tol = 0.2
                s += " ✓" if d > tol else (" ✗" if d < -tol else "")
            cells.append(s)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    (out_dir / "diff.md").write_text(md + "\n")
    for k in keys:
        _compare_sheet(ra / k, rb / k, tag_a, tag_b, out_dir / f"{k}.jpg", "views")
        _compare_sheet(ra / k, rb / k, tag_a, tag_b, out_dir / f"{k}_diag.jpg", "diag")
    print(md)
    print(f"[recon_eval] comparison sheets: {out_dir}/<scene>.jpg and <scene>_diag.jpg")
    return md


def _compare_sheet(da: Path, db: Path, ta: str, tb: str, out: Path, sub: str) -> None:
    ma = json.loads((da / "metrics.json").read_text()).get("views", {}) if (da / "metrics.json").is_file() else {}
    mb = json.loads((db / "metrics.json").read_text()).get("views", {}) if (db / "metrics.json").is_file() else {}
    names = [c.name for c in standard_cameras()]
    tiles = []
    tw = 360
    for n in names:
        a, b = load_rgb(da / sub / f"{n}.jpg"), load_rgb(db / sub / f"{n}.jpg")
        if a is None or b is None:
            continue
        hh = int(round(tw * a.shape[0] / a.shape[1]))
        tiles.append(label(fit(a, tw, hh), [f"{ta}: " + _view_label(n, ma.get(n, {}))[0], _view_label(n, ma.get(n, {}))[-1]], 0.38))
        tiles.append(label(fit(b, tw, hh), [f"{tb}: " + _view_label(n, mb.get(n, {}))[0], _view_label(n, mb.get(n, {}))[-1]], 0.38))
    head = []
    for f in ("bg.jpg", "main.jpg"):
        a, b = load_rgb(da / f), load_rgb(db / f)
        if a is not None and b is not None:
            head += [label(fit(a, tw), [f"{ta}: {f}"], 0.38), label(fit(b, tw), [f"{tb}: {f}"], 0.38)]
    save_jpg(out, stack([grid(head, 4), grid(tiles, 4)]), 82)
