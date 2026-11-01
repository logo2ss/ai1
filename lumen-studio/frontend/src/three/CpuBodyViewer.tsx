import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { postBlob } from "../api";
import { BodyData, BodyViewerHandle, BodyViewOptions, buildBodyGeometry } from "./body";
import { exportBodyGLB, exportBodyPLY, loadTexture } from "./exporters";
import { CpuFrameOverlay, describeRenderError, FrameStats, FrameStream } from "./cpuFrames";

export interface BodyFiles {
  body: string;
  texture: string;
}

interface Props {
  data: BodyData;
  texture: string;
  /** Engine file names of the body model (the engine renders from these). */
  files: BodyFiles;
  options: BodyViewOptions;
}

// camera of the GPU body viewer: PerspectiveCamera(fov 32) at (0, 0.05, 2.3) looking at the origin
const FOV = 32;
const HOME = new THREE.Vector3(0, 0.05, 2.3);
const MIN_DIST = 0.4;
const MAX_DIST = 6;
const EPS = 1e-6;
const AUTO_ROTATE = ((2 * Math.PI) / 60) * 1.2; // rad/s, OrbitControls autoRotateSpeed 1.2 at 60 fps
const r5 = (v: number) => Math.round(v * 1e5) / 1e5;

/**
 * The 3D body model drawn by the engine's CPU renderer (POST /api/body3d/render), for browsers
 * without WebGL. Same API as BodyViewer; the orbit camera (drag rotates, right/shift-drag pans,
 * wheel zooms, auto-rotate) is computed here with OrbitControls' conventions.
 */
export const CpuBodyViewer = forwardRef<BodyViewerHandle, Props>(function CpuBodyViewer({ data, texture, files, options }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const img = useRef<HTMLImageElement>(null);
  const stream = useRef<FrameStream | null>(null);
  const cam = useRef({ sph: new THREE.Spherical().setFromVector3(HOME), target: new THREE.Vector3() });
  const opts = useRef(options);
  opts.current = options;
  const filesRef = useRef(files);
  filesRef.current = files;
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<FrameStats | null>(null);

  useEffect(() => {
    const el = host.current!;
    setReady(false);
    setError(null);
    setStats(null);
    const frames = new FrameStream(() => img.current, {
      frame: (f) => {
        setStats(f);
        setReady(true);
      },
      error: (e) => setError(e ? describeRenderError(e) : null),
    });
    stream.current = frames;
    const c = cam.current;
    c.sph.setFromVector3(HOME);
    c.target.set(0, 0, 0);

    let cw = 0;
    let ch = 0;
    const measure = () => {
      const r = el.getBoundingClientRect();
      cw = r.width;
      ch = r.height;
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();

    const position = () => new THREE.Vector3().setFromSpherical(c.sph).add(c.target);

    // ---- input: left-drag orbits, right- or shift-drag pans, wheel zooms (OrbitControls mapping)
    let drag: { x: number; y: number; pan: boolean } | null = null;
    const onDown = (e: PointerEvent) => {
      if ((e.target as HTMLElement | null)?.closest?.("[data-no-drag]")) return;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        /* synthetic pointer */
      }
      el.focus();
      drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey || e.ctrlKey || e.metaKey };
    };
    const onMove = (e: PointerEvent) => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
      const hgt = Math.max(1, el.clientHeight);
      if (drag.pan) {
        const pos = position();
        const fwd = c.target.clone().sub(pos).normalize();
        const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
        const up = new THREE.Vector3().crossVectors(right, fwd);
        const td = pos.distanceTo(c.target) * Math.tan((FOV / 2) * (Math.PI / 180));
        c.target.addScaledVector(right, (-2 * dx * td) / hgt).addScaledVector(up, (2 * dy * td) / hgt);
      } else {
        c.sph.theta -= (2 * Math.PI * dx) / hgt;
        c.sph.phi -= (2 * Math.PI * dy) / hgt;
      }
    };
    const onUp = () => (drag = null);
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      let d = e.deltaY;
      if (e.deltaMode === 1) d *= 16;
      else if (e.deltaMode === 2) d *= 100;
      if (e.ctrlKey) d *= 10;
      const s = Math.pow(0.95, Math.abs(d * 0.01));
      if (d < 0) c.sph.radius *= s;
      else if (d > 0) c.sph.radius /= s;
    };
    const noMenu = (e: Event) => e.preventDefault();
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("contextmenu", noMenu);

    let raf = 0;
    let last = performance.now();
    const tick = () => {
      raf = requestAnimationFrame(tick);
      try {
        frame();
      } catch (e) {
        console.error("[Lumen] CPU body viewer update failed:", e);
      }
    };
    const frame = () => {
      const now = performance.now();
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const o = opts.current;
      if (o.autoRotate && !drag) c.sph.theta -= AUTO_ROTATE * dt;
      c.sph.phi = Math.max(EPS, Math.min(Math.PI - EPS, c.sph.phi));
      c.sph.radius = Math.max(MIN_DIST, Math.min(MAX_DIST, c.sph.radius));
      const p = position();
      const camera = { position: [r5(p.x), r5(p.y), r5(p.z)], target: [r5(c.target.x), r5(c.target.y), r5(c.target.z)], fov: FOV };
      const f = filesRef.current;
      const key = JSON.stringify([camera, o.style, o.colors, o.lineStep, o.hideHidden, o.relief, f.body, f.texture]);
      frames.update(key, cw, ch, (quality, width, height) =>
        postBlob("/api/body3d/render", {
          files: { body: f.body, texture: f.texture },
          camera,
          width,
          height,
          quality,
          style: o.style,
          colors: o.colors,
          line_step: o.lineStep,
          hide_hidden: o.hideHidden,
          relief: o.relief,
        }),
      );
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("contextmenu", noMenu);
      frames.dispose();
      if (stream.current === frames) stream.current = null;
    };
  }, [data, files.body, files.texture]);

  useImperativeHandle(ref, () => ({
    reset: () => {
      cam.current.sph.setFromVector3(HOME);
      cam.current.target.set(0, 0, 0);
    },
    screenshot: () => stream.current?.snapshot("image/png") ?? "",
    exportPLY: () => {
      const { geo } = buildBodyGeometry(data, opts.current.colors, opts.current.relief);
      const blob = exportBodyPLY(data, geo);
      geo.dispose();
      return blob;
    },
    exportGLB: async () => {
      const { geo } = buildBodyGeometry(data, opts.current.colors, opts.current.relief);
      const tex = await loadTexture(texture);
      try {
        return await exportBodyGLB(geo, tex);
      } finally {
        geo.dispose();
        tex.dispose();
      }
    },
  }));

  return (
    <div ref={host} tabIndex={0} className="absolute inset-0 outline-none bg-[#07090d] touch-none cursor-grab active:cursor-grabbing" data-testid="body-viewer" data-renderer="cpu">
      <img ref={img} alt="3D body" draggable={false} className="absolute inset-0 w-full h-full pointer-events-none select-none" style={{ objectFit: "fill", visibility: ready ? "visible" : "hidden" }} data-testid="cpu-frame" />
      <CpuFrameOverlay
        ready={ready}
        error={error}
        stats={stats}
        onRetry={() => {
          setError(null);
          stream.current?.retryNow();
        }}
      />
    </div>
  );
});
