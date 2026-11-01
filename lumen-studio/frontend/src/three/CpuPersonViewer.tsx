import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { postBlob } from "../api";
import { CpuFrameOverlay, describeRenderError, FrameStats, FrameStream } from "./cpuFrames";

export interface PersonFiles { person: string; texture: string }
export interface PersonHandle { reset(): void; angle(degrees: number): void; screenshot(): string }

/** Uses only CPU frames and vector math; never creates a WebGL context. */
export const CpuPersonViewer = forwardRef<PersonHandle, { files: PersonFiles }>(function CpuPersonViewer({ files }, ref) {
  const host = useRef<HTMLDivElement>(null), img = useRef<HTMLImageElement>(null);
  const stream = useRef<FrameStream | null>(null);
  const cam = useRef({ yaw: 0, pitch: 0, radius: 2.3, target: new THREE.Vector3() });
  const [ready, setReady] = useState(false), [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<FrameStats | null>(null);
  const reset = () => { Object.assign(cam.current, { yaw: 0, pitch: 0, radius: 2.3 }); cam.current.target.set(0, 0, 0); };
  useImperativeHandle(ref, () => ({ reset, angle: d => { cam.current.yaw = d * Math.PI / 180; }, screenshot: () => stream.current?.snapshot("image/png") ?? "" }));
  useEffect(() => {
    const el = host.current!;
    reset(); setReady(false); setError(null);
    const frames = new FrameStream(() => img.current, { frame: s => { setStats(s); setReady(true); }, error: e => setError(e ? describeRenderError(e) : null) });
    stream.current = frames;
    const keys = new Set<string>();
    let drag: { x: number; y: number; pan: boolean } | null = null;
    const down = (e: PointerEvent) => { el.focus(); el.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      const c = cam.current, scale = c.radius / Math.max(1, el.clientHeight);
      if (drag.pan) { c.target.x -= dx*scale; c.target.y += dy*scale; }
      else { c.yaw -= dx*0.006; c.pitch += dy*0.004; }
    };
    const up = () => { drag = null; };
    const wheel = (e: WheelEvent) => { e.preventDefault(); cam.current.radius *= Math.exp(Math.max(-300, Math.min(300, e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientHeight : 1))) * 0.001); };
    const keydown = (e: KeyboardEvent) => { if (["w","a","s","d","r","ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].includes(e.key.toLowerCase()) || e.key.startsWith("Arrow")) { e.preventDefault(); keys.add(e.key.toLowerCase()); if (e.key.toLowerCase() === "r") reset(); } };
    const keyup = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    const blur = () => { keys.clear(); drag = null; };
    const menu = (e: Event) => e.preventDefault();
    el.addEventListener("pointerdown", down); el.addEventListener("pointermove", move); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", wheel, { passive: false }); el.addEventListener("keydown", keydown); el.addEventListener("keyup", keyup); el.addEventListener("blur", blur); el.addEventListener("contextmenu", menu);
    let raf = 0, last = performance.now();
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const now = performance.now(), dt = Math.min(0.05, (now-last)/1000); last = now;
      const c = cam.current;
      if (keys.has("w") || keys.has("arrowup")) c.radius -= dt;
      if (keys.has("s") || keys.has("arrowdown")) c.radius += dt;
      if (keys.has("a") || keys.has("arrowleft")) c.target.x -= dt*0.4;
      if (keys.has("d") || keys.has("arrowright")) c.target.x += dt*0.4;
      c.yaw = Math.max(-70*Math.PI/180, Math.min(70*Math.PI/180, c.yaw));
      c.pitch = Math.max(-0.5, Math.min(0.5, c.pitch)); c.radius = Math.max(0.7, Math.min(6, c.radius));
      const p = [c.target.x+Math.sin(c.yaw)*Math.cos(c.pitch)*c.radius, c.target.y+Math.sin(c.pitch)*c.radius, c.target.z+Math.cos(c.yaw)*Math.cos(c.pitch)*c.radius];
      const camera = { position: p.map(v => Math.round(v*1e5)/1e5), target: c.target.toArray(), fov: 32 };
      frames.update(JSON.stringify(camera), el.clientWidth, el.clientHeight, !!drag || keys.size > 0,
        (quality, width, height) => postBlob("/api/person3d/render", { files, camera, quality, width, height }));
    };
    tick();
    return () => { cancelAnimationFrame(raf); frames.dispose(); stream.current = null;
      el.removeEventListener("pointerdown", down); el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up);
      el.removeEventListener("wheel", wheel); el.removeEventListener("keydown", keydown); el.removeEventListener("keyup", keyup); el.removeEventListener("blur", blur); el.removeEventListener("contextmenu", menu);
    };
  }, [files.person, files.texture]);
  return <div ref={host} tabIndex={0} className="absolute inset-0 outline-none bg-[#0e1117] touch-none cursor-grab" data-testid="person-viewer" data-renderer="cpu">
    <img ref={img} alt="Person in 3D on an empty background" draggable={false} className="absolute inset-0 w-full h-full pointer-events-none" style={{ visibility: ready ? "visible" : "hidden" }} data-testid="cpu-frame" />
    <CpuFrameOverlay ready={ready} error={error} stats={stats} onRetry={() => stream.current?.retryNow()} />
  </div>;
});
