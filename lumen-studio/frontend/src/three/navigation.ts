// Camera navigation shared by the WebGL (SceneViewer) and CPU (CpuSceneViewer) scene viewers, so both
// move identically:
//   orbit        (default) drag = orbit around the scene centre like a 3D photo, wheel = dolly
//   walk / look  (free roam) drag = look around (yaw/pitch), WASD / arrows = walk + strafe, Q/E or C/Space = down/up,
//                wheel = forward/back, Shift = faster, right-drag / Shift-drag = pan, R = reset view
//   orbit        drag = orbit around the scene centre, wheel = dolly, keys move the orbit centre
// Keys work while the viewer has focus (click it) or while the pointer is over it (unless a text field
// or another control has focus). Movement is velocity-smoothed. In "safe range" mode the camera stays near
// the original viewpoint (single-photo reconstructions are only reliable there); free roam has no limits.
import * as THREE from "three";
import type { ViewerOptions } from "./SceneViewer";

export interface NavPose {
  position: THREE.Vector3;
  yaw: number;
  pitch: number;
  /** look at this point instead of yaw/pitch (camera motion presets) */
  lookAt: THREE.Vector3 | null;
  outside: boolean;
  distance: number;
  /** the user is interacting (keys held, dragging) or the camera is still gliding */
  moving: boolean;
}

const MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "c", " ", "arrowup", "arrowdown", "arrowleft", "arrowright", "shift", "r"]);
const PITCH_LIMIT_SAFE = 1.2;
const PITCH_LIMIT_FREE = 1.45;

function editable(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el || !el.tagName) return false;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
}

export class NavController {
  pos = new THREE.Vector3();
  yaw = 0;
  pitch = 0;
  private orbit = { yaw: 0, pitch: 0, dist: 1, pan: new THREE.Vector3() };
  private orbitInit = false;
  private keys = new Set<string>();
  private vel = new THREE.Vector3(); // smoothed key velocity (world units / s)
  private targetPos: THREE.Vector3 | null = null;
  private drag: { x: number; y: number; button: number } | null = null;
  private hover = false;
  private t0 = performance.now();
  private last = performance.now();
  private lastInput = 0;
  private off: (() => void)[] = [];

  constructor(
    private el: HTMLElement,
    private opts: () => ViewerOptions,
    private safeRadius: number,
    private medD: number,
  ) {
    const on = <K extends keyof HTMLElementEventMap>(t: HTMLElement | Window, ev: K | string, fn: (e: any) => void, o?: AddEventListenerOptions) => {
      t.addEventListener(ev, fn, o);
      this.off.push(() => t.removeEventListener(ev, fn, o));
    };
    on(el, "pointerdown", (e: PointerEvent) => this.onDown(e));
    on(el, "pointermove", (e: PointerEvent) => this.onMove(e));
    on(el, "pointerup", () => (this.drag = null));
    on(el, "pointercancel", () => (this.drag = null));
    on(el, "pointerenter", () => (this.hover = true));
    on(el, "pointerleave", () => (this.hover = false));
    on(el, "wheel", (e: WheelEvent) => this.onWheel(e), { passive: false });
    on(el, "contextmenu", (e: Event) => e.preventDefault());
    on(window, "keydown", (e: KeyboardEvent) => this.onKey(e, true));
    on(window, "keyup", (e: KeyboardEvent) => this.onKey(e, false));
    on(window, "blur", () => this.keys.clear());
    document.addEventListener("visibilitychange", this.onHidden);
    this.off.push(() => document.removeEventListener("visibilitychange", this.onHidden));
  }

  private onHidden = () => {
    if (document.hidden) this.keys.clear();
  };

  /** Depth of the scene centre (median disparity) at the current depth range. */
  pivotZ(): number {
    const zf = this.opts().zfar;
    return 1 / (this.medD * (1 - 1 / zf) + 1 / zf);
  }

  private active(e: KeyboardEvent): boolean {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const ae = document.activeElement;
    if (ae === this.el || this.el.contains(ae)) return true;
    // pointer over the view: keys drive the camera unless a text field / slider has the keyboard, and
    // Space still presses a focused button
    if (!this.hover || editable(e.target) || editable(ae)) return false;
    if (e.key === " " && ae instanceof HTMLElement && ae.closest("button,a,[role=switch]")) return false;
    return true;
  }

  private onKey(e: KeyboardEvent, down: boolean) {
    const k = e.key.toLowerCase();
    if (!down) {
      this.keys.delete(k);
      if (k === "shift") this.keys.delete("shift");
      return;
    }
    if (!MOVE_KEYS.has(k) || !this.active(e)) return;
    e.preventDefault(); // no page scrolling (arrows, space) while navigating
    if (k === "r") {
      if (!e.repeat) this.reset();
      return;
    }
    this.keys.add(k);
    this.targetPos = null;
    this.lastInput = performance.now();
  }

  private onDown(e: PointerEvent) {
    if ((e.target as HTMLElement | null)?.closest?.("[data-no-drag]")) return;
    try {
      this.el.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointer */
    }
    this.el.focus({ preventScroll: true });
    this.drag = { x: e.clientX, y: e.clientY, button: e.button };
    this.targetPos = null;
  }

  private basis() {
    const eul = new THREE.Euler(this.pitch, this.yaw, 0, "YXZ");
    return { fwd: new THREE.Vector3(0, 0, -1).applyEuler(eul), right: new THREE.Vector3(1, 0, 0).applyEuler(eul), up: new THREE.Vector3(0, 1, 0).applyEuler(eul) };
  }

  private onMove(e: PointerEvent) {
    if (!this.drag) return;
    const dx = e.clientX - this.drag.x;
    const dy = e.clientY - this.drag.y;
    this.drag.x = e.clientX;
    this.drag.y = e.clientY;
    this.lastInput = performance.now();
    const o = this.opts();
    const pz = this.pivotZ();
    if (this.drag.button === 2 || e.shiftKey) {
      // pan in the camera plane
      const k = pz * 0.0016;
      const { right, up } = this.basis();
      if (o.controls === "orbit") this.orbit.pan.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
      else this.pos.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
    } else if (o.controls === "orbit") {
      // ~0.09 deg per px: a small drag is a small turn around the subject
      this.orbit.yaw -= dx * 0.0016;
      this.orbit.pitch -= dy * 0.0016;
    } else {
      const lim = o.constrain ? PITCH_LIMIT_SAFE : PITCH_LIMIT_FREE;
      this.yaw -= dx * 0.0022; // ~0.13 deg per px
      this.pitch = THREE.MathUtils.clamp(this.pitch - dy * 0.0022, -lim, lim);
    }
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault();
    this.lastInput = performance.now();
    const d = THREE.MathUtils.clamp(e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY, -400, 400);
    if (this.opts().controls === "orbit") {
      this.orbit.dist = Math.max(0.15, this.orbit.dist * Math.exp(d * 0.001));
    } else {
      this.pos.addScaledVector(this.basis().fwd, -d * 0.0012 * this.pivotZ());
    }
    this.targetPos = null;
  }

  /** Glide back to the original viewpoint. */
  reset() {
    this.targetPos = new THREE.Vector3(0, 0, 0);
    this.vel.set(0, 0, 0);
  }

  /** Advance one animation frame; returns the camera pose. */
  update(): NavPose {
    const now = performance.now();
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const o = this.opts();
    const pz = this.pivotZ();
    const k = this.keys;
    const { fwd, right } = this.basis();
    // keyboard: target velocity, smoothed (accelerates/decelerates over ~0.1 s)
    const speed = pz * 0.6 * (k.has("shift") ? 3 : 1);
    const want = new THREE.Vector3();
    if (k.has("w") || k.has("arrowup")) want.add(fwd);
    if (k.has("s") || k.has("arrowdown")) want.sub(fwd);
    if (k.has("d") || k.has("arrowright")) want.add(right);
    if (k.has("a") || k.has("arrowleft")) want.sub(right);
    if (k.has("e") || k.has(" ")) want.y += 1;
    if (k.has("q") || k.has("c")) want.y -= 1;
    if (want.lengthSq() > 0) want.normalize().multiplyScalar(speed);
    this.vel.lerp(want, 1 - Math.exp(-dt * 12));
    if (this.vel.lengthSq() < 1e-10) this.vel.set(0, 0, 0);
    const step = this.vel.clone().multiplyScalar(dt);
    if (o.controls === "walk") this.pos.add(step);
    if (k.size) this.targetPos = null;
    if (o.controls === "orbit") {
      if (!this.orbitInit) {
        // start orbiting from the current pose
        const eul = new THREE.Euler(this.pitch, this.yaw, 0, "YXZ");
        this.orbit = { yaw: this.yaw, pitch: this.pitch, dist: 1, pan: this.pos.clone().add(new THREE.Vector3(0, 0, -pz).applyEuler(eul)).sub(new THREE.Vector3(0, 0, -pz)) };
        this.orbitInit = true;
      }
      if (o.constrain) {
        // gentle limits: the single-view reconstruction is trustworthy for moderate viewpoint changes
        this.orbit.yaw = THREE.MathUtils.clamp(this.orbit.yaw, -Math.PI * 50 / 180, Math.PI * 50 / 180);
        this.orbit.pitch = THREE.MathUtils.clamp(this.orbit.pitch, -0.2, 0.2);
        this.orbit.dist = THREE.MathUtils.clamp(this.orbit.dist, 0.6, 1.3);
      } else {
        this.orbit.pitch = THREE.MathUtils.clamp(this.orbit.pitch, -PITCH_LIMIT_FREE, PITCH_LIMIT_FREE);
      }
      this.orbit.pan.add(step); // keys move the orbit centre
      if (o.constrain && this.orbit.pan.length() > this.safeRadius * pz)
        this.orbit.pan.setLength(this.safeRadius * pz);
      if (this.targetPos) {
        const f = Math.pow(0.001, dt);
        this.orbit.yaw *= f;
        this.orbit.pitch *= f;
        this.orbit.dist = 1 + (this.orbit.dist - 1) * f;
        this.orbit.pan.multiplyScalar(f);
        if (Math.abs(this.orbit.yaw) + Math.abs(this.orbit.pitch) + this.orbit.pan.length() + Math.abs(this.orbit.dist - 1) < 1e-3) this.targetPos = null;
      }
      const pivot = new THREE.Vector3(0, 0, -pz).add(this.orbit.pan);
      const off = new THREE.Vector3(0, 0, pz * this.orbit.dist).applyEuler(new THREE.Euler(this.orbit.pitch, this.orbit.yaw, 0, "YXZ"));
      this.pos.copy(pivot).add(off);
      this.yaw = this.orbit.yaw;
      this.pitch = this.orbit.pitch;
    } else {
      this.orbitInit = false;
      if (o.constrain) this.pitch = THREE.MathUtils.clamp(this.pitch, -PITCH_LIMIT_SAFE, PITCH_LIMIT_SAFE);
    }
    // animated reset
    if (this.targetPos && o.controls !== "orbit") {
      const f = Math.pow(0.001, dt);
      this.pos.lerp(this.targetPos, 1 - f);
      // shortest way back to yaw 0
      this.yaw = Math.atan2(Math.sin(this.yaw), Math.cos(this.yaw)) * f;
      this.pitch *= f;
      if (this.pos.distanceTo(this.targetPos) < 1e-3 && Math.abs(this.yaw) + Math.abs(this.pitch) < 1e-3) {
        this.pos.copy(this.targetPos);
        this.yaw = 0;
        this.pitch = 0;
        this.targetPos = null;
      }
    }
    // safe range: the reconstruction is reliable near the original viewpoint
    const limit = this.safeRadius * pz;
    const dist = this.pos.length();
    if (o.constrain && o.controls !== "orbit" && dist > limit) this.pos.multiplyScalar(limit / dist);
    // motion presets (applied on top of the user's pose)
    const t = (now - this.t0) / 1000;
    const m = new THREE.Vector3();
    let lookAt: THREE.Vector3 | null = null;
    if (o.motion === "sway") {
      m.set(Math.sin(t * 0.6) * 0.12 * pz, Math.sin(t * 0.9) * 0.03 * pz, 0);
      lookAt = new THREE.Vector3(this.pos.x, this.pos.y, -pz);
    } else if (o.motion === "dolly") {
      m.set(0, 0, -(0.5 - 0.5 * Math.cos(t * 0.5)) * 0.25 * pz);
    } else if (o.motion === "circle") {
      m.set(Math.cos(t * 0.7) * 0.1 * pz, Math.sin(t * 0.7) * 0.07 * pz, 0);
      lookAt = new THREE.Vector3(this.pos.x, this.pos.y, -pz);
    }
    const d = Math.min(dist, o.constrain ? limit : Infinity);
    const moving = k.size > 0 || this.drag !== null || this.vel.lengthSq() > 1e-8 || this.targetPos !== null || now - this.lastInput < 120;
    return {
      position: this.pos.clone().add(m),
      yaw: this.yaw,
      pitch: this.pitch,
      lookAt,
      outside: o.constrain && (o.controls === "orbit" ? Math.abs(this.orbit.yaw) > Math.PI * 49 / 180 : d > limit * 0.98),
      distance: d / Math.max(limit, 1e-6),
      moving,
    };
  }

  dispose() {
    for (const f of this.off) f();
    this.off = [];
    this.keys.clear();
  }
}

/** Median disparity of the main layer (the orbit pivot / speed reference), identical in both viewers. */
export function medianDisparity(disp: Float32Array): number {
  const sorted = Array.from(disp.filter((d, i) => d >= 0 && i % 7 === 0)).sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length * 0.5)] : 0.3;
}
