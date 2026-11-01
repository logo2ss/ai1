// Which renderer draws the 3D views: the browser's GPU (WebGL 2, required by three.js r170) or the
// engine's CPU renderer (frames rendered by the local engine and shown as images). Browsers in
// virtual machines often have no WebGL at all; the CPU path keeps the 3D views working there.
import { useSyncExternalStore } from "react";

export type RendererPref = "auto" | "gpu" | "cpu";
export type RendererKind = "gpu" | "cpu";
/** Why the effective renderer was chosen. */
export type RendererReason = "gpu" | "chosen" | "no-webgl" | "gpu-failed";

export interface WebGLSupport {
  webgl2: boolean;
  webgl1: boolean;
  /** WebGL is implemented in software (SwiftShader, llvmpipe, ...): it works, but slowly. */
  software: boolean;
  renderer: string;
  error?: string;
}

const SOFTWARE_RE = /swiftshader|llvmpipe|softpipe|lavapipe|software|basic render/i;
export const isSoftwareRenderer = (name: string) => SOFTWARE_RE.test(name);

function probe(kind: "webgl2" | "webgl"): { ok: boolean; renderer: string; error?: string } {
  let gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  try {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 1;
    gl = canvas.getContext(kind) as WebGLRenderingContext | WebGL2RenderingContext | null;
    if (!gl) return { ok: false, renderer: "" };
    let renderer = "";
    try {
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      renderer = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || "");
    } catch {
      /* the renderer name is informative only */
    }
    return { ok: !gl.isContextLost(), renderer };
  } catch (e) {
    return { ok: false, renderer: "", error: e instanceof Error ? e.message : String(e) };
  } finally {
    // free the probe context right away (browsers limit the number of live contexts)
    try {
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    } catch {
      /* ignore */
    }
  }
}

let support: WebGLSupport | null = null;

/** WebGL capabilities of this browser (probed once, then cached). */
export function webglSupport(): WebGLSupport {
  if (support) return support;
  const w2 = probe("webgl2");
  const w1 = w2.ok ? w2 : probe("webgl");
  const renderer = w2.renderer || w1.renderer;
  support = { webgl2: w2.ok, webgl1: w1.ok, software: isSoftwareRenderer(renderer), renderer, error: w2.error };
  return support;
}

// ---- shared renderer preference (remembered per browser) and GPU failure state

const PREF_KEY = "lumen-renderer";

function loadPref(): RendererPref {
  try {
    const v = localStorage.getItem(PREF_KEY);
    if (v === "auto" || v === "gpu" || v === "cpu") return v;
  } catch {
    /* storage unavailable: use the default */
  }
  return "auto";
}

let store: { pref: RendererPref; failure: string | null } = { pref: loadPref(), failure: null };
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
};

/**
 * Effective renderer for a preference. auto: the GPU only when WebGL 2 works (and has not failed
 * in this session), otherwise the engine's CPU renderer.
 */
export function chooseRenderer(pref: RendererPref = store.pref): RendererKind {
  if (pref === "cpu") return "cpu";
  if (store.failure) return "cpu";
  if (pref === "gpu") return "gpu";
  return webglSupport().webgl2 ? "gpu" : "cpu";
}

/** Choosing a renderer also clears an earlier GPU failure, so "GPU"/"Auto" try the GPU again. */
export function setRendererPref(pref: RendererPref) {
  store = { pref, failure: null };
  try {
    localStorage.setItem(PREF_KEY, pref);
  } catch {
    /* not remembered: fine */
  }
  emit();
}

/** A GPU viewer could not create or keep its WebGL context: use the CPU renderer from now on. */
export function reportGpuFailure(message: string) {
  if (store.failure) return;
  console.warn("[Lumen] GPU 3D renderer failed, switching to the CPU renderer:", message);
  store = { ...store, failure: message || "unknown error" };
  emit();
}

export function useRenderer() {
  const s = useSyncExternalStore(subscribe, () => store);
  const kind = chooseRenderer(s.pref);
  const reason: RendererReason = kind === "gpu" ? "gpu" : s.pref === "cpu" ? "chosen" : s.failure ? "gpu-failed" : "no-webgl";
  return { pref: s.pref, kind, reason, failure: s.failure, support: webglSupport(), setPref: setRendererPref, reportFailure: reportGpuFailure };
}

/** One-sentence explanation shown when the CPU renderer is active. */
export function cpuRendererMessage(reason: RendererReason, failure: string | null): string {
  if (reason === "chosen") return "The CPU renderer is selected, so the engine renders the 3D view on the CPU.";
  if (reason === "gpu-failed") return `The GPU renderer failed (${failure}), so the engine renders the 3D view on the CPU.`;
  const s = webglSupport();
  if (s.webgl1 && !s.webgl2) return "Your browser has no WebGL 2 (only WebGL 1, common in virtual machines), so the engine renders the 3D view on the CPU.";
  return "Your browser has no WebGL (common in virtual machines), so the engine renders the 3D view on the CPU.";
}
