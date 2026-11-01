// Frame scheduling for the CPU (engine-rendered) 3D viewers. The viewer computes its camera on every
// animation frame and calls FrameStream.update() with a key describing everything that affects the
// picture; the stream decides when to ask the engine for a new frame:
//   * while the view changes (dragging, keys, camera motions): 'preview' frames (size adapted to the
//     measured round trip, ~0.3-0.75x), up to two requests in flight, each using the latest pose,
//   * answers are shown in request order (a late answer to an older request is dropped),
//   * after ~250 ms without changes: one 'full' frame at the container size x min(devicePixelRatio, 1.5).
import { useEffect, useState } from "react";
import { Loader2, RotateCcw } from "lucide-react";
import type { ApiError } from "../api";

export type FrameQuality = "preview" | "full";

export interface FrameStats {
  ms: number | null; // engine render time (X-Render-Ms)
  roundTrip: number; // request + decode in the browser
  triangles: number | null; // X-Triangles
  quality: FrameQuality;
  width: number;
  height: number;
}

export type FrameRequest = (quality: FrameQuality, width: number, height: number) => Promise<{ blob: Blob; headers: Headers }>;

export const IDLE_MS = 250;
export const PREVIEW_SCALE = 0.6;
// a request that never answers must not freeze the view (one request in flight at a time)
const FRAME_TIMEOUT_MS = 90000;
const MIN_SIDE = 64;
const MAX_SIDE = 2048;

/** Output size for a container (CSS px) and scale, kept within the engine's limits at the same aspect. */
export function frameSize(cw: number, ch: number, scale: number): [number, number] {
  let w = Math.max(1, cw * scale);
  let h = Math.max(1, ch * scale);
  const up = Math.max(1, MIN_SIDE / w, MIN_SIDE / h);
  w *= up;
  h *= up;
  const down = Math.min(1, MAX_SIDE / w, MAX_SIDE / h);
  w *= down;
  h *= down;
  const clamp = (v: number) => Math.max(MIN_SIDE, Math.min(MAX_SIDE, Math.round(v)));
  return [clamp(w), clamp(h)];
}

export function describeRenderError(e: unknown): string {
  const err = e as ApiError;
  if (err && (err.code === "404" || err.code === "405") && /not found|not allowed/i.test(err.message || ""))
    return "This version of the engine has no CPU renderer. Update Lumen Studio, or use a browser with WebGL.";
  return (err && err.message) || String(e);
}

const num = (v: string | null) => {
  if (v == null || v === "") return null;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
};

export class FrameStream {
  hasFrame = false;
  private inflight = 0;
  private disposed = false;
  private seenKey = "";
  private changedAt = 0;
  private sentKey = "";
  private sentQuality: FrameQuality | null = null;
  private url: string | null = null;
  private retryAt = 0;
  private backoff = 1000;
  private seq = 0; // id of the latest request
  private shownSeq = 0; // id of the frame on screen: answers to older requests are dropped (ordered display)
  private dead = new Set<number>(); // timed-out requests
  private lastSentAt = 0;
  private previewScale = PREVIEW_SCALE; // adapted to the measured round trip while moving

  constructor(
    private img: () => HTMLImageElement | null,
    private on: { frame?: (s: FrameStats) => void; error?: (e: unknown) => void } = {},
  ) {}

  /**
   * Call on every animation frame. `key` must change whenever the picture would change. `moving` (optional)
   * says the user is still interacting (keys held, dragging): only low-resolution previews are requested
   * then, up to two in flight so the engine never waits for the browser; the full frame follows once idle.
   */
  update(key: string, cw: number, ch: number, movingOrRequest: boolean | FrameRequest, maybeRequest?: FrameRequest): void {
    const request = (typeof movingOrRequest === "function" ? movingOrRequest : maybeRequest) as FrameRequest;
    const moving = typeof movingOrRequest === "boolean" ? movingOrRequest : false;
    const now = performance.now();
    const k = `${key}|${Math.round(cw)}x${Math.round(ch)}`;
    if (k !== this.seenKey) {
      this.seenKey = k;
      this.changedAt = now;
    }
    // hidden (display: none, background tab) or waiting after an error: nothing to do
    if (this.disposed || cw < 8 || ch < 8 || now < this.retryAt || document.hidden) return;
    const idle = !moving && now - this.changedAt >= IDLE_MS;
    let quality: FrameQuality | null = null;
    if (!idle) {
      // previews: a second request may start while one is being answered (the engine renders them back
      // to back), but not more often than every ~30 ms
      if (k !== this.sentKey && (this.inflight === 0 || (this.inflight === 1 && this.sentQuality === "preview" && now - this.lastSentAt > 30))) quality = "preview";
    } else if (this.inflight === 0 && (k !== this.sentKey || this.sentQuality !== "full")) quality = "full";
    if (!quality) return;
    const scale = quality === "preview" ? this.previewScale : Math.min(window.devicePixelRatio || 1, 1.5);
    const [w, h] = frameSize(cw, ch, scale);
    this.inflight++;
    this.sentKey = k;
    this.sentQuality = quality;
    this.lastSentAt = now;
    const t0 = now;
    const id = ++this.seq;
    const fail = (e: unknown) => {
      console.warn("[Lumen] CPU render request failed:", e);
      this.sentKey = ""; // render this view again after a pause
      this.retryAt = performance.now() + this.backoff;
      this.backoff = Math.min(this.backoff * 2, 8000);
      this.on.error?.(e);
    };
    const timer = window.setTimeout(() => {
      if (this.disposed || this.dead.has(id)) return;
      this.dead.add(id);
      this.inflight = Math.max(0, this.inflight - 1);
      fail(new Error("The engine is taking unusually long to render the 3D view (it may still be preparing its CPU renderer). Retrying…"));
    }, FRAME_TIMEOUT_MS);
    const stale = () => this.disposed || this.dead.has(id) || id < this.shownSeq;
    request(quality, w, h)
      .then(async ({ blob, headers }) => {
        if (stale()) return;
        const url = URL.createObjectURL(blob);
        try {
          // decode before swapping so the view never flashes empty
          const probe = new Image();
          probe.src = url;
          await probe.decode();
        } catch {
          URL.revokeObjectURL(url);
          throw new Error("The engine returned a frame the browser cannot display.");
        }
        if (stale()) {
          URL.revokeObjectURL(url);
          return;
        }
        const el = this.img();
        if (el) el.src = url;
        if (this.url) URL.revokeObjectURL(this.url);
        this.url = url;
        this.shownSeq = id;
        this.hasFrame = true;
        this.backoff = 1000;
        const roundTrip = performance.now() - t0;
        if (quality === "preview") {
          // keep previews interactive: aim for ~120 ms per frame including transfer and decoding
          if (roundTrip > 180) this.previewScale = Math.max(0.3, this.previewScale * 0.85);
          else if (roundTrip < 90) this.previewScale = Math.min(0.75, this.previewScale * 1.08);
        }
        this.on.error?.(null);
        this.on.frame?.({ ms: num(headers.get("X-Render-Ms")), roundTrip, triangles: num(headers.get("X-Triangles")), quality, width: w, height: h });
      })
      .catch((e) => {
        if (!this.disposed && !this.dead.has(id)) fail(e);
      })
      .finally(() => {
        window.clearTimeout(timer);
        if (!this.dead.has(id)) this.inflight = Math.max(0, this.inflight - 1);
        this.dead.delete(id);
      });
  }

  /** Try again right away (after an error). */
  retryNow() {
    this.retryAt = 0;
    this.backoff = 1000;
    this.sentKey = "";
  }

  /** The frame on screen as a data URL ('' before the first frame). */
  snapshot(type = "image/jpeg", quality = 0.92): string {
    const el = this.img();
    if (!el || !this.hasFrame || !el.naturalWidth) return "";
    const c = document.createElement("canvas");
    c.width = el.naturalWidth;
    c.height = el.naturalHeight;
    c.getContext("2d")!.drawImage(el, 0, 0);
    return c.toDataURL(type, quality);
  }

  dispose() {
    this.disposed = true;
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
  }
}

/** Loading spinner, error card and the "CPU rendering · N ms" badge drawn over a CPU viewer. */
export function CpuFrameOverlay({ ready, error, stats, onRetry }: { ready: boolean; error: string | null; stats: FrameStats | null; onRetry: () => void }) {
  const ms = stats ? Math.round(stats.ms ?? stats.roundTrip) : null;
  const waiting = !ready && !error;
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setSlow(false);
    if (!waiting) return;
    const t = window.setTimeout(() => setSlow(true), 6000);
    return () => window.clearTimeout(t);
  }, [waiting]);
  return (
    <>
      {waiting && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 text-[12px] text-muted pointer-events-none text-center px-6">
          <span className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" /> Rendering the 3D view on the CPU…
          </span>
          {slow && <span className="text-[11px] text-faint max-w-sm">The first frame takes longer while the engine prepares its CPU renderer; later frames are much faster.</span>}
        </div>
      )}
      {error && !ready && (
        <div className="absolute inset-0 flex items-center justify-center p-4" data-no-drag>
          <div className="max-w-md rounded-lg border border-[#5a2a2e] bg-[#1f1012] text-[#ffb3b3] p-4 text-[12px] leading-relaxed" role="alert" data-testid="cpu-render-error">
            <div className="font-semibold">The engine could not render the 3D view</div>
            <div className="mt-1 opacity-90 break-words">{error}</div>
            <button type="button" onClick={onRetry} className="mt-3 inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md bg-[#2a1517] hover:bg-[#3a1b1e] border border-[#5a2a2e] text-xs font-medium">
              <RotateCcw className="w-3.5 h-3.5" /> Retry
            </button>
          </div>
        </div>
      )}
      <div
        className="absolute right-3 bottom-3 text-[10.5px] font-mono tabular-nums rounded-md px-2 py-1 bg-black/60 backdrop-blur"
        title={stats ? `${stats.quality} frame ${stats.width}×${stats.height}${stats.triangles != null ? ` · ${stats.triangles.toLocaleString()} triangles` : ""} · ${Math.round(stats.roundTrip)} ms round trip` : undefined}
        data-testid="cpu-badge"
      >
        {error && ready ? <span className="text-[#ffb3b3]">CPU render failed · retrying</span> : <span className="text-white/80">CPU rendering{ms != null ? ` · ${ms} ms` : ""}</span>}
      </div>
    </>
  );
}
