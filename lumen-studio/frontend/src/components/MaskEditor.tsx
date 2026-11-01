import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import {
  Brush,
  Check,
  ChevronDown,
  Eraser,
  FlipHorizontal2,
  Hand,
  Magnet,
  Maximize,
  Minus,
  MousePointerClick,
  PenLine,
  Plus,
  Redo2,
  Spline,
  Trash2,
  Undo2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { api, imageUrl, ApiError } from "../api";
import { cx } from "./ui";

/** draw: freehand - the stroke plus everything it encloses; points: click a polygon; brush: paint exactly; smart: click an object. */
export type Tool = "draw" | "points" | "brush" | "smart" | "pan";

export interface MaskEditorHandle {
  /** The selection as a PNG (white, alpha = selected). Finishes an open points shape and waits for edge snapping first. */
  getMaskBlob: () => Promise<Blob | null>;
  hasMask: () => boolean;
  clear: () => void;
}

interface Props {
  imageId: string;
  width: number;
  height: number;
  label: string;
  color?: string;
  onChange?: (hasMask: boolean) => void;
  onError?: (e: ApiError) => void;
}

type Pt = [number, number];
type Rect = { x: number; y: number; w: number; h: number };
type HistEntry = { r: Rect; data: Uint8Array };

const MAX_MASK_SIDE = 1600;
const PREVIEW_SIDE = 520;
const MAX_HISTORY = 40;

// ---------------------------------------------------------------- pure helpers
function clampRect(r: Rect, W: number, H: number): Rect {
  const x0 = Math.max(0, Math.floor(r.x));
  const y0 = Math.max(0, Math.floor(r.y));
  const x1 = Math.min(W, Math.ceil(r.x + r.w));
  const y1 = Math.min(H, Math.ceil(r.y + r.h));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}

function bboxOf(pts: Pt[], pad: number): Rect {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
}

function unionRect(a: Rect | null, b: Rect): Rect {
  if (!a) return b;
  const x0 = Math.min(a.x, b.x), y0 = Math.min(a.y, b.y);
  return { x: x0, y: y0, w: Math.max(a.x + a.w, b.x + b.w) - x0, h: Math.max(a.y + a.h, b.y + b.h) - y0 };
}

const scaleRect = (r: Rect, k: number): Rect => ({ x: r.x * k, y: r.y * k, w: r.w * k, h: r.h * k });
const growRect = (r: Rect, d: number): Rect => ({ x: r.x - d, y: r.y - d, w: r.w + 2 * d, h: r.h + 2 * d });

function hexRgb(c: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(c);
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [143, 131, 255];
}

/**
 * Everything enclosed by the shape: pixels that the outside of the box cannot
 * reach with 4-connected steps (so even thin diagonal strokes seal an area).
 * A loop selects its inside, a scribble selects what it covers plus the gaps
 * it encloses.
 */
function fillEnclosed(bin: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h).fill(1);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const seed = (i: number) => {
    if (!bin[i] && out[i]) {
      out[i] = 0;
      stack[sp++] = i;
    }
  };
  for (let x = 0; x < w; x++) {
    seed(x);
    seed((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    seed(y * w);
    seed(y * w + w - 1);
  }
  while (sp > 0) {
    const i = stack[--sp];
    const x = i % w;
    if (x > 0) seed(i - 1);
    if (x < w - 1) seed(i + 1);
    if (i >= w) seed(i - w);
    if (i < w * (h - 1)) seed(i + w);
  }
  return out;
}

let scratch: HTMLCanvasElement | null = null;
/** Rasterise a vector shape (drawn in mask coordinates) into a binary array covering ``r`` (target pixels, scale ``k``). */
function raster(r: Rect, k: number, paint: (g: CanvasRenderingContext2D) => void): Uint8Array {
  scratch ??= document.createElement("canvas");
  scratch.width = r.w;
  scratch.height = r.h;
  const g = scratch.getContext("2d", { willReadFrequently: true })!;
  g.setTransform(k, 0, 0, k, -r.x, -r.y);
  paint(g);
  const d = g.getImageData(0, 0, r.w, r.h).data;
  const out = new Uint8Array(r.w * r.h);
  for (let i = 0, j = 3; i < out.length; i++, j += 4) out[i] = d[j] > 96 ? 1 : 0;
  return out;
}

/** Stroke with round caps; ``close`` adds a thin straight segment from the end back to the start. */
const paintStroke = (pts: Pt[], lw: number, close: boolean, minW: number) => (g: CanvasRenderingContext2D) => {
  g.fillStyle = g.strokeStyle = "#fff";
  g.lineCap = "round";
  g.lineJoin = "round";
  g.lineWidth = lw;
  g.beginPath();
  g.arc(pts[0][0], pts[0][1], lw / 2, 0, Math.PI * 2);
  g.fill();
  if (pts.length < 2) return;
  g.beginPath();
  g.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
  g.stroke();
  if (close && pts.length > 2) {
    const a = pts[pts.length - 1];
    g.lineWidth = Math.max(minW, lw * 0.2);
    g.beginPath();
    g.moveTo(a[0], a[1]);
    g.lineTo(pts[0][0], pts[0][1]);
    g.stroke();
  }
};

const paintPolygon = (pts: Pt[]) => (g: CanvasRenderingContext2D) => {
  g.fillStyle = "#fff";
  g.beginPath();
  g.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
  g.closePath();
  g.fill();
};

/** Outline of the selection as an SVG path (marching squares on a <= 900 px grid, collinear points merged). */
function contourPath(sel: Uint8Array, W: number, H: number): string {
  const f = Math.max(1, Math.ceil(Math.max(W, H) / 900));
  const gw = Math.ceil(W / f) + 2, gh = Math.ceil(H / f) + 2; // zero-padded so every contour closes
  const g = new Uint8Array(gw * gh);
  let any = false;
  for (let y = 1; y < gh - 1; y++) {
    const sy = Math.min(H - 1, Math.floor((y - 1) * f + f / 2));
    for (let x = 1; x < gw - 1; x++) {
      const v = sel[sy * W + Math.min(W - 1, Math.floor((x - 1) * f + f / 2))];
      g[y * gw + x] = v;
      if (v) any = true;
    }
  }
  if (!any) return "";
  // edge-midpoint ids: horizontal edge (x,y)-(x+1,y) -> 2*(y*gw+x), vertical (x,y)-(x,y+1) -> 2*(y*gw+x)+1
  const n = 2 * gw * gh;
  const nb0 = new Int32Array(n).fill(-1), nb1 = new Int32Array(n).fill(-1);
  const link = (a: number, b: number) => {
    if (nb0[a] < 0) nb0[a] = b;
    else nb1[a] = b;
    if (nb0[b] < 0) nb0[b] = a;
    else nb1[b] = a;
  };
  for (let y = 0; y < gh - 1; y++) {
    for (let x = 0; x < gw - 1; x++) {
      const i = y * gw + x;
      const c = (g[i] << 3) | (g[i + 1] << 2) | (g[i + gw + 1] << 1) | g[i + gw];
      if (c === 0 || c === 15) continue;
      const T = 2 * i, B = 2 * (i + gw), L = 2 * i + 1, R = 2 * (i + 1) + 1;
      switch (c) {
        case 1: case 14: link(L, B); break;
        case 2: case 13: link(B, R); break;
        case 3: case 12: link(L, R); break;
        case 4: case 11: link(T, R); break;
        case 6: case 9: link(T, B); break;
        case 7: case 8: link(L, T); break;
        case 5: link(T, R); link(L, B); break; // diagonal corners are not connected
        case 10: link(L, T); link(B, R); break;
      }
    }
  }
  const pos = (id: number): Pt => {
    const cell = id >> 1, x = cell % gw, y = (cell / gw) | 0;
    const gx = id & 1 ? x : x + 0.5, gy = id & 1 ? y + 0.5 : y;
    return [(gx - 0.5) * f, (gy - 0.5) * f];
  };
  const seen = new Uint8Array(n);
  const parts: string[] = [];
  for (let s = 0; s < n; s++) {
    if (nb0[s] < 0 || seen[s]) continue;
    const loop: Pt[] = [];
    let prev = -1, cur = s;
    while (cur >= 0 && !seen[cur]) {
      seen[cur] = 1;
      loop.push(pos(cur));
      const nx = nb0[cur] !== prev ? nb0[cur] : nb1[cur];
      prev = cur;
      cur = nx;
    }
    if (loop.length < 3) continue;
    let d = `M${loop[0][0].toFixed(1)} ${loop[0][1].toFixed(1)}`;
    for (let i = 1; i < loop.length; i++) {
      const a = loop[i - 1], b = loop[i], c = loop[(i + 1) % loop.length];
      const collinear = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) === 0;
      if (!collinear) d += `L${b[0].toFixed(1)} ${b[1].toFixed(1)}`;
    }
    parts.push(d + "Z");
  }
  return parts.join("");
}

function maskToBlob(sel: Uint8Array, W: number, H: number): Promise<Blob | null> {
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d")!;
  const img = g.createImageData(W, H);
  const d = img.data;
  for (let i = 0, j = 0; i < sel.length; i++, j += 4) {
    d[j] = d[j + 1] = d[j + 2] = 255;
    d[j + 3] = sel[i] ? 255 : 0;
  }
  g.putImageData(img, 0, 0);
  return new Promise((res) => c.toBlob((b) => res(b), "image/png"));
}

async function blobToMask(blob: Blob, W: number, H: number): Promise<Uint8Array> {
  const bmp = await createImageBitmap(blob);
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(bmp, 0, 0, W, H);
  const d = g.getImageData(0, 0, W, H).data;
  const out = new Uint8Array(W * H);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) out[i] = d[j] > 127 ? 1 : 0;
  return out;
}

/**
 * Selection editor built around two simple tools:
 *
 * - **Draw**: drag freehand. Whatever the line covers *and encloses* is
 *   selected - draw a loop around the part you want, or scribble over it.
 *   The gap between the end and the start is closed with a straight line.
 * - **Points**: click points around the area; the polygon is filled live.
 *
 * Each shape adds to the selection; Erase mode (or Alt / right mouse button)
 * removes with the same tools. After each shape the outline is snapped to the
 * real object edges by the engine - only within a narrow band around what was
 * drawn. Brush and one-click smart select are under "More". The mask is kept
 * at up to 1600 px; the engine rescales it to the full image resolution.
 */
export const MaskEditor = forwardRef<MaskEditorHandle, Props>(function MaskEditor({ imageId, width, height, label, color = "#8f83ff", onChange, onError }, ref) {
  const scale0 = Math.min(1, MAX_MASK_SIDE / Math.max(width, height));
  const W = Math.max(1, Math.round(width * scale0));
  const H = Math.max(1, Math.round(height * scale0));
  const toImg = 1 / scale0; // mask px -> image px
  const pk = Math.min(1, PREVIEW_SIDE / Math.max(W, H)); // mask px -> live preview px
  const PW = Math.max(1, Math.round(W * pk));
  const PH = Math.max(1, Math.round(H * pk));
  const defaultLine = Math.round(Math.min(64, Math.max(6, Math.max(W, H) * 0.018)));

  const wrap = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const tint = useRef<HTMLCanvasElement>(null);
  const preview = useRef<HTMLCanvasElement>(null);
  const liveLine = useRef<SVGPolylineElement>(null);
  const rubber = useRef<SVGPolylineElement>(null);
  const firstDot = useRef<SVGCircleElement>(null);
  const cursorRing = useRef<SVGCircleElement>(null);
  const sel = useRef<Uint8Array>(new Uint8Array(0));
  const tintData = useRef<ImageData | null>(null);
  const history = useRef<HistEntry[]>([]);
  const future = useRef<HistEntry[]>([]);
  const version = useRef(0);
  const stroke = useRef<{ pts: Pt[]; erase: boolean; kind: "draw" | "brush" } | null>(null);
  const ptsErase = useRef(false);
  const lastClick = useRef<{ t: number; x: number; y: number }>({ t: 0, x: 0, y: 0 });
  const dragDot = useRef<{ i: number; moved: boolean; x: number; y: number } | null>(null);
  const hover = useRef<Pt | null>(null);
  const panning = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const space = useRef(false);
  const active = useRef(false);
  const raf = useRef(0);
  const outlineTimer = useRef(0);
  const snapQ = useRef<{ roi: Rect | null; timer: number; running: Promise<void> | null; epoch: number }>({ roi: null, timer: 0, running: null, epoch: 0 });

  const [tool, setToolState] = useState<Tool>("points");
  const [erase, setErase] = useState(false);
  const [lineW, setLineW] = useState(defaultLine);
  const [snap, setSnap] = useState(true);
  const [view, setView] = useState({ s: 1, x: 0, y: 0 });
  const [pts, setPts] = useState<Pt[]>([]);
  const [outline, setOutline] = useState("");
  const [count, setCount] = useState(0);
  const [busy, setBusy] = useState(false);
  const [snapping, setSnapping] = useState(false);
  const [menu, setMenu] = useState(false);
  const [, force] = useState(0);

  // ---- setup (new image)
  useEffect(() => {
    sel.current = new Uint8Array(W * H);
    tintData.current = null;
    history.current = [];
    future.current = [];
    stroke.current = null;
    const q = snapQ.current;
    window.clearTimeout(q.timer);
    q.roi = null;
    q.epoch++;
    setPts([]);
    setLineW(defaultLine);
    paintTint();
    clearPreview();
    setOutline("");
    setCount(0);
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageId, W, H]);

  const has = count > 0 || pts.length >= 3;
  useEffect(() => {
    onChange?.(has);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [has]);

  // ---- rendering of the committed selection
  const paintTint = (r?: Rect) => {
    const c = tint.current;
    if (!c) return;
    if (c.width !== W || c.height !== H) {
      c.width = W;
      c.height = H;
      tintData.current = null;
    }
    const g = c.getContext("2d")!;
    if (!tintData.current) {
      tintData.current = g.createImageData(W, H);
      r = undefined;
    }
    const d = tintData.current.data;
    const m = sel.current;
    const [cr, cg, cb] = hexRgb(color);
    const rr = r ?? { x: 0, y: 0, w: W, h: H };
    for (let y = rr.y; y < rr.y + rr.h; y++) {
      let i = y * W + rr.x;
      let j = i * 4;
      for (let x = 0; x < rr.w; x++, i++, j += 4) {
        if (m[i]) {
          d[j] = cr;
          d[j + 1] = cg;
          d[j + 2] = cb;
          d[j + 3] = 105;
        } else d[j + 3] = 0;
      }
    }
    g.putImageData(tintData.current, 0, 0, rr.x, rr.y, rr.w, rr.h);
  };

  const changed = (r?: Rect) => {
    version.current++;
    paintTint(r);
    const m = sel.current;
    let n = 0;
    for (let i = 0; i < m.length; i++) n += m[i];
    setCount(n);
    window.clearTimeout(outlineTimer.current);
    outlineTimer.current = window.setTimeout(() => setOutline(contourPath(sel.current, W, H)), 16);
    force((v) => v + 1);
  };

  const fit = () => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r || !r.width) return;
    const s = Math.min((r.width - 24) / W, (r.height - 24) / H);
    setView({ s, x: (r.width - W * s) / 2, y: (r.height - H * s) / 2 });
  };
  useEffect(() => {
    const ro = new ResizeObserver(() => fit());
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W, H]);

  // ---- history (stores only the changed rectangle)
  const readRect = (r: Rect) => {
    const out = new Uint8Array(r.w * r.h);
    for (let y = 0; y < r.h; y++) out.set(sel.current.subarray((r.y + y) * W + r.x, (r.y + y) * W + r.x + r.w), y * r.w);
    return out;
  };
  const writeRect = (r: Rect, data: Uint8Array) => {
    for (let y = 0; y < r.h; y++) sel.current.set(data.subarray(y * r.w, (y + 1) * r.w), (r.y + y) * W + r.x);
  };
  const pushHistory = (r: Rect) => {
    history.current.push({ r, data: readRect(r) });
    if (history.current.length > MAX_HISTORY) history.current.shift();
    future.current = [];
  };
  const cancelSnap = () => {
    const q = snapQ.current;
    window.clearTimeout(q.timer);
    q.roi = null;
    q.epoch++;
  };
  const undo = () => {
    const e = history.current.pop();
    if (!e) return;
    cancelSnap();
    future.current.push({ r: e.r, data: readRect(e.r) });
    writeRect(e.r, e.data);
    changed(e.r);
  };
  const redo = () => {
    const e = future.current.pop();
    if (!e) return;
    cancelSnap();
    history.current.push({ r: e.r, data: readRect(e.r) });
    writeRect(e.r, e.data);
    changed(e.r);
  };
  const full = (): Rect => ({ x: 0, y: 0, w: W, h: H });
  const clear = () => {
    cancelSnap();
    setPts([]);
    stroke.current = null;
    clearPreview();
    if (!count) return;
    pushHistory(full());
    sel.current.fill(0);
    changed();
  };
  const invert = () => {
    cancelSnap();
    pushHistory(full());
    const m = sel.current;
    for (let i = 0; i < m.length; i++) m[i] = 1 - m[i];
    changed();
  };
  const morph = (grow: boolean) => {
    // grow/shrink by ~0.6% of the image: blur + threshold
    cancelSnap();
    pushHistory(full());
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const g = c.getContext("2d", { willReadFrequently: true })!;
    const img = g.createImageData(W, H);
    for (let i = 0, j = 3; i < sel.current.length; i++, j += 4) img.data[j] = sel.current[i] ? 255 : 0;
    g.putImageData(img, 0, 0);
    const t = document.createElement("canvas");
    t.width = W;
    t.height = H;
    const tg = t.getContext("2d", { willReadFrequently: true })!;
    tg.filter = `blur(${Math.max(1, Math.round(Math.max(W, H) * 0.006))}px)`;
    tg.drawImage(c, 0, 0);
    const d = tg.getImageData(0, 0, W, H).data;
    const thr = grow ? 20 : 235;
    for (let i = 0, j = 3; i < sel.current.length; i++, j += 4) sel.current[i] = d[j] > thr ? 1 : 0;
    changed();
  };

  /** Add (or erase) a binary shape covering rectangle ``r``. */
  const applyShape = (r: Rect, m: Uint8Array, er: boolean, snapIt: boolean) => {
    if (!r.w || !r.h) return;
    let any = false;
    for (let k = 0; k < m.length && !any; k++) any = m[k] === 1;
    if (!any) return;
    pushHistory(r);
    const s = sel.current;
    for (let y = 0; y < r.h; y++) {
      let i = (r.y + y) * W + r.x;
      let k = y * r.w;
      for (let x = 0; x < r.w; x++, i++, k++) if (m[k]) s[i] = er ? 0 : 1;
    }
    changed(r);
    if (snapIt && snap) scheduleSnap(r);
  };

  // ---- edge snapping (engine), limited to a band around what was drawn
  const band = () => Math.max(4, Math.round(lineW * 1.5));
  const scheduleSnap = (r: Rect) => {
    const q = snapQ.current;
    q.roi = unionRect(q.roi, r);
    window.clearTimeout(q.timer);
    q.timer = window.setTimeout(() => void runSnap(), 220);
  };
  const runSnap = async (): Promise<void> => {
    const q = snapQ.current;
    window.clearTimeout(q.timer);
    while (q.running) await q.running;
    if (!q.roi) return;
    const roi = clampRect(growRect(q.roi, band() + 4), W, H);
    q.roi = null;
    const v = version.current;
    const epoch = q.epoch;
    const job = (async () => {
      setSnapping(true);
      try {
        const blob = await maskToBlob(sel.current, W, H);
        if (!blob) return;
        const fd = new FormData();
        fd.append("payload", JSON.stringify({ image_id: imageId, size: [W, H], roi: [roi.x, roi.y, roi.w, roi.h], band: band() }));
        fd.append("mask", blob, "mask.png");
        const res = await api.form<Blob>("/api/region/refine-mask", fd);
        const m = await blobToMask(res, W, H);
        if (epoch !== q.epoch) return; // undone / cleared meanwhile
        if (version.current !== v) {
          scheduleSnap(roi); // changed meanwhile: snap this area again together with the newer shape
          return;
        }
        const cur = readRect(roi);
        const next = new Uint8Array(roi.w * roi.h);
        let diff = false;
        for (let y = 0; y < roi.h; y++) {
          const o = (roi.y + y) * W + roi.x;
          for (let x = 0; x < roi.w; x++) {
            const val = m[o + x];
            next[y * roi.w + x] = val;
            if (val !== cur[y * roi.w + x]) diff = true;
          }
        }
        if (diff) {
          pushHistory(roi);
          writeRect(roi, next);
          changed(roi);
        }
      } catch {
        /* snapping is a refinement: keep the drawn selection if the engine cannot do it */
      } finally {
        setSnapping(false);
      }
    })();
    q.running = job;
    await job;
    q.running = null;
  };
  const flushSnap = async () => {
    const q = snapQ.current;
    if (q.roi) await runSnap();
    while (q.running) await q.running;
  };

  // ---- shapes
  const clearPreview = () => {
    const c = preview.current;
    if (c) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);
    liveLine.current?.setAttribute("points", "");
    rubber.current?.setAttribute("points", "");
  };

  const commitStroke = () => {
    const st = stroke.current;
    stroke.current = null;
    clearPreview();
    if (!st) return;
    const r = clampRect(bboxOf(st.pts, lineW / 2 + 3), W, H);
    if (!r.w || !r.h) return;
    let m = raster(r, 1, paintStroke(st.pts, lineW, st.kind === "draw", 2));
    if (st.kind === "draw") m = fillEnclosed(m, r.w, r.h);
    applyShape(r, m, st.erase, st.kind === "draw");
  };

  const finishPoints = (list: Pt[] = pts) => {
    if (list.length >= 3) {
      const r = clampRect(bboxOf(list, 2), W, H);
      applyShape(r, raster(r, 1, paintPolygon(list)), ptsErase.current, true);
    }
    setPts([]);
    clearPreview();
  };

  const setTool = (t: Tool) => {
    if (tool === "points" && t !== "points") finishPoints();
    setToolState(t);
    setMenu(false);
  };

  const renderPreview = () => {
    const c = preview.current;
    if (!c) return;
    if (c.width !== PW || c.height !== PH) {
      c.width = PW;
      c.height = PH;
    }
    const g = c.getContext("2d")!;
    g.clearRect(0, 0, PW, PH);
    let m: Uint8Array | null = null;
    let r: Rect | null = null;
    let er = false;
    const st = stroke.current;
    if (st) {
      er = st.erase;
      r = clampRect(scaleRect(bboxOf(st.pts, lineW / 2 + 3), pk), PW, PH);
      if (r.w && r.h) {
        m = raster(r, pk, paintStroke(st.pts, lineW, st.kind === "draw", 2 / pk));
        if (st.kind === "draw") m = fillEnclosed(m, r.w, r.h);
      }
      liveLine.current?.setAttribute("points", st.pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "));
    } else if (tool === "points" && pts.length) {
      er = ptsErase.current;
      const h = dragDot.current ? null : hover.current;
      if (pts.length >= 3) {
        // the dots are always shown as a closed, filled area: that is exactly what will be selected
        r = clampRect(scaleRect(bboxOf(pts, 2), pk), PW, PH);
        if (r.w && r.h) m = raster(r, pk, paintPolygon(pts));
      }
      const last = pts[pts.length - 1];
      rubber.current?.setAttribute("points", h ? `${last[0]},${last[1]} ${h[0]},${h[1]}${pts.length >= 2 ? ` ${pts[0][0]},${pts[0][1]}` : ""}` : "");
      const near = h && pts.length >= 3 && Math.hypot(h[0] - pts[0][0], h[1] - pts[0][1]) * view.s < 12;
      firstDot.current?.setAttribute("r", String((near ? 8 : 5) / view.s));
    }
    if (m && r && r.w && r.h) {
      const img = g.createImageData(r.w, r.h);
      const d = img.data;
      const [cr, cg, cb] = er ? [255, 96, 96] : hexRgb(color);
      for (let i = 0, j = 0; i < m.length; i++, j += 4)
        if (m[i]) {
          d[j] = cr;
          d[j + 1] = cg;
          d[j + 2] = cb;
          d[j + 3] = er ? 130 : 110;
        }
      g.putImageData(img, r.x, r.y);
    }
  };
  const requestPreview = () => {
    if (!raf.current)
      raf.current = requestAnimationFrame(() => {
        raf.current = 0;
        renderPreview();
      });
  };
  useEffect(() => {
    requestPreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pts, view.s]);
  useEffect(() => {
    moveCursor(hover.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool]);

  const smart = async (p: Pt, er: boolean) => {
    setBusy(true);
    try {
      const blob = await api.post<Blob>("/api/region/smart-select", { image_id: imageId, points: [[p[0] * toImg, p[1] * toImg]] });
      applyShape(full(), await blobToMask(blob, W, H), er, false);
    } catch (e) {
      onError?.(e as ApiError);
    } finally {
      setBusy(false);
    }
  };

  useImperativeHandle(ref, () => ({
    getMaskBlob: async () => {
      if (pts.length) finishPoints();
      await flushSnap();
      let any = false;
      for (let i = 0; i < sel.current.length && !any; i++) any = sel.current[i] === 1;
      return any ? maskToBlob(sel.current, W, H) : null;
    },
    hasMask: () => has,
    clear,
  }));

  // ---- pointer handling (on the whole viewport, so a loop may start or run outside the image: it is clamped to the edge)
  const toMask = (e: { clientX: number; clientY: number }, clamp = true): Pt => {
    const r = stage.current!.getBoundingClientRect();
    let x = ((e.clientX - r.left) / r.width) * W;
    let y = ((e.clientY - r.top) / r.height) * H;
    if (clamp) {
      x = Math.min(W, Math.max(0, x));
      y = Math.min(H, Math.max(0, y));
    }
    return [x, y];
  };
  const moveCursor = (p: Pt | null) => {
    const c = cursorRing.current;
    if (!c) return;
    if (!p || !(tool === "draw" || tool === "brush") || space.current) c.setAttribute("visibility", "hidden");
    else {
      c.setAttribute("visibility", "visible");
      c.setAttribute("cx", String(p[0]));
      c.setAttribute("cy", String(p[1]));
    }
  };

  const onDown = (e: React.PointerEvent) => {
    if (menu) setMenu(false);
    wrap.current?.setPointerCapture?.(e.pointerId);
    if (e.button === 1 || space.current || tool === "pan") {
      panning.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
      return;
    }
    if (e.button !== 0 && e.button !== 2) return;
    const p = toMask(e);
    if (tool === "points" && e.button === 2) {
      if (pts.length) setPts(pts.slice(0, -1)); // right-click removes the last dot
      return;
    }
    const er = erase || e.altKey || e.button === 2;
    if (tool === "draw" || tool === "brush") {
      stroke.current = { pts: [p], erase: er, kind: tool };
      requestPreview();
    } else if (tool === "points") {
      const now = performance.now();
      const lc = lastClick.current;
      const dbl = now - lc.t < 380 && Math.hypot(e.clientX - lc.x, e.clientY - lc.y) < 8;
      lastClick.current = { t: now, x: e.clientX, y: e.clientY };
      if (pts.length >= 3 && dbl) {
        finishPoints();
        return;
      }
      // pressing on a dot drags it (a click on the first dot without moving finishes the area)
      let near = -1, best = 12;
      pts.forEach((q, i) => {
        const d = Math.hypot(q[0] - p[0], q[1] - p[1]) * view.s;
        if (d < best) (best = d), (near = i);
      });
      if (near >= 0) {
        dragDot.current = { i: near, moved: false, x: e.clientX, y: e.clientY };
        return;
      }
      if (dbl && pts.length) return;
      if (!pts.length) ptsErase.current = er;
      setPts([...pts, p]);
    } else if (tool === "smart") {
      void smart(toMask(e, false), er);
    }
  };
  const onMove = (e: React.PointerEvent) => {
    const p = toMask(e);
    hover.current = p;
    moveCursor(p);
    if (panning.current) {
      const pn = panning.current;
      setView((v) => ({ ...v, x: pn.vx + e.clientX - pn.x, y: pn.vy + e.clientY - pn.y }));
      return;
    }
    const dd = dragDot.current;
    if (dd) {
      if (!dd.moved && Math.hypot(e.clientX - dd.x, e.clientY - dd.y) > 3) dd.moved = true;
      if (dd.moved) setPts(pts.map((q, i) => (i === dd.i ? p : q)));
      return;
    }
    const st = stroke.current;
    if (st) {
      const events = (e.nativeEvent as PointerEvent).getCoalescedEvents?.() ?? [];
      for (const ev of events.length ? events : [e]) {
        const q = toMask(ev);
        const l = st.pts[st.pts.length - 1];
        if (Math.hypot(l[0] - q[0], l[1] - q[1]) * view.s > 1.5) st.pts.push(q);
      }
      requestPreview();
    } else if (tool === "points" && pts.length) requestPreview();
  };
  const onUp = () => {
    if (panning.current) {
      panning.current = null;
      return;
    }
    const dd = dragDot.current;
    if (dd) {
      dragDot.current = null;
      if (!dd.moved && dd.i === 0 && pts.length >= 3) finishPoints();
      requestPreview();
      return;
    }
    if (stroke.current) commitStroke();
  };
  const onWheel = (e: React.WheelEvent) => {
    const r = wrap.current!.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0015);
    setView((v) => {
      const s = Math.min(40, Math.max(0.05, v.s * k));
      const kk = s / v.s;
      return { s, x: mx - (mx - v.x) * kk, y: my - (my - v.y) * kk };
    });
  };
  const zoomBy = (k: number) => {
    const r = wrap.current!.getBoundingClientRect();
    const mx = r.width / 2;
    const my = r.height / 2;
    setView((v) => {
      const s = Math.min(40, Math.max(0.05, v.s * k));
      const kk = s / v.s;
      return { s, x: mx - (mx - v.x) * kk, y: my - (my - v.y) * kk };
    });
  };

  // ---- keyboard shortcuts (only for the editor under the pointer)
  useEffect(() => {
    const kd = (e: KeyboardEvent) => {
      if (!active.current || (e.target as HTMLElement)?.tagName === "INPUT") return;
      const k = e.key.toLowerCase();
      if (k === " ") {
        space.current = true;
        moveCursor(null);
        e.preventDefault();
      } else if ((e.ctrlKey || e.metaKey) && k === "z") {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      } else if ((e.ctrlKey || e.metaKey) && k === "y") {
        e.preventDefault();
        redo();
      } else if (e.ctrlKey || e.metaKey || e.altKey) {
        return;
      } else if (k === "enter" && pts.length) {
        e.preventDefault();
        finishPoints();
      } else if (k === "backspace" && pts.length) {
        e.preventDefault();
        setPts(pts.slice(0, -1));
      } else if (k === "escape") {
        setPts([]);
        stroke.current = null;
        clearPreview();
        setMenu(false);
      } else if (k === "d") setTool("draw");
      else if (k === "p") setTool("points");
      else if (k === "b") setTool("brush");
      else if (k === "s") setTool("smart");
      else if (k === "h") setTool("pan");
      else if (k === "e" || k === "x") setErase((v) => !v);
      else if (k === "i") invert();
      else if (k === "[") setLineW((s) => Math.max(2, Math.round(s / 1.2)));
      else if (k === "]") setLineW((s) => Math.min(400, Math.round(s * 1.2)));
      else if (k === "f") fit();
    };
    const ku = (e: KeyboardEvent) => {
      if (e.key === " ") space.current = false;
    };
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);
    return () => {
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
    };
  });

  const hint =
    tool === "draw"
      ? "Draw around or over the part you want — everything inside the line is selected."
      : tool === "points"
        ? pts.length
          ? "Keep clicking dots around the area · drag a dot to move it · right-click or Backspace removes the last dot · Enter starts another area"
          : "Click dots around the area you want — the inside is selected."
        : tool === "brush"
          ? "Paint exactly the pixels you want (no filling)."
          : tool === "smart"
            ? "Click an object to select it."
            : "Drag to move the view · wheel to zoom";
  const cursor = tool === "pan" || space.current ? "grab" : tool === "draw" || tool === "brush" ? "none" : "crosshair";
  const sw = 1.4 / view.s; // 1.4 screen px in mask units
  const er = erase;
  const more: { id: Tool; icon: any; label: string; key: string }[] = [
    { id: "brush", icon: Brush, label: "Brush — paint exactly", key: "B" },
    { id: "smart", icon: MousePointerClick, label: "Smart select — click an object", key: "S" },
    { id: "pan", icon: Hand, label: "Move view", key: "H / Space" },
  ];
  const inMore = more.some((m) => m.id === tool);

  return (
    <div className="flex flex-col h-full min-w-0 bg-panel rounded-lg border border-line overflow-hidden" data-testid={`editor-${label}`}>
      <div className="flex items-center gap-1 px-2 h-10 border-b border-line bg-panel-2/50">
        <span className="w-6 h-6 rounded text-[11px] font-bold flex items-center justify-center mr-1 text-black shrink-0" style={{ background: color }} title={label}>
          {label.slice(-1)}
        </span>
        <ToolBtn on={tool === "points"} icon={Spline} label="Dots" tip="Dots (P) — click dots around the area; the inside is selected" onClick={() => setTool("points")} testid="tool-points" />
        <ToolBtn on={tool === "draw"} icon={PenLine} label="Draw" tip="Draw (D) — draw around or over the area; the inside is selected" onClick={() => setTool("draw")} testid="tool-draw" />
        <div className="relative">
          <button
            title="More tools"
            onClick={() => setMenu(!menu)}
            className={cx("h-7 px-1.5 rounded flex items-center gap-0.5 text-[12px] transition", inMore ? "bg-accent/20 text-accent ring-1 ring-accent/40" : "text-muted hover:text-ink hover:bg-white/5")}
            data-testid="tool-more"
          >
            {inMore ? (() => { const I = more.find((m) => m.id === tool)!.icon; return <I className="w-4 h-4" />; })() : "More"}
            <ChevronDown className="w-3 h-3" />
          </button>
          {menu && (
            <>
              <div className="fixed inset-0 z-20" onPointerDown={() => setMenu(false)} />
              <div className="absolute left-0 top-8 z-30 w-60 rounded-md border border-line-2 bg-panel-2 shadow-xl py-1 text-[12px]">
                {more.map((m) => (
                  <MenuItem key={m.id} icon={m.icon} label={m.label} kbd={m.key} on={tool === m.id} onClick={() => setTool(m.id)} />
                ))}
                <div className="my-1 border-t border-line" />
                <MenuItem icon={Magnet} label="Snap outline to object edges" on={snap} onClick={() => setSnap(!snap)} />
                <MenuItem icon={Plus} label="Grow selection" onClick={() => { setMenu(false); morph(true); }} />
                <MenuItem icon={Minus} label="Shrink selection" onClick={() => { setMenu(false); morph(false); }} />
                <MenuItem icon={FlipHorizontal2} label="Invert selection" kbd="I" onClick={() => { setMenu(false); invert(); }} />
              </div>
            </>
          )}
        </div>
        <div className="w-px h-5 bg-line mx-1" />
        <div className="flex rounded-md bg-[#0d1015] border border-line p-0.5 gap-0.5 shrink-0" title="Add to the selection, or erase from it (E; or hold Alt / use the right mouse button)">
          <button onClick={() => setErase(false)} className={cx("h-6 px-2 rounded text-[11.5px] flex items-center gap-1", !er ? "bg-ok/15 text-ok" : "text-muted hover:text-ink")} data-testid="mode-add">
            <Plus className="w-3 h-3" /> Add
          </button>
          <button onClick={() => setErase(true)} className={cx("h-6 px-2 rounded text-[11.5px] flex items-center gap-1", er ? "bg-bad/15 text-bad" : "text-muted hover:text-ink")} data-testid="mode-erase">
            <Eraser className="w-3 h-3" /> Erase
          </button>
        </div>
      </div>
      <div className="flex items-center gap-2 px-3 h-9 border-b border-line text-[11px]">
        <span className={cx("truncate flex-1", er ? "text-bad/90" : "text-muted")} data-testid="editor-hint">
          {er && (tool === "draw" || tool === "points" || tool === "brush") ? "Erasing — " : ""}
          {hint}
        </span>
        {(tool === "draw" || tool === "brush") && (
          <div className="flex items-center gap-2 w-28 shrink-0" title="Line width ([ and ])">
            <span className="text-faint">Line</span>
            <input type="range" min={2} max={200} value={Math.min(200, lineW)} style={{ ["--fill" as any]: `${(Math.min(200, lineW) / 200) * 100}%` }} onChange={(e) => setLineW(+e.target.value)} aria-label="Line width" />
          </div>
        )}
        <div className="flex items-center -mr-2 shrink-0">
          <IconBtn icon={Undo2} tip="Undo (Ctrl+Z)" onClick={undo} disabled={!history.current.length} />
          <IconBtn icon={Redo2} tip="Redo (Ctrl+Shift+Z)" onClick={redo} disabled={!future.current.length} />
          <IconBtn icon={Trash2} tip="Clear selection" onClick={clear} disabled={!has} />
        </div>
      </div>
      <div
        ref={wrap}
        className="relative flex-1 overflow-hidden checker touch-none"
        style={{ cursor }}
        onWheel={onWheel}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={() => {
          stroke.current = null;
          dragDot.current = null;
          panning.current = null;
          clearPreview();
        }}
        onDoubleClick={() => tool === "points" && pts.length >= 3 && finishPoints()}
        onPointerEnter={() => (active.current = true)}
        onPointerLeave={() => {
          active.current = false;
          hover.current = null;
          moveCursor(null);
          if (tool === "points") requestPreview();
        }}
        onContextMenu={(e) => e.preventDefault()}
        data-testid={`canvas-${label}`}
      >
        <div ref={stage} className="absolute top-0 left-0 origin-top-left" style={{ width: W, height: H, transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}>
          <img src={imageUrl(imageId, MAX_MASK_SIDE)} width={W} height={H} className="absolute inset-0 w-full h-full pointer-events-none" draggable={false} alt={label} style={{ imageRendering: view.s > 2 ? "pixelated" : "auto" }} />
          <canvas ref={tint} className="absolute inset-0 w-full h-full pointer-events-none" style={{ imageRendering: view.s > 2 ? "pixelated" : "auto" }} />
          <canvas ref={preview} className="absolute inset-0 w-full h-full pointer-events-none" />
          <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible" viewBox={`0 0 ${W} ${H}`}>
            {outline && (
              <g data-testid="selection-outline">
                <path d={outline} fill="none" stroke="#000" strokeOpacity={0.75} strokeWidth={sw * 2.2} strokeLinejoin="round" />
                <path d={outline} fill="none" stroke="#fff" strokeWidth={sw} strokeDasharray={`${5 / view.s} ${4 / view.s}`} strokeLinejoin="round" />
              </g>
            )}
            <polyline ref={liveLine} fill="none" stroke={er ? "#ffb0b0" : "#fff"} strokeOpacity={0.9} strokeWidth={sw} strokeDasharray={`${4 / view.s} ${3 / view.s}`} />
            {tool === "points" && pts.length > 0 && (
              <>
                <polygon points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={ptsErase.current ? "#ff9b9b" : "#fff"} strokeWidth={sw * 1.4} strokeLinejoin="round" />
                <polyline ref={rubber} fill="none" stroke={ptsErase.current ? "#ff9b9b" : "#fff"} strokeOpacity={0.7} strokeWidth={sw} strokeDasharray={`${4 / view.s} ${3 / view.s}`} />
                {pts.map((p, i) =>
                  i === 0 ? (
                    <circle key={i} ref={firstDot} cx={p[0]} cy={p[1]} r={5 / view.s} fill={ptsErase.current ? "#ff6b6b" : color} stroke="#000" strokeWidth={sw} />
                  ) : (
                    <circle key={i} cx={p[0]} cy={p[1]} r={3.2 / view.s} fill="#fff" stroke="#000" strokeWidth={sw * 0.8} />
                  ),
                )}
              </>
            )}
            <circle ref={cursorRing} r={lineW / 2} fill="none" stroke={er ? "#ff8a8a" : "#fff"} strokeWidth={sw} visibility="hidden" />
          </svg>
        </div>
        {(busy || snapping) && (
          <div className="absolute top-2 left-2 text-[11px] bg-black/65 rounded px-2 py-1 text-muted pointer-events-none flex items-center gap-1.5" data-testid="snapping">
            <Magnet className="w-3 h-3" /> {busy ? "Selecting…" : "Snapping to edges…"}
          </div>
        )}
        {!has && !pts.length && (
          <div className="absolute top-3 inset-x-3 flex justify-center pointer-events-none" data-testid="empty-hint">
            <div className="text-[14px] font-medium bg-black/75 rounded-lg px-4 py-2.5 text-white text-center leading-snug shadow-lg">
              {tool === "points" ? "Click dots around the area you want to swap — the inside is selected" : tool === "draw" ? "Draw around or over the area you want to swap — the inside is selected" : hint}
            </div>
          </div>
        )}
        <div className="absolute bottom-2 right-2 flex items-center gap-1 bg-black/60 backdrop-blur rounded-md p-1" onPointerDown={(e) => e.stopPropagation()}>
          <IconBtn icon={ZoomOut} tip="Zoom out" onClick={() => zoomBy(1 / 1.25)} />
          <span className="text-[10px] font-mono w-10 text-center text-muted">{Math.round(view.s * scale0 * 100)}%</span>
          <IconBtn icon={ZoomIn} tip="Zoom in (wheel)" onClick={() => zoomBy(1.25)} />
          <IconBtn icon={Maximize} tip="Fit (F)" onClick={fit} />
        </div>
      </div>
    </div>
  );
});

function ToolBtn({ on, icon: Icon, label, tip, onClick, testid }: { on: boolean; icon: any; label: string; tip: string; onClick: () => void; testid?: string }) {
  return (
    <button
      title={tip}
      aria-label={tip}
      onClick={onClick}
      data-testid={testid}
      className={cx("h-7 px-2 rounded flex items-center gap-1.5 text-[12px] font-medium transition shrink-0", on ? "bg-accent/20 text-accent ring-1 ring-accent/40" : "text-muted hover:text-ink hover:bg-white/5")}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );
}

function MenuItem({ icon: Icon, label, kbd, on, onClick }: { icon: any; label: string; kbd?: string; on?: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className="w-full flex items-center gap-2 px-3 h-8 text-left text-muted hover:text-ink hover:bg-white/5">
      <Icon className="w-3.5 h-3.5 shrink-0" />
      <span className="flex-1 truncate">{label}</span>
      {kbd && <span className="text-faint font-mono text-[10px]">{kbd}</span>}
      <Check className={cx("w-3.5 h-3.5", on ? "text-accent" : "invisible")} />
    </button>
  );
}

function IconBtn({ icon: Icon, tip, onClick, disabled }: { icon: any; tip: string; onClick: () => void; disabled?: boolean }) {
  return (
    <button title={tip} aria-label={tip} onClick={onClick} disabled={disabled} className="w-7 h-7 rounded flex items-center justify-center text-muted hover:text-ink hover:bg-white/5 disabled:opacity-30">
      <Icon className="w-4 h-4" />
    </button>
  );
}
