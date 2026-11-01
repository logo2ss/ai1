import { useCallback, useEffect, useRef, useState } from "react";
import { Download, PersonStanding, RotateCcw, Sparkles } from "lucide-react";
import { api, ApiError, downloadUrl, imageUrl, ImageMeta, runJob, withToken } from "../api";
import { useApp, useModel } from "../state";
import { Button, CompareSlider, cx, ErrorNotice, Notice, ProgressBar, Section, Select, Slider, Toggle } from "../components/ui";

type Pt = [number, number];
interface Detect {
  bbox: [number, number, number, number];
  lattice: Pt[][];
  rows: string[];
  people: number;
  full_body: boolean;
  method?: string;
}
const SL = { height: 1, legs: 1, torso: 1, width: 1, shoulders: 1, waist: 1, hips: 1 };
type Sliders = typeof SL;

/** Lattice positions from the detected base lattice + proportion sliders + manual handle offsets. */
function computeLattice(base: Pt[][], s: Sliders, manual: Record<string, Pt>): Pt[][] {
  const R = base.length;
  const C = base[0].length;
  const L = base.map((r) => r.map((p) => [p[0], p[1]] as Pt));
  const y = (r: number) => base[r][0][1];
  const bottom = y(R - 1);
  const [, SH, WA, HI, KN] = [0, 1, 2, 3, 4];
  // vertical: legs (knees+hips relative to feet), torso (rows above hips), overall height (about the feet)
  const dyHip = (bottom - y(HI)) * (s.legs - 1);
  const dyKnee = (bottom - y(KN)) * (s.legs - 1);
  const dyTorso = (y(HI) - y(SH)) * (s.torso - 1);
  for (let c = 0; c < C; c++) {
    L[KN][c][1] -= dyKnee;
    L[HI][c][1] -= dyHip;
    L[WA][c][1] -= dyHip + dyTorso * 0.5;
    L[SH][c][1] -= dyHip + dyTorso;
    L[0][c][1] -= dyHip + dyTorso;
  }
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) L[r][c][1] = bottom - (bottom - L[r][c][1]) * s.height;
  // horizontal: inner columns scale about the centre column
  const cx = base[0][Math.floor(C / 2)][0];
  const rowScale = [1, s.shoulders, s.waist, s.hips, 1, 1];
  for (let r = 0; r < R; r++) {
    for (let c = 1; c < C - 1; c++) {
      const k = s.width * (rowScale[r] ?? 1) * (r === 4 ? Math.sqrt(s.hips) : 1);
      L[r][c][0] = cx + (base[r][c][0] - cx) * k;
    }
  }
  for (const key in manual) {
    const [r, c] = key.split(",").map(Number);
    L[r][c][0] += manual[key][0];
    L[r][c][1] += manual[key][1];
  }
  return L;
}

export default function BodyEditor({ img }: { img: ImageMeta | null }) {
  const { go, system } = useApp();
  const model = useModel("pose-landmarker");
  const human = useModel("segment-human");
  const canDetect = !!human?.installed || (!!model?.installed && system?.mediapipe?.usable !== false);
  const [det, setDet] = useState<Detect | null>(null);
  const [sl, setSl] = useState<Sliders>(SL);
  const [manual, setManual] = useState<Record<string, Pt>>({});
  const [settings, setSettings] = useState({ mode: "person" as "person" | "liquify", surroundings: 0.35, remove_revealed: true, edge_softness: 0.4, match_grain: true });
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<{ p: number; msg: string } | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [view, setView] = useState<"edit" | "compare">("edit");
  const [error, setError] = useState<ApiError | null>(null);
  const [showGrid, setShowGrid] = useState(true);
  const box = useRef<HTMLDivElement>(null);
  const [sz, setSz] = useState({ w: 0, h: 0 });
  const drag = useRef<{ key: string; start: Pt; orig: Pt } | null>(null);

  useEffect(() => {
    setDet(null);
    setResult(null);
    setPreview(null);
    setSl(SL);
    setManual({});
    if (!img || !canDetect) return;
    api
      .post<Detect>("/api/body/detect", { image_id: img.id })
      .then(setDet)
      .catch((e) => setError(e));
  }, [img?.id, canDetect]);

  useEffect(() => {
    if (!img) return;
    const ro = new ResizeObserver(() => {
      const r = box.current?.getBoundingClientRect();
      if (!r) return;
      const s = Math.min(r.width / img.width, r.height / img.height);
      setSz({ w: img.width * s, h: img.height * s });
    });
    if (box.current) ro.observe(box.current);
    return () => ro.disconnect();
  }, [img]);

  const lattice = det ? computeLattice(det.lattice, sl, manual) : null;
  const changed = !!det && (JSON.stringify(sl) !== JSON.stringify(SL) || Object.keys(manual).length > 0);

  // debounced live preview (fast path on the engine)
  const req = useRef(0);
  const refresh = useCallback(async () => {
    if (!img || !det || !lattice) return;
    const id = ++req.current;
    if (!changed) {
      setPreview(null);
      return;
    }
    setBusy(true);
    try {
      const blob = await api.post<Blob>("/api/body/preview", { image_id: img.id, src: det.lattice, dst: lattice, settings });
      if (id === req.current) setPreview(URL.createObjectURL(blob));
    } catch (e) {
      setError(e as ApiError);
    } finally {
      if (id === req.current) setBusy(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [img, det, JSON.stringify(lattice), JSON.stringify(settings), changed]);
  useEffect(() => {
    const t = setTimeout(refresh, 300);
    return () => clearTimeout(t);
  }, [refresh]);

  const apply = async () => {
    if (!img || !det || !lattice) return;
    setError(null);
    setJob({ p: 0, msg: "Starting" });
    try {
      const r = await runJob<{ image_id: string }>(api.post("/api/body/apply", { image_id: img.id, src: det.lattice, dst: lattice, settings }), (j) => setJob({ p: j.progress, msg: j.message }));
      setResult(r.image_id);
      setView("compare");
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setJob(null);
    }
  };

  const toImg = (e: React.PointerEvent): Pt => {
    const r = box.current!.querySelector("[data-stage]")!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * img!.width, ((e.clientY - r.top) / r.height) * img!.height];
  };
  const set = (k: keyof Sliders, v: number) => setSl((s) => ({ ...s, [k]: v }));
  const pct = (v: number) => `${v >= 1 ? "+" : ""}${Math.round((v - 1) * 100)}%`;
  const sw = img ? Math.max(img.width, img.height) / 300 : 1;

  if (!img) return <div className="h-full flex items-center justify-center text-muted">Load a photo with a person (left panel) to edit body proportions.</div>;

  return (
    <div className="h-full flex">
      <div className="flex-1 min-w-0 flex flex-col p-4 gap-3">
        <div className="flex items-center gap-2">
          <div className="flex rounded-md bg-panel border border-line p-0.5">
            {(["edit", "compare"] as const).map((v) => (
              <button key={v} disabled={v === "compare" && !result} onClick={() => setView(v)} className={cx("px-3 h-7 rounded text-[12px] disabled:opacity-40", view === v ? "bg-panel-2 text-ink border border-line-2" : "text-muted hover:text-ink")}>
                {v === "edit" ? "Edit (live preview)" : "Before / After (full quality)"}
              </button>
            ))}
          </div>
          {view === "edit" && <Toggle label="Show lattice" checked={showGrid} onChange={setShowGrid} />}
          {busy && <span className="text-[11px] text-muted">Updating preview…</span>}
        </div>
        <div ref={box} className="relative flex-1 min-h-0 rounded-lg border border-line bg-panel checker flex items-center justify-center overflow-hidden">
          {view === "compare" && result ? (
            <div className="absolute inset-0 p-3">
              <CompareSlider before={imageUrl(img.id, 2400)} after={imageUrl(result, 2400)} labels={["Original", "Result"]} />
            </div>
          ) : (
            <div
              data-stage
              className="relative"
              style={{ width: sz.w, height: sz.h }}
              onPointerMove={(e) => {
                if (!drag.current) return;
                const p = toImg(e);
                const d = drag.current;
                setManual((m) => ({ ...m, [d.key]: [d.orig[0] + p[0] - d.start[0], d.orig[1] + p[1] - d.start[1]] }));
              }}
              onPointerUp={() => (drag.current = null)}
            >
              <img src={preview || imageUrl(img.id, 1600)} className="absolute inset-0 w-full h-full select-none" draggable={false} alt="body" />
              {lattice && showGrid && (
                <svg className="absolute inset-0 w-full h-full overflow-visible" viewBox={`0 0 ${img.width} ${img.height}`}>
                  {lattice.map((row, r) => (
                    <polyline key={"r" + r} points={row.map((p) => p.join(",")).join(" ")} fill="none" stroke={r === 0 || r === lattice.length - 1 ? "rgba(255,255,255,.5)" : "#8f83ff"} strokeWidth={sw * 0.8} strokeDasharray={r === 0 || r === lattice.length - 1 ? `${sw * 3} ${sw * 2}` : ""} />
                  ))}
                  {lattice[0].map((_, c) => (
                    <polyline key={"c" + c} points={lattice.map((row) => row[c].join(",")).join(" ")} fill="none" stroke={c === 0 || c === lattice[0].length - 1 ? "rgba(255,255,255,.5)" : "#8f83ff"} strokeWidth={sw * 0.8} strokeDasharray={c === 0 || c === lattice[0].length - 1 ? `${sw * 3} ${sw * 2}` : ""} />
                  ))}
                  {lattice.map((row, r) =>
                    row.map((p, c) => (
                      <circle
                        key={`${r},${c}`}
                        cx={p[0]}
                        cy={p[1]}
                        r={sw * 3.2}
                        fill={manual[`${r},${c}`] ? "#f2b84b" : "#fff"}
                        stroke="#1b1f29"
                        strokeWidth={sw * 0.8}
                        style={{ cursor: "grab" }}
                        onPointerDown={(e) => {
                          (e.target as Element).setPointerCapture?.(e.pointerId);
                          drag.current = { key: `${r},${c}`, start: toImg(e), orig: manual[`${r},${c}`] || [0, 0] };
                        }}
                      >
                        <title>{`${det!.rows[r]} — drag to reshape`}</title>
                      </circle>
                    )),
                  )}
                  {det!.rows.map((name, r) => (
                    <text key={name} x={lattice[r][0][0] - sw * 4} y={lattice[r][0][1] + sw * 3} fontSize={sw * 9} fill="#cfc9ff" textAnchor="end" fontFamily="Inter, system-ui">
                      {name}
                    </text>
                  ))}
                </svg>
              )}
            </div>
          )}
          {!det && !error && canDetect && <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-[12px]">Finding the person…</div>}
        </div>
        <div className="text-[11px] text-faint">Drag any handle to reshape freely (dashed outer frame = boundary; moving it bends the surroundings). Sliders adjust anatomical rows. The preview is fast; “Apply” renders full quality with background repair.</div>
      </div>
      <aside className="w-[310px] shrink-0 border-l border-line bg-panel flex flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 border-b border-line space-y-3">
            {!canDetect && (
              <Notice kind="warn" title="Person model not installed">
                Install U2-Net Human Segmentation for the offline body editor, including CPUs without AVX.{" "}
                <button className="underline" onClick={() => go("models")}>
                  Install it in Models
                </button>
                .
              </Notice>
            )}
            <ErrorNotice error={error} onClose={() => setError(null)} />
            {det && det.people > 1 && <Notice kind="info">{det.people} people detected — editing the largest one.</Notice>}
            {det && det.method === "segmentation" && <Notice kind="info">Compatibility mode: the body rows were placed from the person outline and standard proportions. Drag the handles if a row (shoulders, waist, hips…) is off.</Notice>}
            {det && !det.full_body && <Notice kind="info">Legs/feet are not fully visible; leg-length changes act on the visible part only.</Notice>}
          </div>
          <Section title="Proportions">
            <Slider label="Overall height" value={sl.height} min={0.85} max={1.15} step={0.005} format={pct} onChange={(v) => set("height", v)} />
            <Slider label="Leg length" value={sl.legs} min={0.85} max={1.2} step={0.005} format={pct} onChange={(v) => set("legs", v)} />
            <Slider label="Torso length" value={sl.torso} min={0.85} max={1.15} step={0.005} format={pct} onChange={(v) => set("torso", v)} />
            <Slider label="Overall width" value={sl.width} min={0.8} max={1.2} step={0.005} format={pct} onChange={(v) => set("width", v)} />
            <Slider label="Shoulders" value={sl.shoulders} min={0.85} max={1.2} step={0.005} format={pct} onChange={(v) => set("shoulders", v)} />
            <Slider label="Waist" value={sl.waist} min={0.75} max={1.2} step={0.005} format={pct} onChange={(v) => set("waist", v)} />
            <Slider label="Hips" value={sl.hips} min={0.8} max={1.2} step={0.005} format={pct} onChange={(v) => set("hips", v)} />
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setSl(SL);
                setManual({});
              }}
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset proportions
            </Button>
          </Section>
          <Section title="Naturalisation">
            <Select
              label="Method"
              value={settings.mode}
              onChange={(v) => setSettings((s) => ({ ...s, mode: v }))}
              options={[
                { value: "person", label: "Person layer" },
                { value: "liquify", label: "Liquify" },
              ]}
            />
            <div className="text-[11px] text-faint -mt-2">
              {settings.mode === "person" ? "Only the person is reshaped: walls and lines stay straight, uncovered background is synthesised, edges and grain are re-matched." : "The whole image is warped smoothly (surroundings bend with the body)."}
            </div>
            {settings.mode === "person" && (
              <>
                <Slider label="Surroundings follow body" value={settings.surroundings} onChange={(v) => setSettings((s) => ({ ...s, surroundings: v }))} hint="Lets nearby shadows and contact areas move with the body." />
                <Slider label="Edge softness" value={settings.edge_softness} onChange={(v) => setSettings((s) => ({ ...s, edge_softness: v }))} />
                <Toggle label="Repair uncovered background" checked={settings.remove_revealed} onChange={(v) => setSettings((s) => ({ ...s, remove_revealed: v }))} />
                <Toggle label="Restore sensor grain" checked={settings.match_grain} onChange={(v) => setSettings((s) => ({ ...s, match_grain: v }))} />
              </>
            )}
          </Section>
        </div>
        <div className="p-4 border-t border-line space-y-2">
          {job ? (
            <ProgressBar value={job.p} message={job.msg} />
          ) : (
            <Button variant="primary" size="lg" className="w-full" disabled={!changed} onClick={apply}>
              <Sparkles className="w-4 h-4" /> Apply &amp; naturalise
            </Button>
          )}
          {result && (
            <Button className="w-full" onClick={() => downloadUrl(withToken(`/api/images/${result}/export?format=png`))}>
              <Download className="w-4 h-4" /> Export PNG
            </Button>
          )}
          <div className="text-[10.5px] text-faint flex items-center gap-1.5">
            <PersonStanding className="w-3.5 h-3.5" /> Visualise how proportions would look. Only edit photos you have the right to use.
          </div>
        </div>
      </aside>
    </div>
  );
}
