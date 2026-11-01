import { useEffect, useState } from "react";
import { ArrowLeftRight, Download, ScanFace, SquareDashedMousePointer, Sparkles, RotateCcw } from "lucide-react";
import { api, ApiError, imageUrl, ImageMeta, runJob, uploadImage, withToken, downloadUrl } from "../api";
import { useModel, useApp } from "../state";
import { Button, CompareSlider, cx, Dropzone, ErrorNotice, Notice, ProgressBar, Section, Select, Slider, Stat, Toggle } from "../components/ui";
import { ImageStage } from "../components/ImageStage";

interface FaceInfo {
  bbox: [number, number, number, number];
  yaw: number;
  pitch: number;
  roll: number;
  size: number;
  source: string;
  mouth_open: boolean;
}
interface Slot {
  meta: ImageMeta;
  faces: FaceInfo[] | null;
  selected: number;
  detecting: boolean;
  error?: string;
  note?: string;
}
interface Analysis {
  compatible: boolean;
  score: number;
  issues: string[];
  warnings: string[];
  yaw_difference: number;
  pitch_difference: number;
  self_occlusion: number;
  resolution_ratio: number;
  light_direction_difference: number;
}

const DEFAULTS = {
  blend_strength: 1.0,
  color_match: 0.9,
  lighting_match: 0.7,
  edge_refinement: 0.5,
  preserve_source_shape: 0.2,
  mask_extent: 1.0,
  keep_target_mouth: true,
  match_grain: true,
  match_sharpness: true,
  force: false,
};

export default function FacePage() {
  const model = useModel("face-landmarker");
  const { go } = useApp();
  const [src, setSrc] = useState<Slot | null>(null);
  const [dst, setDst] = useState<Slot | null>(null);
  const [settings, setSettings] = useState(DEFAULTS);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [job, setJob] = useState<{ p: number; msg: string } | null>(null);
  const [result, setResult] = useState<{ image_id: string; mask_id: string; report: any } | null>(null);
  const [view, setView] = useState<"compare" | "result" | "mask" | "inputs">("inputs");
  const [error, setError] = useState<ApiError | null>(null);
  const [manual, setManual] = useState<"src" | "dst" | null>(null);
  const [fmt, setFmt] = useState<"png" | "jpg" | "tiff">("png");

  const [analyzing, setAnalyzing] = useState(false);
  const [uploading, setUploading] = useState<"src" | "dst" | null>(null);

  // Update whichever slot currently holds this image (robust against "Swap source and target" mid-request).
  const updateById = (id: string, f: (s: Slot) => Slot) => {
    setSrc((s) => (s && s.meta.id === id ? f(s) : s));
    setDst((s) => (s && s.meta.id === id ? f(s) : s));
  };
  const errText = (e: unknown) => {
    const a = e as ApiError;
    return [a?.message || String(e), a?.hint].filter(Boolean).join(" — ");
  };

  const detect = async (slot: Slot) => {
    const id = slot.meta.id;
    updateById(id, (s) => ({ ...s, detecting: true, error: undefined }));
    try {
      const r = await api.post<{ faces: FaceInfo[]; note?: string }>("/api/face/detect", { image_id: id });
      updateById(id, (s) => ({ ...s, faces: r.faces, selected: r.faces.length ? largest(r.faces) : -1, detecting: false, error: undefined, note: r.note }));
    } catch (e) {
      updateById(id, (s) => ({ ...s, faces: null, selected: -1, detecting: false, error: errText(e) }));
    }
  };
  const load = async (f: File, which: "src" | "dst") => {
    setError(null);
    setUploading(which);
    try {
      const meta = await uploadImage(f);
      const slot: Slot = { meta, faces: null, selected: -1, detecting: true };
      const set = which === "src" ? setSrc : setDst;
      set(slot);
      setResult(null);
      setView("inputs");
      setSettings((s) => ({ ...s, force: false }));
      detect(slot);
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setUploading(null);
    }
  };

  // live compatibility analysis
  useEffect(() => {
    setAnalysis(null);
    setAnalyzing(false);
    if (!src?.faces || !dst?.faces || src.selected < 0 || dst.selected < 0) return;
    let stale = false;
    setAnalyzing(true);
    api
      .post<Analysis>("/api/face/analyze", { source: { image_id: src.meta.id, face_index: src.selected }, target: { image_id: dst.meta.id, face_index: dst.selected } })
      .then((a) => !stale && setAnalysis(a))
      .catch((e) => !stale && setError(e))
      .finally(() => !stale && setAnalyzing(false));
    return () => {
      stale = true;
    };
  }, [src?.meta.id, src?.selected, src?.faces, dst?.meta.id, dst?.selected, dst?.faces]);

  const run = async () => {
    if (!src || !dst) return;
    setError(null);
    setJob({ p: 0, msg: "Starting" });
    try {
      const r = await runJob<any>(
        api.post("/api/face/swap", { source: { image_id: src.meta.id, face_index: src.selected }, target: { image_id: dst.meta.id, face_index: dst.selected }, settings }),
        (j) => setJob({ p: j.progress, msg: j.message }),
      );
      if (!r || !r.image_id) throw new ApiError("no_result", "The face replacement finished but returned no image.", "Try again; if it keeps happening, check the engine log.");
      setResult(r);
      setView("compare");
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setJob(null);
    }
  };

  const swapRoles = () => {
    setSrc(dst);
    setDst(src);
    setResult(null);
    setView("inputs");
  };

  const modelMissing = !!model && !model.installed;
  const slotBlocker = (slot: Slot | null, name: string, side: string): string | null => {
    if (!slot) return uploading === (name === "source" ? "src" : "dst") ? `Uploading the ${name} image…` : `Load a ${name} image (${side} panel).`;
    if (slot.detecting) return `Detecting faces in the ${name} image…`;
    if (slot.error) return modelMissing ? "Face detection needs the Face Landmarker model — install it in Models." : `Face detection failed on the ${name} image — see the ${side} panel, or use “Mark face”.`;
    if (!slot.faces || slot.faces.length === 0) return `No face found in the ${name} image — use “Mark face” and draw a box around it.`;
    if (slot.selected < 0) return `Click a face in the ${name} image to select it.`;
    return null;
  };
  const incompatibleError = error?.code === "incompatible";
  const showForce = (!!analysis && !analysis.compatible) || incompatibleError;
  const blocker =
    slotBlocker(src, "source", "left") ??
    slotBlocker(dst, "target", "right") ??
    (showForce && !settings.force ? "These faces look incompatible — enable “Proceed anyway” below to replace the face regardless." : null);
  const ready = !blocker;
  const set = <K extends keyof typeof DEFAULTS>(k: K, v: (typeof DEFAULTS)[K]) => setSettings((s) => ({ ...s, [k]: v }));

  return (
    <div className="h-full flex">
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 flex items-center gap-3 px-5 border-b border-line">
          <ScanFace className="w-5 h-5 text-accent" />
          <h1 className="font-semibold">Face Replacement</h1>
          <div className="flex-1" />
          {result && (
            <div className="flex rounded-md bg-panel border border-line p-0.5">
              {(["inputs", "compare", "result", "mask"] as const).map((v) => (
                <button key={v} onClick={() => setView(v)} className={cx("px-3 h-7 rounded text-[12px] capitalize", view === v ? "bg-panel-2 text-ink border border-line-2" : "text-muted hover:text-ink")}>
                  {v === "compare" ? "Before / After" : v}
                </button>
              ))}
            </div>
          )}
        </header>
        {model && !model.installed && (
          <div className="p-4 pb-0">
            <Notice kind="warn" title="Face Landmarker model is not installed">
              Face detection needs the MediaPipe Face Landmarker (3.7 MB, Apache-2.0).{" "}
              <button className="underline" onClick={() => go("models")}>
                Install it in Models
              </button>
              .
            </Notice>
          </div>
        )}
        <div className="flex-1 min-h-0 p-4">
          {result && view !== "inputs" && dst ? (
            <div className="w-full h-full rounded-lg bg-panel border border-line overflow-hidden p-3">
              {view === "compare" && <CompareSlider before={imageUrl(dst.meta.id, 2400)} after={imageUrl(result.image_id, 2400)} labels={["Original", "Result"]} />}
              {view === "result" && <ImageStage src={imageUrl(result.image_id, 2400)} width={dst.meta.width} height={dst.meta.height} />}
              {view === "mask" && <ImageStage src={imageUrl(result.mask_id, 2400)} width={dst.meta.width} height={dst.meta.height} />}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4 h-full">
              {(["src", "dst"] as const).map((which) => {
                const slot = which === "src" ? src : dst;
                const setSlot = which === "src" ? setSrc : setDst;
                return (
                  <FacePanel
                    key={which}
                    title={which === "src" ? "Source face" : "Target image"}
                    subtitle={which === "src" ? "Provides the face" : "Receives the face"}
                    slot={slot}
                    manual={manual === which}
                    onManual={() => setManual(manual === which ? null : which)}
                    onFile={(f) => load(f, which)}
                    onRetry={() => slot && detect(slot)}
                    onSelect={(i) => slot && setSlot({ ...slot, selected: i })}
                    onBox={async (box) => {
                      if (!slot) return;
                      setManual(null);
                      const id = slot.meta.id;
                      updateById(id, (s) => ({ ...s, detecting: true, error: undefined }));
                      try {
                        const r = await api.post<{ faces: FaceInfo[]; index: number }>("/api/face/detect-box", { image_id: id, box });
                        updateById(id, (s) => ({ ...s, faces: r.faces, selected: r.index, detecting: false, error: undefined }));
                      } catch (e) {
                        updateById(id, (s) => ({ ...s, detecting: false, error: errText(e) }));
                      }
                    }}
                  />
                );
              })}
            </div>
          )}
        </div>
      </div>

      <aside className="w-[310px] shrink-0 border-l border-line bg-panel flex flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 border-b border-line space-y-3">
            <Button className="w-full" onClick={swapRoles} disabled={!src && !dst}>
              <ArrowLeftRight className="w-4 h-4" /> Swap source and target
            </Button>
          </div>
          <Section title="Compatibility analysis">
            {!analysis && <div className="text-[12px] text-muted">{analyzing ? "Checking compatibility…" : "Load both images and select a face in each."}</div>}
            {analysis && (
              <>
                <div className="flex items-center gap-3">
                  <div className={cx("text-[22px] font-semibold tabular-nums", analysis.score > 0.75 ? "text-ok" : analysis.score > 0.5 ? "text-warn" : "text-bad")}>{Math.round(analysis.score * 100)}</div>
                  <div className="text-[12px] text-muted leading-tight">
                    {analysis.compatible ? (analysis.score > 0.75 ? "Good match — a natural result is expected." : "Usable with limitations.") : "Not compatible for a realistic result."}
                  </div>
                </div>
                <div>
                  <Stat label="Head yaw difference" value={`${analysis.yaw_difference}°`} tone={analysis.yaw_difference > 38 ? "bad" : analysis.yaw_difference > 20 ? "warn" : "ok"} />
                  <Stat label="Head pitch difference" value={`${analysis.pitch_difference}°`} tone={analysis.pitch_difference > 32 ? "bad" : analysis.pitch_difference > 18 ? "warn" : "ok"} />
                  <Stat label="Self-occlusion" value={`${Math.round(analysis.self_occlusion * 100)}%`} tone={analysis.self_occlusion > 0.06 ? "bad" : analysis.self_occlusion > 0.02 ? "warn" : "ok"} />
                  <Stat label="Resolution ratio" value={`${analysis.resolution_ratio}×`} tone={analysis.resolution_ratio > 3 ? "warn" : "ok"} />
                  <Stat label="Light direction difference" value={`${analysis.light_direction_difference}°`} />
                </div>
                {analysis.issues.map((t, i) => (
                  <Notice key={i} kind="error">
                    {t}
                  </Notice>
                ))}
                {analysis.warnings.map((t, i) => (
                  <Notice key={i} kind="warn">
                    {t}
                  </Notice>
                ))}
              </>
            )}
          </Section>
          <Section title="Blending">
            <Slider label="Blending strength (gradient domain)" value={settings.blend_strength} onChange={(v) => set("blend_strength", v)} hint="How strongly the seam is solved in the gradient domain (Poisson)." />
            <Slider label="Colour matching" value={settings.color_match} onChange={(v) => set("color_match", v)} hint="Adapts skin tone and colour distribution to the target." />
            <Slider label="Lighting match" value={settings.lighting_match} onChange={(v) => set("lighting_match", v)} hint="Transfers the target's shading: light direction, shadows and highlights." />
            <Slider label="Edge refinement" value={settings.edge_refinement} onChange={(v) => set("edge_refinement", v)} hint="Width and edge-awareness of the transition." />
          </Section>
          <Section title="Geometry & mask" defaultOpen={false}>
            <Slider label="Preserve source proportions" value={settings.preserve_source_shape} onChange={(v) => set("preserve_source_shape", v)} hint="0 = follow the target's face shape and expression, 1 = keep the source's proportions." />
            <Slider label="Replaced area" value={settings.mask_extent} min={0.8} max={1.1} step={0.01} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set("mask_extent", v)} />
            <Toggle label="Keep target's open mouth (teeth)" checked={settings.keep_target_mouth} onChange={(v) => set("keep_target_mouth", v)} />
            <Toggle label="Match sensor noise / grain" checked={settings.match_grain} onChange={(v) => set("match_grain", v)} />
            <Toggle label="Match sharpness" checked={settings.match_sharpness} onChange={(v) => set("match_sharpness", v)} />
            <Button size="sm" variant="ghost" onClick={() => setSettings(DEFAULTS)}>
              <RotateCcw className="w-3.5 h-3.5" /> Reset to defaults
            </Button>
          </Section>
          {result && (
            <Section title="Result quality">
              <Stat label="Seam visibility (1 = invisible)" value={result.report?.quality?.seam_visibility ?? "—"} />
              <Stat label="Skin ΔE before → after" value={`${result.report?.quality?.skin_delta_e_before ?? "—"} → ${result.report?.quality?.skin_delta_e_after ?? "—"}`} />
              <Stat label="Transition ΔE" value={result.report?.quality?.transition_delta_e ?? "—"} />
              {result.report.camera?.sharpness && <Stat label="Sharpness" value={`${result.report.camera.sharpness.action} ${result.report.camera.sharpness.amount || ""}`} />}
              {result.report.camera?.noise && <Stat label="Grain" value={result.report.camera.noise.action} />}
            </Section>
          )}
        </div>
        <div className="p-4 border-t border-line space-y-3">
          <ErrorNotice error={error} onClose={() => setError(null)} />
          {showForce && <Toggle label="Proceed anyway (result will be limited)" checked={settings.force} onChange={(v) => set("force", v)} />}
          {!job && blocker && (
            <div className="text-[12px] text-warn leading-snug" role="status" data-testid="face-blocker">
              {blocker}
            </div>
          )}
          {!job && !blocker && analyzing && <div className="text-[12px] text-muted">Checking compatibility…</div>}
          {job ? (
            <ProgressBar value={job.p} message={job.msg} />
          ) : (
            <Button variant="primary" size="lg" className="w-full" disabled={!ready} title={blocker ?? undefined} onClick={run}>
              <Sparkles className="w-4 h-4" /> {result ? "Regenerate" : "Replace face"}
            </Button>
          )}
          {result && (
            <div className="flex gap-2">
              <Select label="" value={fmt} onChange={setFmt} options={[{ value: "png", label: "PNG" }, { value: "jpg", label: "JPEG" }, { value: "tiff", label: "TIFF" }]} />
              <Button className="self-end" onClick={() => downloadUrl(withToken(`/api/images/${result.image_id}/export?format=${fmt}&quality=96`))}>
                <Download className="w-4 h-4" /> Export
              </Button>
            </div>
          )}
          <div className="text-[10.5px] text-faint leading-relaxed">Only edit photos you have the right to use, and respect the people shown. Exports are marked as edited in their metadata.</div>
        </div>
      </aside>
    </div>
  );
}

function largest(f: FaceInfo[]) {
  let best = 0;
  f.forEach((x, i) => (x.size > f[best].size ? (best = i) : 0));
  return best;
}

function FacePanel({
  title,
  subtitle,
  slot,
  onFile,
  onSelect,
  manual,
  onManual,
  onBox,
  onRetry,
}: {
  title: string;
  subtitle: string;
  slot: Slot | null;
  onFile: (f: File) => void;
  onSelect: (i: number) => void;
  manual: boolean;
  onManual: () => void;
  onBox: (b: [number, number, number, number]) => void;
  onRetry: () => void;
}) {
  const [drag, setDrag] = useState<[number, number, number, number] | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const detecting = !!slot?.detecting;
  useEffect(() => {
    setElapsed(0);
    if (!detecting) return;
    const t0 = Date.now();
    const iv = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(iv);
  }, [detecting, slot?.meta.id]);
  return (
    <div className="flex flex-col rounded-lg bg-panel border border-line overflow-hidden min-h-0" data-testid={`facepanel-${title}`}>
      <div className="h-10 flex items-center gap-2 px-3 border-b border-line">
        <span className="font-medium text-[13px]">{title}</span>
        <span className="text-faint text-[11px]">{subtitle}</span>
        <div className="flex-1" />
        {slot && (
          <>
            <span className="text-[11px] text-muted">{slot.detecting ? `Detecting faces…${elapsed >= 2 ? ` ${elapsed}s` : ""}` : slot.error ? "Detection failed" : `${slot.faces?.length ?? 0} face${slot.faces?.length === 1 ? "" : "s"}`}</span>
            <Button size="sm" variant={manual ? "primary" : "ghost"} onClick={onManual} title="Draw a box around a face the detector missed">
              <SquareDashedMousePointer className="w-3.5 h-3.5" /> Mark face
            </Button>
            <label className="text-[11px] text-accent cursor-pointer hover:underline">
              Replace
              <input type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
            </label>
          </>
        )}
      </div>
      <div className="flex-1 min-h-0 p-3 checker">
        {!slot ? (
          <Dropzone onFile={onFile} label={`Load ${title.toLowerCase()}`} />
        ) : (
          <ImageStage
            src={imageUrl(slot.meta.id, 1600)}
            width={slot.meta.width}
            height={slot.meta.height}
            cursor={manual ? "crosshair" : "default"}
            onPointer={(t, p) => {
              if (!manual) return;
              if (t === "down") setDrag([p[0], p[1], p[0], p[1]]);
              if (t === "move" && drag) setDrag([drag[0], drag[1], p[0], p[1]]);
              if (t === "up" && drag) {
                const b: [number, number, number, number] = [Math.min(drag[0], p[0]), Math.min(drag[1], p[1]), Math.max(drag[0], p[0]), Math.max(drag[1], p[1])];
                setDrag(null);
                if (b[2] - b[0] > 8 && b[3] - b[1] > 8) onBox(b);
              }
            }}
          >
            {slot.faces?.map((f, i) => {
              const [x0, y0, x1, y1] = f.bbox;
              const sel = i === slot.selected;
              const sw = Math.max(slot.meta.width, slot.meta.height) / 400;
              return (
                <g key={i} onClick={() => !manual && onSelect(i)} style={{ cursor: manual ? "crosshair" : "pointer", pointerEvents: manual ? "none" : "auto" }}>
                  <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} rx={sw * 3} fill={sel ? "rgba(143,131,255,.12)" : "rgba(255,255,255,.02)"} stroke={sel ? "#8f83ff" : "rgba(255,255,255,.65)"} strokeWidth={sel ? sw * 1.6 : sw} strokeDasharray={sel ? "" : `${sw * 4} ${sw * 3}`} />
                  {sel && (
                    <text x={x0} y={y0 - sw * 4} fill="#cfc9ff" fontSize={sw * 11} fontFamily="Inter, system-ui">
                      {f.source === "manual" ? "manual · " : ""}yaw {Math.round(f.yaw)}° · pitch {Math.round(f.pitch)}°
                    </text>
                  )}
                </g>
              );
            })}
            {drag && <rect x={Math.min(drag[0], drag[2])} y={Math.min(drag[1], drag[3])} width={Math.abs(drag[2] - drag[0])} height={Math.abs(drag[3] - drag[1])} fill="rgba(143,131,255,.15)" stroke="#8f83ff" strokeWidth={Math.max(slot.meta.width, slot.meta.height) / 400} />}
          </ImageStage>
        )}
      </div>
      {slot && slot.note && !slot.detecting && !slot.error && (
        <div className="p-2 border-t border-line">
          <Notice kind="info" title="Compatibility face detector in use">
            {slot.note}
          </Notice>
        </div>
      )}
      {slot && slot.detecting && elapsed >= 4 && (
        <div className="p-2 border-t border-line">
          <Notice kind="info">Still detecting faces ({elapsed}s). The first run loads the face model and can take a while on slower machines.</Notice>
        </div>
      )}
      {slot && slot.error && !slot.detecting && (
        <div className="p-2 border-t border-line">
          <Notice kind="error" title="Face detection failed">
            {slot.error}{" "}
            <button className="underline" onClick={onRetry}>
              Retry
            </button>{" "}
            or use “Mark face” to draw a box around the face.
          </Notice>
        </div>
      )}
      {slot && !slot.error && slot.faces && slot.faces.length === 0 && !slot.detecting && (
        <div className="p-2 border-t border-line">
          <Notice kind="warn">No face detected automatically. Use “Mark face” and draw a box around the face.</Notice>
        </div>
      )}
      {slot && slot.faces && slot.faces.length > 1 && slot.selected >= 0 && !slot.detecting && (
        <div className="px-3 py-1.5 border-t border-line text-[11px] text-muted">{slot.faces.length} faces found — click a box to choose which one to use.</div>
      )}
    </div>
  );
}
