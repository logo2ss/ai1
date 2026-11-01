import { useEffect, useRef, useState } from "react";
import { Download, PersonStanding, RotateCcw, Sparkles } from "lucide-react";
import { api, ApiError, downloadUrl, fileUrl, imageUrl, ImageMeta, runJob } from "../api";
import { useApp, useModel } from "../state";
import { Button, cx, ErrorNotice, Notice, ProgressBar, Section, Select, Slider, Stat, Toggle } from "../components/ui";
import { BodyViewer } from "../three/BodyViewer";
import { CpuBodyViewer } from "../three/CpuBodyViewer";
import { BodyData, BodyViewerHandle, BodyViewOptions, parseBody } from "../three/body";
import { useRenderer } from "../three/webgl";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { CpuRendererNotice, RendererSelect } from "../components/RendererSelect";

interface Result {
  body: string;
  texture: string;
  depth_preview: string;
  header: { cols: number; rows: number; vertices: number };
  method: string;
}

export default function Body3D({ img }: { img: ImageMeta }) {
  const { go } = useApp();
  const depthModel = useModel("depth-anything-v2-small");
  const [settings, setSettings] = useState({ detail: 280, thickness: 0.5, roundness: 0.45 });
  const [opts, setOpts] = useState<BodyViewOptions>({ style: "wireframe", colors: "depth", lineStep: 4, hideHidden: true, relief: 1, autoRotate: true });
  const [job, setJob] = useState<{ p: number; msg: string } | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [data, setData] = useState<BodyData | null>(null);
  const [view, setView] = useState<"model" | "compare" | "depth">("model");
  const [error, setError] = useState<ApiError | null>(null);
  const viewer = useRef<BodyViewerHandle>(null);
  const rend = useRenderer();
  const o = <K extends keyof BodyViewOptions>(k: K, v: BodyViewOptions[K]) => setOpts((x) => ({ ...x, [k]: v }));

  useEffect(() => {
    setResult(null);
    setData(null);
  }, [img.id]);
  useEffect(() => {
    if (!result) return;
    fetch(fileUrl(result.body))
      .then((r) => r.arrayBuffer())
      .then((b) => setData(parseBody(b)))
      .catch((e) => setError(new ApiError("body", "Could not load the 3D body model: " + e.message)));
  }, [result]);

  const build = async () => {
    setError(null);
    setJob({ p: 0, msg: "Starting" });
    try {
      const r = await runJob<Result>(api.post("/api/body3d", { image_id: img.id, settings }), (j) => setJob({ p: j.progress, msg: j.message }));
      setData(null);
      setResult(r);
      setView("model");
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setJob(null);
    }
  };
  const save = async (kind: "glb" | "ply") => {
    if (!viewer.current) return;
    try {
      const blob = kind === "glb" ? await viewer.current.exportGLB() : viewer.current.exportPLY();
      const url = URL.createObjectURL(blob);
      downloadUrl(url, `body-3d.${kind}`);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e: any) {
      setError(new ApiError("export", `Could not export the ${kind.toUpperCase()} file: ${e?.message || e}`));
    }
  };
  const snapshot = () => {
    const url = viewer.current?.screenshot();
    if (url) downloadUrl(url, "body-3d.png");
  };

  // GPU (WebGL) viewer, or frames rendered by the engine when the browser has no WebGL
  const modelView = result && data && (
    <ErrorBoundary key={rend.kind} fill label="3D body viewer" onError={rend.kind === "gpu" ? (e) => rend.reportFailure(e.message) : undefined}>
      {rend.kind === "gpu" ? (
        <BodyViewer
          ref={viewer}
          data={data}
          texture={fileUrl(result.texture)}
          options={opts}
          onRendererError={(e) => rend.reportFailure(e.message)}
          onContextLost={() => rend.reportFailure("the browser lost the WebGL context")}
        />
      ) : (
        <CpuBodyViewer ref={viewer} data={data} texture={fileUrl(result.texture)} files={{ body: result.body, texture: result.texture }} options={opts} />
      )}
    </ErrorBoundary>
  );

  return (
    <div className="h-full flex">
      <div className="flex-1 min-w-0 flex flex-col p-4 gap-3">
        <div className="flex items-center gap-2">
          <div className="flex rounded-md bg-panel border border-line p-0.5">
            {(["model", "compare", "depth"] as const).map((v) => (
              <button key={v} disabled={!result} onClick={() => setView(v)} className={cx("px-3 h-7 rounded text-[12px] disabled:opacity-40", view === v ? "bg-panel-2 text-ink border border-line-2" : "text-muted hover:text-ink")}>
                {v === "model" ? "3D body" : v === "compare" ? "Before / After" : "Body depth map"}
              </button>
            ))}
          </div>
          {result && <span className="text-[11px] text-muted">{result.header.vertices.toLocaleString()} vertices · drag to orbit · wheel to zoom</span>}
        </div>
        <div className="relative flex-1 min-h-0 rounded-lg border border-line bg-panel overflow-hidden">
          {!result && (
            <div className="absolute inset-0 flex items-center justify-center p-4 checker">
              <img src={imageUrl(img.id, 1600)} className="max-w-full max-h-full object-contain" alt="photo" />
            </div>
          )}
          {result && view === "model" && (data ? modelView : <div className="absolute inset-0 flex items-center justify-center text-muted">Loading model…</div>)}
          {result && view === "compare" && (
            <div className="absolute inset-0 grid grid-cols-2">
              <div className="relative border-r border-line flex items-center justify-center p-3 checker">
                <img src={imageUrl(img.id, 1600)} className="max-w-full max-h-full object-contain" alt="before" />
                <span className="absolute left-2 top-2 text-[10px] uppercase tracking-wider bg-black/60 rounded px-1.5 py-0.5">Before · photo</span>
              </div>
              <div className="relative">
                {data && modelView}
                <span className="absolute left-2 top-2 text-[10px] uppercase tracking-wider bg-black/60 rounded px-1.5 py-0.5 pointer-events-none">After · 3D body</span>
              </div>
            </div>
          )}
          {result && view === "depth" && (
            <div className="absolute inset-0 flex items-center justify-center p-4 bg-black">
              <img src={fileUrl(result.depth_preview)} className="max-w-full max-h-full object-contain" alt="body depth" />
            </div>
          )}
          {job && (
            <div className="absolute inset-0 bg-black/50 backdrop-blur-[2px] flex items-center justify-center">
              <div className="w-80 rounded-lg bg-panel border border-line p-4">
                <ProgressBar value={job.p} message={job.msg} />
              </div>
            </div>
          )}
        </div>
      </div>
      <aside className="w-[310px] shrink-0 border-l border-line bg-panel flex flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 border-b border-line space-y-3">
            {depthModel && !depthModel.installed && (
              <Notice kind="warn" title="Depth model not installed">
                <button className="underline" onClick={() => go("models")}>
                  Install Depth Anything V2 in Models
                </button>
                .
              </Notice>
            )}
            <ErrorNotice error={error} onClose={() => setError(null)} />
            {result && <CpuRendererNotice />}
            {result?.method === "segmentation" && <Notice kind="info">Compatibility mode: the person was found with the U2-Net segmenter instead of MediaPipe.</Notice>}
          </div>
          <Section title="Model">
            <Slider label="Detail (vertices across the body)" value={settings.detail} min={80} max={480} step={10} format={(v) => String(v)} onChange={(v) => setSettings((s) => ({ ...s, detail: v }))} />
            <Slider label="Body thickness" value={settings.thickness} min={0.1} max={1.2} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => setSettings((s) => ({ ...s, thickness: v }))} />
            <Slider label="Roundness (silhouette volume)" value={settings.roundness} onChange={(v) => setSettings((s) => ({ ...s, roundness: v }))} hint="0 = depth map only, 1 = volume from the outline only" />
          </Section>
          {result && (
            <Section title="Display">
              <RendererSelect />
              <Select label="Style" value={opts.style} onChange={(v) => o("style", v)} options={[{ value: "wireframe", label: "Wireframe" }, { value: "scanlines", label: "Scan lines" }, { value: "points", label: "Points" }, { value: "solid", label: "Solid" }]} />
              <Select label="Colours" value={opts.colors} onChange={(v) => o("colors", v)} options={[{ value: "depth", label: "Depth" }, { value: "hologram", label: "Hologram" }, { value: "photo", label: "Photo" }]} />
              {(opts.style === "wireframe" || opts.style === "scanlines") && (
                <>
                  <Slider label="Line spacing" value={opts.lineStep} min={1} max={12} step={1} format={(v) => (v === 1 ? "every vertex" : `every ${v}`)} onChange={(v) => o("lineStep", v)} />
                  <Toggle label="Hide lines behind the body" checked={opts.hideHidden} onChange={(v) => o("hideHidden", v)} />
                </>
              )}
              <Slider label="Depth exaggeration" value={opts.relief} min={0.2} max={3} step={0.05} format={(v) => `${v.toFixed(2)}×`} onChange={(v) => o("relief", v)} />
              <Toggle label="Auto-rotate" checked={opts.autoRotate} onChange={(v) => o("autoRotate", v)} />
              <Button size="sm" variant="ghost" onClick={() => viewer.current?.reset()}>
                <RotateCcw className="w-3.5 h-3.5" /> Reset view
              </Button>
              <Stat label="Grid" value={`${result.header.cols} × ${result.header.rows}`} />
            </Section>
          )}
        </div>
        <div className="p-4 border-t border-line space-y-2">
          {!job && (
            <Button variant="primary" size="lg" className="w-full" onClick={build} disabled={!depthModel?.installed}>
              <Sparkles className="w-4 h-4" /> {result ? "Rebuild body model" : "Build 3D body model"}
            </Button>
          )}
          {result && (
            <div className="grid grid-cols-3 gap-2">
              <Button size="sm" onClick={() => save("glb")}>
                <Download className="w-3.5 h-3.5" /> GLB
              </Button>
              <Button size="sm" onClick={() => save("ply")}>
                <Download className="w-3.5 h-3.5" /> PLY
              </Button>
              <Button size="sm" onClick={snapshot}>
                <Download className="w-3.5 h-3.5" /> PNG
              </Button>
            </div>
          )}
          <div className="text-[10.5px] text-faint flex items-center gap-1.5">
            <PersonStanding className="w-3.5 h-3.5" /> The hidden back of the body is not reconstructed: the model shows the visible side in depth.
          </div>
        </div>
      </aside>
    </div>
  );
}
