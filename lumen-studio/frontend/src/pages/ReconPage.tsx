import { useEffect, useRef, useState } from "react";
import { Box, Camera, Download, Footprints, Orbit, RotateCcw, Sparkles } from "lucide-react";
import { api, ApiError, fileUrl, imageUrl, ImageMeta, runJob, uploadImage, downloadUrl, withToken } from "../api";
import { useApp, useModel } from "../state";
import { Button, cx, Dropzone, ErrorNotice, Kbd, Notice, OverlayCompare, ProgressBar, Section, Select, Slider, Stat, Toggle } from "../components/ui";
import { SceneViewer, ViewerHandle, ViewerOptions } from "../three/SceneViewer";
import { CpuSceneViewer } from "../three/CpuSceneViewer";
import { parseScene, ParsedScene } from "../three/scene";
import { useRenderer } from "../three/webgl";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { CpuRendererNotice, RendererSelect } from "../components/RendererSelect";
import BodyEditor from "./BodyEditor";
import Body3D from "./Body3D";
import Person3D from "./Person3D";

interface ReconResult {
  scene: string;
  main_texture: string;
  bg_texture: string;
  back_texture?: string;
  depth_png: string;
  depth_preview: string;
  header: any;
  report: any;
}

const RECON_DEFAULTS = {
  model: "depth-anything-v2-small",
  quality: "balanced" as "fast" | "balanced" | "high",
  detail_fusion: true,
  occlusion_width: 0.05,
  edge_threshold: 0.045,
  outpaint_margin: 0.25,
  detect_sky: true,
};

export default function ReconPage() {
  const { go } = useApp();
  const da = useModel("depth-anything-v2-small");
  const midas = useModel("midas-v21-small");
  const [img, setImg] = useState<ImageMeta | null>(null);
  const [rs, setRs] = useState(RECON_DEFAULTS);
  const [job, setJob] = useState<{ p: number; msg: string } | null>(null);
  const [result, setResult] = useState<ReconResult | null>(null);
  const [scene, setScene] = useState<ParsedScene | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [view, setView] = useState<"3d" | "compare" | "photo" | "depth">("3d");
  const [vo, setVo] = useState<ViewerOptions>({ fov: 60, zfar: 8, mode: "color", pointCloud: false, showBackground: true, constrain: true, controls: "orbit", motion: "none", marginFade: 0.35 });
  const [status, setStatus] = useState({ outside: false, distance: 0 });
  const viewer = useRef<ViewerHandle>(null);
  const [tool, setTool] = useState<"3d" | "body" | "body3d" | "person3d">("3d");
  const rend = useRenderer();
  const setV = <K extends keyof ViewerOptions>(k: K, v: ViewerOptions[K]) => setVo((o) => ({ ...o, [k]: v }));

  useEffect(() => {
    if (!result) return;
    let off = false;
    fetch(fileUrl(result.scene))
      .then((r) => r.arrayBuffer())
      .then((b) => !off && setScene(parseScene(b)))
      .catch((e) => setError(new ApiError("scene", "Could not load the reconstructed scene: " + e.message)));
    return () => {
      off = true;
    };
  }, [result]);

  const load = async (f: File) => {
    setError(null);
    try {
      const m = await uploadImage(f);
      setImg(m);
      setResult(null);
      setScene(null);
    } catch (e) {
      setError(e as ApiError);
    }
  };
  const run = async () => {
    if (!img) return;
    setError(null);
    setJob({ p: 0, msg: "Starting" });
    try {
      const r = await runJob<ReconResult>(api.post("/api/recon", { image_id: img.id, settings: rs }), (j) => setJob({ p: j.progress, msg: j.message }));
      setScene(null);
      setResult(r);
      setVo((o) => ({ ...o, fov: r.header.default_fov, zfar: r.header.default_zfar }));
      setView("3d");
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setJob(null);
    }
  };
  const save = async (kind: "glb" | "ply") => {
    if (!viewer.current) return;
    try {
      const blob = kind === "glb" ? await viewer.current.exportGLB() : await viewer.current.exportPLY();
      const url = URL.createObjectURL(blob);
      downloadUrl(url, `${(img?.name || "scene").replace(/\.[^.]+$/, "")}-3d.${kind}`);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e: any) {
      setError(new ApiError("export", `Could not export the ${kind.toUpperCase()} file: ${e?.message || e}`));
    }
  };
  const snapshot = () => {
    const url = viewer.current?.screenshot();
    if (url) downloadUrl(url, "view.jpg");
  };
  const modelOk = rs.model === "midas-v21-small" ? midas?.installed : da?.installed;

  // The 3D view: GPU (WebGL) viewer, or frames rendered by the engine when the browser has no
  // WebGL. A GPU failure switches to the CPU renderer; any other failure stays inside the box.
  const viewer3d = result && scene && (
    <ErrorBoundary key={rend.kind} fill label="3D viewer" onError={rend.kind === "gpu" ? (e) => rend.reportFailure(e.message) : undefined}>
      {rend.kind === "gpu" ? (
        <SceneViewer
          ref={viewer}
          scene={scene}
          files={{ scene: result.scene, main_texture: result.main_texture, bg_texture: result.bg_texture }}
          mainTexture={fileUrl(result.main_texture)}
          bgTexture={fileUrl(result.bg_texture)}
          backTexture={result.back_texture ? fileUrl(result.back_texture) : undefined}
          options={vo}
          onStatus={setStatus}
          onRendererError={(e) => rend.reportFailure(e.message)}
          onContextLost={() => rend.reportFailure("the browser lost the WebGL context")}
        />
      ) : (
        <CpuSceneViewer
          ref={viewer}
          scene={scene}
          files={{ scene: result.scene, main_texture: result.main_texture, bg_texture: result.bg_texture, back_texture: result.back_texture ?? null }}
          mainTexture={fileUrl(result.main_texture)}
          bgTexture={fileUrl(result.bg_texture)}
          backTexture={result.back_texture ? fileUrl(result.back_texture) : undefined}
          options={vo}
          onStatus={setStatus}
        />
      )}
    </ErrorBoundary>
  );

  return (
    <div className="h-full flex">
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 flex items-center gap-3 px-5 border-b border-line">
          <Box className="w-5 h-5 text-accent" />
          <h1 className="font-semibold">3D Reconstruction</h1>
          <div className="flex rounded-md bg-panel border border-line p-0.5 ml-3" data-testid="recon-tool">
            {(["3d", "person3d", "body3d", "body"] as const).map((t) => (
              <button key={t} onClick={() => setTool(t)} className={cx("px-3 h-7 rounded text-[12px]", tool === t ? "bg-accent/20 text-ink ring-1 ring-accent/40" : "text-muted hover:text-ink")}>
                {t === "3d" ? "3D scene" : t === "person3d" ? "Person 3D" : t === "body3d" ? "Body 3D" : "Body proportions"}
              </button>
            ))}
          </div>
          <div className="flex-1" />
          {result && tool === "3d" && (
            <div className="flex rounded-md bg-panel border border-line p-0.5">
              {(["3d", "compare", "photo", "depth"] as const).map((v) => (
                <button key={v} onClick={() => setView(v)} className={cx("px-3 h-7 rounded text-[12px]", view === v ? "bg-panel-2 text-ink border border-line-2" : "text-muted hover:text-ink")}>
                  {v === "3d" ? "3D scene" : v === "compare" ? "Before / After" : v === "photo" ? "Photo" : "Depth map"}
                </button>
              ))}
            </div>
          )}
        </header>
        {tool === "person3d" && img && <div className="flex-1 min-h-0"><Person3D key={img.id} img={img} onFile={load} /></div>}
        {tool === "body" && img && (
          <div className="flex-1 min-h-0">
            <BodyEditor img={img} />
          </div>
        )}
        {tool === "body3d" && img && (
          <div className="flex-1 min-h-0">
            <Body3D img={img} />
          </div>
        )}
        <div className={cx("flex-1 min-h-0 p-4", tool !== "3d" && img && "hidden")}>
          <div className="relative w-full h-full rounded-lg border border-line bg-panel overflow-hidden">
            {!img && (
              <div className="absolute inset-0 p-4">
                <Dropzone onFile={load} label="Load a photograph" sub="Indoor, outdoor, landscapes, buildings, objects — any ordinary photo" />
              </div>
            )}
            {img && (!result || view === "photo") && (
              <div className="absolute inset-0 flex items-center justify-center p-4 checker">
                <img src={imageUrl(img.id, 2000)} className="max-w-full max-h-full object-contain" alt="input" />
              </div>
            )}
            {result && view === "depth" && (
              <div className="absolute inset-0 flex items-center justify-center p-4 bg-black">
                <img src={fileUrl(result.depth_preview)} className="max-w-full max-h-full object-contain" alt="depth" />
              </div>
            )}
            {result && scene && view === "compare" && (
              <OverlayCompare before={imageUrl(img!.id, 2400)} labels={["Before · photo", "After · 3D (drag the 3D side)"]}>
                {viewer3d}
              </OverlayCompare>
            )}
            {result && scene && view === "3d" && (
              <>
                {viewer3d}
                <div className="absolute left-3 bottom-3 text-[11px] text-white/80 bg-black/55 backdrop-blur rounded-md px-2.5 py-1.5 pointer-events-none space-x-2" data-testid="nav-hint">
                  {vo.controls === "orbit" ? (
                    <span>Drag: orbit · Right-drag/Shift: pan · Wheel: move in/out</span>
                  ) : (
                    <span>Drag: look around · Wheel: forward/back</span>
                  )}
                  <span>
                    <Kbd>W</Kbd>
                    <Kbd>A</Kbd>
                    <Kbd>S</Kbd>
                    <Kbd>D</Kbd>/arrows move · <Kbd>Q</Kbd>
                    <Kbd>E</Kbd> down/up · <Kbd>Shift</Kbd> faster · <Kbd>R</Kbd> reset
                  </span>
                </div>
                <div className="absolute right-3 top-3 flex gap-1.5" data-no-drag>
                  <button
                    type="button"
                    data-testid="free-roam"
                    aria-pressed={!vo.constrain}
                    title="Free roam: move anywhere (beyond the reliable range the surroundings are a soft guess)"
                    onClick={() => setVo((o) => (o.constrain ? { ...o, constrain: false, controls: "walk" } : { ...o, constrain: true, controls: "orbit" }))}
                    className={cx("h-7 px-2.5 rounded-md text-[11px] backdrop-blur border", !vo.constrain ? "bg-accent/30 border-accent/60 text-ink" : "bg-black/55 border-white/10 text-white/80 hover:text-white")}
                  >
                    <Footprints className="w-3.5 h-3.5 inline -mt-0.5 mr-1" />
                    {vo.constrain ? "Safe range" : "Free roam"}
                  </button>
                  <button type="button" data-testid="reset-view" title="Back to the original viewpoint (R)" onClick={() => viewer.current?.reset()} className="h-7 px-2 rounded-md text-[11px] bg-black/55 backdrop-blur border border-white/10 text-white/80 hover:text-white">
                    <RotateCcw className="w-3.5 h-3.5 inline -mt-0.5" />
                  </button>
                </div>
                {vo.constrain && status.outside && (
                  <div className="absolute top-3 left-1/2 -translate-x-1/2 text-[11px] bg-warn/15 border border-warn/30 text-warn rounded-full px-3 py-1 pointer-events-none">
                    Edge of the reliable viewing range — beyond this, hidden areas are mostly guesswork
                  </div>
                )}
              </>
            )}
            {result && !scene && view === "3d" && <div className="absolute inset-0 flex items-center justify-center text-muted">Loading scene…</div>}
            {job && (
              <div className="absolute inset-0 bg-black/50 backdrop-blur-[2px] flex items-center justify-center">
                <div className="w-80 rounded-lg bg-panel border border-line p-4">
                  <ProgressBar value={job.p} message={job.msg} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      <aside className={cx("w-[310px] shrink-0 border-l border-line bg-panel flex flex-col", tool !== "3d" && img && "hidden")}>
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 border-b border-line space-y-3">
            {!modelOk && (
              <Notice kind="warn" title="Depth model not installed">
                3D reconstruction needs a local depth model.{" "}
                <button className="underline" onClick={() => go("models")}>
                  Install it in Models
                </button>
                .
              </Notice>
            )}
            <ErrorNotice error={error} onClose={() => setError(null)} />
            {img && <CpuRendererNotice />}
            {img && (
              <label className="text-[11px] text-accent cursor-pointer hover:underline">
                Load another photo…
                <input type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && load(e.target.files[0])} />
              </label>
            )}
          </div>
          <Section title="Reconstruction" defaultOpen={!result}>
            <Select
              label="Depth model"
              value={rs.model as any}
              onChange={(v: string) => setRs((x) => ({ ...x, model: v }))}
              options={[
                { value: "depth-anything-v2-small", label: "Depth Anything V2" },
                { value: "midas-v21-small", label: "MiDaS (fast)" },
              ]}
            />
            <Select label="Quality" value={rs.quality} onChange={(v) => setRs((x) => ({ ...x, quality: v }))} options={[{ value: "fast", label: "Fast" }, { value: "balanced", label: "Balanced" }, { value: "high", label: "High" }]} />
            <Toggle label="Multi-crop depth detail fusion" checked={rs.detail_fusion} onChange={(v) => setRs((x) => ({ ...x, detail_fusion: v }))} />
            <Toggle label="Detect sky (far dome)" checked={rs.detect_sky} onChange={(v) => setRs((x) => ({ ...x, detect_sky: v }))} />
            <Slider label="Hidden background to synthesise" value={rs.occlusion_width} min={0.01} max={0.12} step={0.005} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => setRs((x) => ({ ...x, occlusion_width: v }))} hint="How far behind foreground edges the occluded background is estimated." />
            <Slider label="Depth edge sensitivity" value={rs.edge_threshold} min={0.02} max={0.12} step={0.005} format={(v) => v.toFixed(3)} onChange={(v) => setRs((x) => ({ ...x, edge_threshold: v }))} hint="Lower = tear the mesh at smaller depth jumps." />
            <Slider label="Out-of-frame extension" value={rs.outpaint_margin} min={0} max={0.4} step={0.01} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => setRs((x) => ({ ...x, outpaint_margin: v }))} />
          </Section>
          {result && (
            <>
              <Section title="Navigation">
                <Select label="Controls" value={vo.controls} onChange={(v) => setV("controls", v)} options={[{ value: "orbit", label: "Orbit" }, { value: "walk", label: "Walk / look" }]} />
                <Select label="Camera motion" value={vo.motion} onChange={(v) => setV("motion", v)} options={[{ value: "none", label: "Manual" }, { value: "sway", label: "Sway" }, { value: "dolly", label: "Dolly" }, { value: "circle", label: "Circle" }]} />
                <Toggle label="Free roam (walk anywhere, no limits)" checked={!vo.constrain} onChange={(v) => setVo((o) => ({ ...o, constrain: !v, controls: v ? "walk" : "orbit" }))} hint="Off: stay within the reliable viewing range near the original camera. On: walk anywhere — beyond the photo the surroundings are a soft, estimated environment." />
                <div className="flex gap-2">
                  <Button size="sm" onClick={() => viewer.current?.reset()}>
                    <RotateCcw className="w-3.5 h-3.5" /> Reset view
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setV("controls", vo.controls === "orbit" ? "walk" : "orbit")}>
                    {vo.controls === "orbit" ? <Footprints className="w-3.5 h-3.5" /> : <Orbit className="w-3.5 h-3.5" />} {vo.controls === "orbit" ? "Walk mode" : "Orbit mode"}
                  </Button>
                </div>
              </Section>
              <Section title="Geometry & display">
                <RendererSelect />
                <Slider label="Field of view (original camera)" value={vo.fov} min={30} max={100} step={1} format={(v) => `${v}°`} onChange={(v) => setV("fov", v)} />
                <Slider label="Depth range" value={vo.zfar} min={2} max={30} step={0.5} format={(v) => `${v.toFixed(1)}×`} onChange={(v) => setV("zfar", v)} hint="Ratio between the farthest and nearest surfaces." />
                <Select label="Shading" value={vo.mode} onChange={(v) => setV("mode", v)} options={[{ value: "color", label: "Photo" }, { value: "depth", label: "Depth" }, { value: "layers", label: "Layers" }]} />
                <Toggle
                  label="Point cloud"
                  checked={vo.pointCloud && rend.kind === "gpu"}
                  disabled={rend.kind !== "gpu"}
                  hint={rend.kind !== "gpu" ? "Point cloud display needs the GPU renderer (WebGL)." : undefined}
                  onChange={(v) => setV("pointCloud", v)}
                />
                <Toggle label="Show synthesised background layer" checked={vo.showBackground} onChange={(v) => setV("showBackground", v)} />
                <Slider label="Fade estimated periphery" value={vo.marginFade} onChange={(v) => setV("marginFade", v)} />
              </Section>
              <Section title="Reconstruction report">
                <Stat label="Visible surface triangles" value={result.report.main_triangles.toLocaleString()} />
                <Stat label="Synthesised background triangles" value={result.report.background_triangles.toLocaleString()} />
                <Stat label="Occluded area estimated" value={`${(result.report.occluded_fraction * 100).toFixed(1)}%`} />
                <Stat label="Sky" value={`${(result.report.sky_fraction * 100).toFixed(1)}%`} />
                {result.report.warnings.map((w: string, i: number) => (
                  <Notice key={i} kind="warn">
                    {w}
                  </Notice>
                ))}
                <Notice kind="info">
                  A single photo only shows one side of the world. Surfaces hidden behind objects and outside the frame are <b>estimated</b> (pink in “Layers” view) — they are plausible, not recovered.
                </Notice>
              </Section>
            </>
          )}
        </div>
        <div className="p-4 border-t border-line space-y-2">
          {!job && (
            <Button variant="primary" size="lg" className="w-full" disabled={!img || !modelOk} onClick={run}>
              <Sparkles className="w-4 h-4" /> {result ? "Reconstruct again" : "Reconstruct 3D scene"}
            </Button>
          )}
          {result && (
            <div className="grid grid-cols-2 gap-2">
              <Button size="sm" onClick={() => save("glb")} title="Textured mesh (glTF binary)">
                <Download className="w-3.5 h-3.5" /> GLB mesh
              </Button>
              <Button size="sm" onClick={() => save("ply")} title="Coloured point cloud">
                <Download className="w-3.5 h-3.5" /> PLY points
              </Button>
              <Button size="sm" onClick={() => downloadUrl(withToken(`/api/files/${result.depth_png}?download=1`))} title="16-bit relative depth (disparity)">
                <Download className="w-3.5 h-3.5" /> Depth PNG
              </Button>
              <Button size="sm" onClick={snapshot}>
                <Camera className="w-3.5 h-3.5" /> Snapshot
              </Button>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
