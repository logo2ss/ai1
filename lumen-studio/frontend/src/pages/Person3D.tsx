import { useRef, useState } from "react";
import { api, ApiError, downloadUrl, ImageMeta, imageUrl, runJob } from "../api";
import { Button, ErrorNotice, Notice, ProgressBar, Slider } from "../components/ui";
import { CpuPersonViewer, PersonFiles, PersonHandle } from "../three/CpuPersonViewer";

interface Result extends PersonFiles { warnings: string[] }
export default function Person3D({ img, onFile }: { img: ImageMeta; onFile: (file: File) => void }) {
  const [result, setResult] = useState<Result | null>(null), [error, setError] = useState<ApiError | null>(null);
  const [job, setJob] = useState<{p: number; msg: string} | null>(null);
  const [thickness, setThickness] = useState(0.65);
  const viewer = useRef<PersonHandle>(null);
  const build = async () => {
    setError(null); setJob({ p: 0, msg: "Starting" });
    try { setResult(await runJob<Result>(api.post("/api/person3d", { image_id: img.id, settings: { thickness } }), j => setJob({ p: j.progress, msg: j.message }))); }
    catch (e) { setError(e as ApiError); } finally { setJob(null); }
  };
  return <div className="h-full flex">
    <div className="flex-1 min-w-0 relative m-4 border border-line rounded-lg overflow-hidden bg-panel">
      {result ? <CpuPersonViewer ref={viewer} files={result} /> : <img src={imageUrl(img.id, 1400)} alt="Person photo" className="w-full h-full object-contain" />}
      {job && <div className="absolute inset-0 bg-black/60 flex items-center justify-center"><div className="w-80 p-4 bg-panel rounded"><ProgressBar value={job.p} message={job.msg} /></div></div>}
      {result && <div className="absolute top-3 left-3 flex gap-2" data-no-drag>{[-50, 0, 50].map(d => <Button key={d} size="sm" onClick={() => viewer.current?.angle(d)}>{d > 0 ? "+" : ""}{d}°</Button>)}<Button size="sm" onClick={() => viewer.current?.reset()}>Reset</Button></div>}
      {result && <div className="absolute bottom-3 left-3 text-xs text-white/80 pointer-events-none">Drag: orbit ±70° · Right-drag: pan · Wheel: zoom · WASD: move · R: reset</div>}
    </div>
    <aside className="w-[310px] border-l border-line bg-panel p-4 space-y-4 overflow-y-auto">
      <h2 className="font-semibold">Person 3D</h2>
      <p className="text-xs text-muted">Isolate the person and explore a rounded volume on an empty background. A clear, near-frontal photo gives the best sides. CPU rendering works without WebGL.</p>
      <label className="text-xs text-accent cursor-pointer block">Load another photo…<input type="file" accept="image/*" className="hidden" onChange={e => e.target.files?.[0] && onFile(e.target.files[0])} /></label>
      <Slider label="Body roundness" value={thickness} min={0.2} max={1} step={0.05} onChange={setThickness} />
      <ErrorNotice error={error} onClose={() => setError(null)} />
      <Button variant="primary" disabled={!!job} onClick={build}>{result ? "Rebuild person" : "Build Person 3D"}</Button>
      {result?.warnings.map((w, i) => <Notice key={i} kind="info">{w}</Notice>)}
      {result && <Button onClick={() => { const url = viewer.current?.screenshot(); if (url) downloadUrl(url, "person-3d.png"); }}>Save view</Button>}
    </aside>
  </div>;
}
