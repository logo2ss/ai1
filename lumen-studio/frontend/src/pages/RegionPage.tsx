import { useRef, useState } from "react";
import { ArrowLeftRight, Check, Download, RotateCcw, Shapes } from "lucide-react";
import { api, ApiError, imageUrl, ImageMeta, runJob, uploadImage, withToken, downloadUrl } from "../api";
import { MaskEditor, MaskEditorHandle } from "../components/MaskEditor";
import { Button, CompareSlider, cx, Dropzone, ErrorNotice, Notice, ProgressBar, Section, Select, Slider, Stat, Toggle } from "../components/ui";

// Everything technical is decided by the engine ("auto"); these only override it.
const ADVANCED = {
  mode: "auto" as "auto" | "object" | "area",
  fit: "auto" as "auto" | "contain" | "cover" | "stretch",
  anchor: "auto" as "auto" | "center" | "bottom",
  blend_mode: "auto" as "auto" | "matte" | "poisson" | "multiband",
  color_harmony: 0.8,
  lighting_match: 0.6,
  edge_softness: 0.4,
  light_wrap: 0.35,
  contact_shadow: -1, // -1 = automatic
  remove_original: true,
  match_grain: true,
  match_sharpness: true,
  scale: 1,
  rotation: 0,
  offset_x: 0,
  offset_y: 0,
  flip: false,
};

type Result = { image_id: string; report: any };
type Results = { a?: Result; b?: Result };

const MODE_LABEL: Record<string, string> = { object: "Object (cut out)", area: "Area / texture" };

export default function RegionPage() {
  const [a, setA] = useState<ImageMeta | null>(null);
  const [b, setB] = useState<ImageMeta | null>(null);
  const [hasA, setHasA] = useState(false);
  const [hasB, setHasB] = useState(false);
  const edA = useRef<MaskEditorHandle>(null);
  const edB = useRef<MaskEditorHandle>(null);
  const [adv, setAdv] = useState(ADVANCED);
  const [direction, setDirection] = useState<"both" | "b_to_a" | "a_to_b">("both");
  const [job, setJob] = useState<{ p: number; msg: string } | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [tab, setTab] = useState<"edit" | "results">("edit");
  const [error, setError] = useState<ApiError | null>(null);
  const [fmt, setFmt] = useState<"png" | "jpg" | "tiff">("png");
  const set = <K extends keyof typeof ADVANCED>(k: K, v: (typeof ADVANCED)[K]) => setAdv((x) => ({ ...x, [k]: v }));

  const load = async (f: File, which: "a" | "b") => {
    setError(null);
    try {
      const m = await uploadImage(f);
      if (which === "a") {
        setA(m);
        setHasA(false);
      } else {
        setB(m);
        setHasB(false);
      }
      setResults(null);
      setTab("edit");
    } catch (e) {
      setError(e as ApiError);
    }
  };

  const run = async () => {
    if (!a || !b) return;
    setError(null);
    setJob({ p: 0, msg: "Preparing selections" });
    try {
      const [ma, mb] = await Promise.all([edA.current?.getMaskBlob(), edB.current?.getMaskBlob()]);
      if (!ma || !mb) {
        setJob(null);
        setError(new ApiError("invalid_input", `Select an area in ${!ma && !mb ? "both images" : !ma ? "image A" : "image B"} first.`, "Draw around or over the part you want, or click points around it."));
        return;
      }
      const { contact_shadow, ...rest } = adv;
      // the editor already snapped the outlines to the object edges (what you see is what is swapped)
      const settings = { ...rest, snap_edges: false, ...(contact_shadow >= 0 ? { contact_shadow } : {}) };
      const fd = new FormData();
      fd.append("payload", JSON.stringify({ image_a: a.id, image_b: b.id, settings, direction }));
      fd.append("mask_a", ma, "a.png");
      fd.append("mask_b", mb, "b.png");
      const r = await runJob<Results>(api.form("/api/region/swap", fd), (j) => setJob({ p: j.progress, msg: j.message }));
      setResults(r);
      setTab("results");
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setJob(null);
    }
  };

  const steps: { label: string; done: boolean }[] = [
    { label: "Image A: click dots around the area", done: !!a && hasA },
    { label: "Image B: click dots around the area", done: !!b && hasB },
    { label: "Press Swap", done: !!results },
  ];
  const ready = !!a && !!b && hasA && hasB;
  const why = !a || !b ? `Load image ${!a && !b ? "A and B" : !a ? "A" : "B"} first` : !hasA && !hasB ? "Select an area in A and B first" : !hasA ? "Select an area in image A first" : !hasB ? "Select an area in image B first" : "";

  return (
    <div className="h-full flex">
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 flex items-center gap-3 px-5 border-b border-line">
          <Shapes className="w-5 h-5 text-accent" />
          <h1 className="font-semibold">Region Swapping</h1>
          <span className="text-muted text-[12px] hidden lg:inline truncate">Click dots around the area you want in each image, then press Swap.</span>
          <div className="flex-1" />
          {results && (
            <div className="flex rounded-md bg-panel border border-line p-0.5">
              {(["edit", "results"] as const).map((v) => (
                <button key={v} onClick={() => setTab(v)} className={cx("px-3 h-7 rounded text-[12px]", tab === v ? "bg-panel-2 text-ink border border-line-2" : "text-muted hover:text-ink")}>
                  {v === "edit" ? "Edit selections" : "Results"}
                </button>
              ))}
            </div>
          )}
        </header>
        <div className="flex-1 min-h-0 p-4">
          <div className={cx("grid grid-cols-2 gap-4 h-full", tab !== "edit" && "hidden")}>
            {(["a", "b"] as const).map((k) => {
              const meta = k === "a" ? a : b;
              const has = k === "a" ? hasA : hasB;
              const K = k.toUpperCase();
              return meta ? (
                <div key={k + meta.id} className="min-h-0 flex flex-col gap-2">
                  <div className="flex items-center gap-2 text-[12px] h-5">
                    <span className={cx("w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold", has ? "bg-ok/20 text-ok" : "bg-panel-2 text-muted border border-line-2")}>
                      {has ? <Check className="w-3 h-3" /> : k === "a" ? 1 : 2}
                    </span>
                    <span className={has ? "text-ok" : "text-ink font-medium"}>{has ? `Image ${K}: area selected` : `Image ${K}: click dots around the area`}</span>
                    <div className="flex-1" />
                    <label className="text-[11px] text-muted cursor-pointer hover:text-ink">
                      Replace image…
                      <input type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && load(e.target.files[0], k)} />
                    </label>
                  </div>
                  <div className="flex-1 min-h-0">
                    <MaskEditor
                      ref={k === "a" ? edA : edB}
                      imageId={meta.id}
                      width={meta.width}
                      height={meta.height}
                      label={`IMAGE ${K}`}
                      color={k === "a" ? "#8f83ff" : "#38c4d9"}
                      onChange={k === "a" ? setHasA : setHasB}
                      onError={setError}
                    />
                  </div>
                </div>
              ) : (
                <div key={k} className="rounded-lg border border-line bg-panel p-3">
                  <Dropzone onFile={(f) => load(f, k)} label={`${k === "a" ? 1 : 2}. Load image ${K}`} sub="Drop an image here or click to browse — then click dots around the area to swap" />
                </div>
              );
            })}
          </div>
          {tab === "results" && results && (
            <div className="grid grid-cols-2 gap-4 h-full">
              {(["a", "b"] as const).map((k) => {
                const r = results[k];
                const meta = k === "a" ? a : b;
                if (!r || !meta) return <div key={k} className="rounded-lg border border-line bg-panel flex items-center justify-center text-muted">Not generated (direction)</div>;
                return (
                  <div key={k} className="flex flex-col min-h-0 rounded-lg border border-line bg-panel overflow-hidden" data-testid={`result-${k}`}>
                    <div className="h-10 flex items-center px-3 border-b border-line gap-2">
                      <span className="font-medium text-[13px]">Image {k.toUpperCase()}</span>
                      <span className="text-faint text-[11px] truncate">with the area from image {k === "a" ? "B" : "A"}</span>
                      <div className="flex-1" />
                      <Button size="sm" onClick={() => downloadUrl(withToken(`/api/images/${r.image_id}/export?format=${fmt}&quality=96`))}>
                        <Download className="w-3.5 h-3.5" /> Export {fmt.toUpperCase()}
                      </Button>
                    </div>
                    <div className="flex-1 min-h-0 p-3 checker">
                      <CompareSlider before={imageUrl(meta.id, 2400)} after={imageUrl(r.image_id, 2400)} labels={["Before", "After"]} />
                    </div>
                    {r.report.warnings?.length > 0 && (
                      <div className="p-2 border-t border-line space-y-1">
                        {r.report.warnings.map((w: string, i: number) => (
                          <Notice key={i} kind="warn">
                            {w}
                          </Notice>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <aside className="w-[300px] shrink-0 border-l border-line bg-panel flex flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 border-b border-line space-y-3">
            <ol className="space-y-1.5" data-testid="region-steps">
              {steps.map((s, i) => (
                <li key={i} className={cx("flex items-center gap-2 text-[12.5px]", s.done ? "text-ok" : "text-muted")}>
                  <span className={cx("w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0", s.done ? "bg-ok/20" : "bg-panel-2 border border-line-2")}>
                    {s.done ? <Check className="w-3 h-3" /> : i + 1}
                  </span>
                  {s.label}
                </li>
              ))}
            </ol>
            <div className="text-[11.5px] text-muted leading-relaxed rounded-md bg-panel-2/60 border border-line px-3 py-2">
              Click <b className="text-ink font-medium">dots</b> around the area you want — the inside is selected (shown tinted). Drag a dot to move it; right-click or Backspace removes the last dot. Prefer drawing? Use <b className="text-ink font-medium">Draw</b> and draw around or over the area. Everything else is automatic.
            </div>
            <ErrorNotice error={error} onClose={() => setError(null)} />
          </div>
          <div className="p-4 border-b border-line space-y-3">
            {job ? (
              <ProgressBar value={job.p} message={job.msg} />
            ) : (
              <Button variant="primary" size="lg" className="w-full" disabled={!ready} onClick={run} title={ready ? "Swap the selected areas" : why}>
                {ready ? (
                  <>
                    <ArrowLeftRight className="w-4 h-4" /> Swap
                  </>
                ) : (
                  <span className="text-[12.5px]">{why}</span>
                )}
              </Button>
            )}
            {results && <Select label="Export format" value={fmt} onChange={setFmt} options={[{ value: "png", label: "PNG" }, { value: "jpg", label: "JPEG" }, { value: "tiff", label: "TIFF" }]} />}
          </div>
          {results && (
            <Section title="What was done">
              {(["a", "b"] as const).map(
                (k) =>
                  results[k] && (
                    <div key={k} className="space-y-0.5">
                      <div className="text-[11px] text-faint uppercase">Into image {k.toUpperCase()}</div>
                      <Stat label="Content" value={MODE_LABEL[results[k]!.report.mode] ?? results[k]!.report.mode} />
                      <Stat label="Fit" value={results[k]!.report.fit} />
                      <Stat label="Scale" value={`${results[k]!.report.geometry.scale}×`} />
                      <Stat label="Blend" value={results[k]!.report.blend_mode} />
                      <Stat label="Old content removed" value={results[k]!.report.inpainting} />
                    </div>
                  ),
              )}
            </Section>
          )}
          <Section title="Advanced" defaultOpen={false}>
            <div className="text-[11px] text-faint -mt-1">Everything is chosen automatically; change these only if the result needs it.</div>
            <Select label="Swap direction" value={direction} onChange={setDirection} options={[{ value: "both", label: "Both ways" }, { value: "b_to_a", label: "B into A" }, { value: "a_to_b", label: "A into B" }]} />
            <Select label="Content" value={adv.mode} onChange={(v) => set("mode", v)} options={[{ value: "auto", label: "Auto" }, { value: "object", label: "Object" }, { value: "area", label: "Area/texture" }]} />
            <Select label="Fit" value={adv.fit} onChange={(v) => set("fit", v)} options={[{ value: "auto", label: "Auto" }, { value: "contain", label: "Contain" }, { value: "cover", label: "Cover" }, { value: "stretch", label: "Stretch" }]} />
            <Select label="Position" value={adv.anchor} onChange={(v) => set("anchor", v)} options={[{ value: "auto", label: "Auto" }, { value: "center", label: "Centre" }, { value: "bottom", label: "On the ground" }]} />
            <Select label="Blend" value={adv.blend_mode} onChange={(v) => set("blend_mode", v)} options={[{ value: "auto", label: "Auto" }, { value: "matte", label: "Matte" }, { value: "poisson", label: "Poisson" }, { value: "multiband", label: "Multiband" }]} />
            <Slider label="Colour & exposure harmony" value={adv.color_harmony} onChange={(v) => set("color_harmony", v)} />
            <Slider label="Lighting gradient match" value={adv.lighting_match} onChange={(v) => set("lighting_match", v)} />
            <Slider label="Edge softness" value={adv.edge_softness} onChange={(v) => set("edge_softness", v)} />
            <Slider label="Light wrap" value={adv.light_wrap} onChange={(v) => set("light_wrap", v)} />
            <Slider label="Contact shadow" value={adv.contact_shadow < 0 ? 0 : adv.contact_shadow} format={(v) => (adv.contact_shadow < 0 ? "auto" : `${Math.round(v * 100)}%`)} onChange={(v) => set("contact_shadow", v)} />
            <Toggle label="Remove what was there before" checked={adv.remove_original} onChange={(v) => set("remove_original", v)} />
            <Toggle label="Match grain" checked={adv.match_grain} onChange={(v) => set("match_grain", v)} />
            <Toggle label="Match sharpness" checked={adv.match_sharpness} onChange={(v) => set("match_sharpness", v)} />
            <Slider label="Scale" value={adv.scale} min={0.2} max={3} step={0.01} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set("scale", v)} />
            <Slider label="Rotation" value={adv.rotation} min={-180} max={180} step={1} format={(v) => `${v}°`} onChange={(v) => set("rotation", v)} />
            <Slider label="Offset X" value={adv.offset_x} min={-1} max={1} step={0.01} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set("offset_x", v)} />
            <Slider label="Offset Y" value={adv.offset_y} min={-1} max={1} step={0.01} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set("offset_y", v)} />
            <Toggle label="Mirror horizontally" checked={adv.flip} onChange={(v) => set("flip", v)} />
            <Button size="sm" variant="ghost" onClick={() => setAdv(ADVANCED)}>
              <RotateCcw className="w-3.5 h-3.5" /> Reset to automatic
            </Button>
          </Section>
        </div>
      </aside>
    </div>
  );
}
