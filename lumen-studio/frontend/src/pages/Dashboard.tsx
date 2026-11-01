import { ArrowRight, Box, Cpu, HardDrive, Package, ScanFace, Shapes, ShieldCheck } from "lucide-react";
import { fmtBytes, useApp } from "../state";
import { Notice } from "../components/ui";

const FEATURES = [
  {
    id: "face" as const,
    icon: ScanFace,
    title: "Face Replacement",
    text: "Dense 478-point landmark alignment, skin-aware masks, illumination transfer, colour, grain and sharpness matching with gradient-domain blending.",
    tags: ["Landmarks", "Relighting", "Poisson", "Multiband"],
    grad: "from-[#8f83ff]/25",
  },
  {
    id: "region" as const,
    icon: Shapes,
    title: "Region Swapping",
    text: "Exchange any freehand selection between two photos. Edge-snapped mattes, background inpainting and environment-aware colour & light adaptation.",
    tags: ["Lasso & brush", "Matting", "Inpainting", "Harmonisation"],
    grad: "from-[#4fb6d1]/25",
  },
  {
    id: "recon" as const,
    icon: Box,
    title: "3D Reconstruction",
    text: "Turn one photo into an explorable layered-depth scene: monocular depth, torn depth discontinuities and synthesised occluded background.",
    tags: ["Depth Anything V2", "Layered mesh", "Occlusion fill", "Three.js"],
    grad: "from-[#e39a5b]/25",
  },
];

export default function Dashboard() {
  const { go, system, models } = useApp();
  const installed = models.filter((m) => m.installed).length;
  const missing = models.filter((m) => m.required && !m.installed);
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="mb-8">
          <h1 className="text-[26px] font-semibold tracking-tight">Welcome to Lumen Studio</h1>
          <p className="text-muted mt-1.5 max-w-2xl">
            Professional, physically-informed image editing that runs entirely on this computer. Nothing is uploaded — every pixel is analysed and processed locally.
          </p>
        </div>
        {missing.length > 0 && (
          <div className="mb-6">
            <Notice kind="warn" title={`${missing.length} required model${missing.length > 1 ? "s are" : " is"} not installed`}>
              {missing.map((m) => m.name).join(", ")}. Install {missing.length > 1 ? "them" : "it"} once from{" "}
              <button className="underline" onClick={() => go("models")}>
                Models
              </button>
              ; afterwards everything works offline.
            </Notice>
          </div>
        )}
        {system?.mediapipe && !system.mediapipe.usable && (
          <div className="mb-6">
            <Notice kind="info" title="Running in compatibility mode">
              {system.mediapipe.reason} Face replacement uses the built-in OpenCV face detector instead; the body editor and one-click smart select are unavailable. {system.mediapipe.hint}
            </Notice>
          </div>
        )}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => (
            <button
              key={f.id}
              onClick={() => go(f.id)}
              data-testid={`card-${f.id}`}
              className={`group text-left rounded-xl border border-line bg-gradient-to-br ${f.grad} to-panel p-5 hover:border-line-2 transition relative overflow-hidden`}
            >
              <div className="w-10 h-10 rounded-lg bg-black/30 border border-white/10 flex items-center justify-center mb-4">
                <f.icon className="w-5 h-5" />
              </div>
              <div className="text-[15px] font-semibold mb-1.5 flex items-center gap-2">
                {f.title} <ArrowRight className="w-4 h-4 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition" />
              </div>
              <p className="text-muted text-[12.5px] leading-relaxed">{f.text}</p>
              <div className="flex flex-wrap gap-1.5 mt-4">
                {f.tags.map((t) => (
                  <span key={t} className="text-[10.5px] px-2 py-0.5 rounded-full bg-black/25 border border-white/10 text-ink/80">
                    {t}
                  </span>
                ))}
              </div>
            </button>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <InfoCard icon={Cpu} title="Compute">
            {system ? (
              <>
                <div className="text-ink text-[14px] font-medium">{system.hardware.selected_device.toUpperCase()}</div>
                <div>
                  {system.hardware.cpu_count} CPU threads · {system.hardware.ram_gb} GB RAM
                </div>
                <div className="text-faint">Available: {system.hardware.devices.join(", ")}</div>
              </>
            ) : (
              "…"
            )}
          </InfoCard>
          <InfoCard icon={Package} title="Local models">
            <div className="text-ink text-[14px] font-medium">
              {installed} of {models.length} installed
            </div>
            <div>Inference runs with ONNX Runtime / MediaPipe on this device.</div>
            <button className="text-accent mt-1" onClick={() => go("models")}>
              Manage models →
            </button>
          </InfoCard>
          <InfoCard icon={ShieldCheck} title="Privacy">
            <div className="text-ink text-[14px] font-medium">Fully offline</div>
            <div>No cloud services, no telemetry, no accounts. The UI cannot load remote content.</div>
            <div className="text-faint flex items-center gap-1 mt-1">
              <HardDrive className="w-3 h-3" /> Temporary files: {system ? fmtBytes(system.storage_bytes) : "…"}
            </div>
          </InfoCard>
        </div>
      </div>
    </div>
  );
}

function InfoCard({ icon: Icon, title, children }: { icon: any; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-[.1em] text-faint mb-2">
        <Icon className="w-3.5 h-3.5" /> {title}
      </div>
      <div className="text-muted text-[12px] space-y-0.5 leading-relaxed">{children}</div>
    </div>
  );
}
