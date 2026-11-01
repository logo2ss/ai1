import { useState } from "react";
import { api } from "../api";
import { fmtBytes, useApp } from "../state";
import { Button, Notice, Select, Slider, Toggle } from "../components/ui";

export default function SettingsPage() {
  const { system, refresh } = useApp();
  const [saved, setSaved] = useState(false);
  if (!system) return null;
  const s = system.settings;
  const save = async (patch: Partial<typeof s>) => {
    await api.put("/api/settings", patch);
    await refresh();
    setSaved(true);
    setTimeout(() => setSaved(false), 1200);
  };
  const devices = [{ value: "auto", label: "Automatic" }, ...system.hardware.devices.map((d) => ({ value: d, label: d.toUpperCase() }))];
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto px-8 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-[22px] font-semibold tracking-tight">Settings</h1>
          {saved && <span className="text-ok text-[12px]">Saved</span>}
        </div>
        <Card title="Processing hardware">
          <Select label="Compute device for neural models" value={s.device as any} onChange={(v) => save({ device: v })} options={devices as any} />
          <div className="text-[11px] text-faint">
            Detected providers: {system.hardware.onnx_providers.join(", ")}. NVIDIA acceleration requires the <code>onnxruntime-gpu</code> package (see README); unavailable devices fall back to CPU automatically.
          </div>
          <Slider label="CPU threads" value={s.threads} min={0} max={system.hardware.cpu_count} step={1} format={(v) => (v === 0 ? "Auto" : String(v))} onChange={(v) => save({ threads: v })} />
          <Slider label="Maximum working resolution" value={s.max_working_resolution} min={1024} max={6000} step={128} format={(v) => `${v}px`} onChange={(v) => save({ max_working_resolution: v })} hint="Results are always exported at the original resolution." />
        </Card>
        <Card title="Export">
          <Toggle label="Mark exported files as edited (metadata comment)" checked={s.tag_exports} onChange={(v) => save({ tag_exports: v })} />
        </Card>
        <Card title="Local storage">
          <div className="flex items-center justify-between text-[12px]">
            <span className="text-muted">Temporary processing files</span>
            <span className="font-mono">{fmtBytes(system.storage_bytes)}</span>
          </div>
          <Button
            variant="danger"
            onClick={async () => {
              await api.del("/api/storage");
              refresh();
            }}
          >
            Clear temporary files
          </Button>
          <div className="text-[11px] text-faint">Uploaded images and results are stored only in your local data folder and are deleted automatically after 7 days.</div>
        </Card>
        <Notice kind="ok" title="Privacy guarantee">
          Lumen Studio processes images exclusively on this computer. The engine listens on 127.0.0.1 only, requires a per-session key, and the interface is forbidden (Content-Security-Policy) from contacting any other host. The only outbound
          connection ever made is a model download you explicitly start.
        </Notice>
        <div className="text-[11px] text-faint">
          Lumen Studio {system.version} · {system.hardware.os} · ONNX Runtime {system.hardware.onnxruntime}
        </div>
      </div>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-panel p-5 space-y-3.5">
      <div className="text-[11px] uppercase tracking-[.1em] text-faint">{title}</div>
      {children}
    </div>
  );
}
