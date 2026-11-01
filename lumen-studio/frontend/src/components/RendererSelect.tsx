import { cpuRendererMessage, RendererPref, useRenderer } from "../three/webgl";
import { Notice, Select } from "./ui";

/** Compact "Renderer: Auto / GPU / CPU" selector with a one-line status. */
export function RendererSelect() {
  const r = useRenderer();
  const status =
    r.kind === "gpu"
      ? `Drawing on the GPU (WebGL 2${r.support.software ? " in software — if it feels slow, choose CPU" : ""}).`
      : r.reason === "gpu-failed"
        ? "GPU failed — the engine renders on the CPU. Choose GPU to try again."
        : "The engine renders the view on the CPU.";
  return (
    <div data-testid="renderer-select" data-renderer={r.kind}>
      <Select<RendererPref>
        label="Renderer"
        value={r.pref}
        onChange={r.setPref}
        options={[
          { value: "auto", label: "Auto" },
          { value: "gpu", label: "GPU" },
          { value: "cpu", label: "CPU" },
        ]}
      />
      <div className="text-[10.5px] text-faint mt-1 leading-snug">{status}</div>
    </div>
  );
}

/** Info notice shown while the CPU renderer draws the 3D views. */
export function CpuRendererNotice() {
  const r = useRenderer();
  if (r.kind !== "cpu") return null;
  return (
    <div data-testid="cpu-notice">
      <Notice kind={r.reason === "gpu-failed" ? "warn" : "info"}>{cpuRendererMessage(r.reason, r.failure)}</Notice>
    </div>
  );
}
