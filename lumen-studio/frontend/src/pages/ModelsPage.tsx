import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Download, ExternalLink, FileUp, Trash2 } from "lucide-react";
import { api, ApiError } from "../api";
import { fmtBytes, ModelInfo, useApp } from "../state";
import { Button, ErrorNotice, Notice } from "../components/ui";

export default function ModelsPage() {
  const { models, refresh } = useApp();
  const [error, setError] = useState<ApiError | null>(null);
  const downloading = models.some((m) => m.progress?.state === "downloading" || m.progress?.state === "verifying");
  useEffect(() => {
    if (!downloading) return;
    const t = setInterval(refresh, 700);
    return () => clearInterval(t);
  }, [downloading, refresh]);
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-8 py-8">
        <h1 className="text-[22px] font-semibold tracking-tight">Local models</h1>
        <p className="text-muted mt-1 mb-5 max-w-3xl">
          Models are downloaded once — only when you click <b>Install</b> — and stored on this computer. All inference then runs offline. On air-gapped machines, download the file elsewhere and use <b>Import file</b>.
        </p>
        <div className="mb-4">
          <ErrorNotice error={error} onClose={() => setError(null)} />
        </div>
        <div className="space-y-3">
          {models.map((m) => (
            <ModelRow key={m.id} m={m} onError={setError} refresh={refresh} />
          ))}
        </div>
        <div className="mt-6">
          <Notice kind="info" title="Why these models?">
            Each was chosen for a permissive license (Apache-2.0 / MIT), CPU-friendly inference and offline use. Depth Anything V2 Base/Large are higher quality but licensed CC-BY-NC-4.0, so
            they are not offered here.
          </Notice>
        </div>
      </div>
    </div>
  );
}

function ModelRow({ m, onError, refresh }: { m: ModelInfo; onError: (e: ApiError) => void; refresh: () => Promise<void> }) {
  const file = useRef<HTMLInputElement>(null);
  const p = m.progress;
  const busy = p?.state === "downloading" || p?.state === "verifying";
  const install = async () => {
    try {
      await api.post(`/api/models/${m.id}/install`);
      setTimeout(refresh, 300);
    } catch (e) {
      onError(e as ApiError);
    }
  };
  const remove = async () => {
    await api.del(`/api/models/${m.id}`);
    refresh();
  };
  const importFile = async (f: File) => {
    const fd = new FormData();
    fd.append("file", f);
    try {
      await api.form(`/api/models/${m.id}/import`, fd);
      refresh();
    } catch (e) {
      onError(e as ApiError);
    }
  };
  return (
    <div className="rounded-xl border border-line bg-panel p-4 flex gap-4 items-start" data-testid={`model-${m.id}`}>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold">{m.name}</span>
          {m.required && <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/15 text-accent">recommended</span>}
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-muted">{m.license}</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-muted">{fmtBytes(m.size_bytes)}</span>
          {m.installed && (
            <span className="flex items-center gap-1 text-[11px] text-ok">
              <CheckCircle2 className="w-3.5 h-3.5" /> Installed
            </span>
          )}
        </div>
        <div className="text-muted text-[12px] mt-1">{m.purpose}</div>
        {m.notes && <div className="text-faint text-[11px] mt-1">{m.notes}</div>}
        <div className="text-faint text-[11px] mt-1 font-mono truncate" title={m.installed ? m.path : m.url}>
          {m.installed ? m.path : m.url}
        </div>
        {busy && (
          <div className="mt-2">
            <div className="h-1.5 rounded-full bg-[#1e2430] overflow-hidden">
              <div className="h-full bg-accent transition-all" style={{ width: `${((p?.received || 0) / Math.max(p?.total || 1, 1)) * 100}%` }} />
            </div>
            <div className="text-[11px] text-muted mt-1">
              {p?.state === "verifying" ? "Verifying checksum…" : `${fmtBytes(p?.received || 0)} / ${fmtBytes(p?.total || m.size_bytes)}`}
            </div>
          </div>
        )}
        {p?.state === "error" && <div className="text-[11px] text-bad mt-1">{p.error}</div>}
      </div>
      <div className="flex gap-2 shrink-0">
        {m.homepage && (
          <a href={m.homepage} target="_blank" rel="noreferrer" title="Model homepage (opens your browser)" className="w-8 h-8 rounded-md flex items-center justify-center text-muted hover:text-ink hover:bg-white/5">
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
        {!m.installed && (
          <>
            <Button onClick={() => file.current?.click()} title="Install from a file you downloaded separately">
              <FileUp className="w-4 h-4" /> Import file
            </Button>
            <Button variant="primary" onClick={install} disabled={busy}>
              <Download className="w-4 h-4" /> Install
            </Button>
          </>
        )}
        {m.installed && (
          <Button variant="ghost" onClick={remove} title="Remove from disk">
            <Trash2 className="w-4 h-4" />
          </Button>
        )}
        <input ref={file} type="file" className="hidden" onChange={(e) => e.target.files?.[0] && importFile(e.target.files[0])} />
      </div>
    </div>
  );
}
