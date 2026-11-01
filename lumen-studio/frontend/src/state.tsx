import { createContext, ReactNode, useCallback, useContext, useEffect, useState } from "react";
import { api } from "./api";

export interface ModelInfo {
  id: string;
  name: string;
  filename: string;
  url: string;
  size_bytes: number;
  license: string;
  purpose: string;
  used_by: string[];
  required: boolean;
  notes: string;
  homepage: string;
  installed: boolean;
  path: string;
  progress: { state: string; received?: number; total?: number; error?: string | null } | null;
}

export interface SystemInfo {
  hardware: { os: string; machine: string; cpu_count: number; ram_gb: number; onnxruntime: string; onnx_providers: string[]; devices: string[]; selected_device: string };
  settings: { device: string; threads: number; max_working_resolution: number; depth_model: string; tag_exports: boolean };
  storage_bytes: number;
  version: string;
  mediapipe?: { usable: boolean; reason: string | null; hint: string };
}

export type Page = "dashboard" | "face" | "region" | "recon" | "models" | "settings";

interface Ctx {
  system: SystemInfo | null;
  models: ModelInfo[];
  refresh: () => Promise<void>;
  page: Page;
  go: (p: Page) => void;
  engineError: string | null;
}

const C = createContext<Ctx>(null as any);
export const useApp = () => useContext(C);

export function AppProvider({ children }: { children: ReactNode }) {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [page, setPage] = useState<Page>(() => (window.location.hash.slice(1) as Page) || "dashboard");
  const [engineError, setEngineError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([api.get<SystemInfo>("/api/system"), api.get<ModelInfo[]>("/api/models")]);
      setSystem(s);
      setModels(m);
      setEngineError(null);
    } catch (e: any) {
      setEngineError(e.message || "The local processing engine is not reachable.");
    }
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh]);
  // watch the local engine: show a clear banner if it stops, recover automatically when it is back
  useEffect(() => {
    let down = false;
    const t = setInterval(async () => {
      try {
        await api.get("/api/health");
        if (down) {
          down = false;
          refresh();
        }
      } catch {
        down = true;
        setEngineError("The local engine is not responding (it may be restarting after a crash). Nothing is sent to the internet.");
      }
    }, 4000);
    return () => clearInterval(t);
  }, [refresh]);
  const go = (p: Page) => {
    setPage(p);
    window.location.hash = p;
  };
  return <C.Provider value={{ system, models, refresh, page, go, engineError }}>{children}</C.Provider>;
}

export function useModel(id: string): ModelInfo | undefined {
  return useApp().models.find((m) => m.id === id);
}

export const fmtBytes = (n: number) => (n > 1e9 ? (n / 1e9).toFixed(2) + " GB" : n > 1e6 ? (n / 1e6).toFixed(1) + " MB" : (n / 1e3).toFixed(0) + " KB");
