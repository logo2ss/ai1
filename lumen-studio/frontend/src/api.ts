// Client for the local Lumen engine (127.0.0.1 only). No other network access is made by the UI.

const TOKEN_KEY = "lumen-token";

function initToken(): string {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    sessionStorage.setItem(TOKEN_KEY, fromUrl);
    url.searchParams.delete("token");
    window.history.replaceState(null, "", url.toString());
    return fromUrl;
  }
  return sessionStorage.getItem(TOKEN_KEY) || "";
}

export const token = initToken();

export class ApiError extends Error {
  code: string;
  hint?: string;
  details?: Record<string, unknown>;
  constructor(code: string, message: string, hint?: string, details?: Record<string, unknown>) {
    super(message);
    this.code = code;
    this.hint = hint;
    this.details = details;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: any = {};
    try {
      body = await res.json();
    } catch {
      /* not json */
    }
    throw new ApiError(body.code || String(res.status), body.message || res.statusText, body.hint, body.details);
  }
  const ct = res.headers.get("content-type") || "";
  return (ct.includes("json") ? res.json() : res.blob()) as Promise<T>;
}

const headers = () => ({ "X-Lumen-Token": token });

// A failed fetch here never means "no internet": the UI only talks to the engine on this computer.
// Browsers report a stopped/crashed local engine as a generic "NetworkError", so translate it.
export const engineUnreachable = () =>
  new ApiError(
    "engine_unreachable",
    "Lost connection to the local Lumen engine.",
    "Nothing is sent to the internet. The engine running on this computer stopped responding (it may have crashed while processing). " +
      "In the browser version it restarts automatically within a few seconds; check the terminal running Lumen Studio for details, then try again.",
  );

function lfetch(input: string, init?: RequestInit): Promise<Response> {
  return fetch(input, init).catch(() => {
    throw engineUnreachable();
  });
}

export const api = {
  get: <T>(path: string) => lfetch(path, { headers: headers() }).then((r) => handle<T>(r)),
  del: <T>(path: string) => lfetch(path, { method: "DELETE", headers: headers() }).then((r) => handle<T>(r)),
  post: <T>(path: string, body?: unknown) =>
    lfetch(path, { method: "POST", headers: { ...headers(), "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) }).then((r) => handle<T>(r)),
  put: <T>(path: string, body?: unknown) =>
    lfetch(path, { method: "PUT", headers: { ...headers(), "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) }).then((r) => handle<T>(r)),
  form: <T>(path: string, form: FormData) => lfetch(path, { method: "POST", headers: headers(), body: form }).then((r) => handle<T>(r)),
};

/** POST JSON and return a binary response (e.g. a frame rendered by the engine) with its headers. */
export async function postBlob(path: string, body: unknown): Promise<{ blob: Blob; headers: Headers }> {
  const res = await lfetch(path, { method: "POST", headers: { ...headers(), "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
  if (!res.ok) await handle(res); // throws an ApiError with the engine's message
  return { blob: await res.blob(), headers: res.headers };
}

export function withToken(path: string): string {
  return path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
}

export interface ImageMeta {
  id: string;
  width: number;
  height: number;
  name?: string;
}

export const imageUrl = (id: string, max = 0, format: "jpg" | "png" = "jpg") => withToken(`/api/images/${id}?max=${max}&format=${format}`);
export const fileUrl = (name: string) => withToken(`/api/files/${name}`);

export async function uploadImage(file: File | Blob, name = "image"): Promise<ImageMeta> {
  const fd = new FormData();
  fd.append("file", file, (file as File).name || name);
  return api.form<ImageMeta>("/api/images", fd);
}

export interface Job<T = any> {
  id: string;
  state: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  result: T | null;
  error: { code: string; message: string; hint?: string; details?: any } | null;
  elapsed?: number;
}

export async function runJob<T>(start: Promise<{ job_id: string }>, onProgress: (j: Job<T>) => void, signal?: { cancelled: boolean }): Promise<T> {
  const { job_id } = await start;
  for (;;) {
    await new Promise((r) => setTimeout(r, 350));
    if (signal?.cancelled) throw new ApiError("cancelled", "Cancelled");
    const j = await api.get<Job<T>>(`/api/jobs/${job_id}`);
    onProgress(j);
    if (j.state === "done") return j.result as T;
    if (j.state === "error") throw new ApiError(j.error?.code || "error", j.error?.message || "Failed", j.error?.hint, j.error?.details);
  }
}

export function downloadUrl(url: string, filename?: string) {
  const a = document.createElement("a");
  a.href = url;
  if (filename) a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
