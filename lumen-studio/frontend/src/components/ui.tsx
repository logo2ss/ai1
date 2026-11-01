import { ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Info, Loader2, Upload, XCircle } from "lucide-react";
import type { ApiError } from "../api";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

export function Button({
  children,
  onClick,
  variant = "default",
  disabled,
  title,
  className,
  size = "md",
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "ghost" | "danger";
  disabled?: boolean;
  title?: string;
  className?: string;
  size?: "sm" | "md" | "lg";
  type?: "button" | "submit";
}) {
  const v = {
    default: "bg-panel-2 hover:bg-[#1c2130] border border-line-2 text-ink",
    primary: "bg-gradient-to-b from-[#9a8fff] to-[#7a6cf5] hover:brightness-110 text-white border border-[#a59bff]/40 shadow-[0_6px_20px_-8px_rgba(143,131,255,.7)]",
    ghost: "hover:bg-panel-2 text-muted hover:text-ink",
    danger: "bg-[#2a1517] hover:bg-[#3a1b1e] border border-[#5a2a2e] text-[#ffb3b3]",
  }[variant];
  const s = { sm: "h-7 px-2.5 text-xs gap-1.5", md: "h-8 px-3 text-[13px] gap-2", lg: "h-10 px-5 text-sm gap-2" }[size];
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cx("inline-flex items-center justify-center rounded-md font-medium transition select-none disabled:opacity-40 disabled:pointer-events-none whitespace-nowrap", v, s, className)}
    >
      {children}
    </button>
  );
}

export function Slider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  format,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: (v: number) => string;
  hint?: string;
}) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <label className="block" title={hint}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[12px] text-muted">{label}</span>
        <span className="text-[11px] font-mono text-ink/80 tabular-nums">{format ? format(value) : Math.round(((value - min) / (max - min)) * 100) + "%"}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ ["--fill" as any]: `${fill}%` }}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label}
      />
    </label>
  );
}

export function Toggle({ label, checked, onChange, hint, disabled }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string; disabled?: boolean }) {
  return (
    <label className={cx("flex items-center justify-between gap-3 py-0.5", disabled ? "opacity-45 cursor-not-allowed" : "cursor-pointer")} title={hint}>
      <span className="text-[12px] text-muted">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx("relative w-8 h-[18px] rounded-full transition shrink-0 disabled:pointer-events-none", checked ? "bg-accent" : "bg-[#2a3140]")}
      >
        <span className={cx("absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-all", checked ? "left-[16px]" : "left-[2px]")} />
      </button>
    </label>
  );
}

export function Select<T extends string>({ label, value, onChange, options }: { label: string; value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <label className="block">
      <div className="text-[12px] text-muted mb-1">{label}</div>
      <div className="flex rounded-md bg-[#0d1015] border border-line p-0.5 gap-0.5">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            className={cx("flex-1 h-6 rounded text-[11.5px] transition", value === o.value ? "bg-panel-2 text-ink shadow border border-line-2" : "text-muted hover:text-ink")}
          >
            {o.label}
          </button>
        ))}
      </div>
    </label>
  );
}

export function Section({ title, children, right, defaultOpen = true }: { title: string; children: ReactNode; right?: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-line">
      <button type="button" onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-4 h-9 text-[11px] uppercase tracking-[.08em] text-faint hover:text-muted">
        <span>{title}</span>
        <span className="flex items-center gap-2 normal-case tracking-normal">{right}<span className={cx("transition", open ? "rotate-90" : "")}>›</span></span>
      </button>
      {open && <div className="px-4 pb-4 space-y-3.5">{children}</div>}
    </div>
  );
}

export function ProgressBar({ value, message }: { value: number; message?: string }) {
  return (
    <div className="w-full">
      <div className="flex justify-between text-[11px] text-muted mb-1.5">
        <span className="flex items-center gap-1.5"><Loader2 className="w-3 h-3 animate-spin" />{message || "Processing…"}</span>
        <span className="font-mono tabular-nums">{Math.round(value * 100)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-[#1e2430] overflow-hidden">
        <div className="h-full bg-gradient-to-r from-accent-2 to-accent transition-all duration-300 relative" style={{ width: `${Math.max(3, value * 100)}%` }}>
          <div className="absolute inset-0 shimmer" />
        </div>
      </div>
    </div>
  );
}

export function Notice({ kind = "info", title, children, onClose }: { kind?: "info" | "warn" | "error" | "ok"; title?: string; children?: ReactNode; onClose?: () => void }) {
  const style = {
    info: ["border-[#2b3550] bg-[#121828] text-[#b9c6ff]", Info],
    warn: ["border-[#4d3c1c] bg-[#1d170c] text-[#f6d38d]", AlertTriangle],
    error: ["border-[#5a2a2e] bg-[#1f1012] text-[#ffb3b3]", XCircle],
    ok: ["border-[#1e4a33] bg-[#0d1c15] text-[#9ce9bf]", CheckCircle2],
  }[kind] as [string, any];
  const Icon = style[1];
  return (
    <div className={cx("rounded-md border px-3 py-2.5 text-[12px] leading-relaxed flex gap-2.5", style[0])} role={kind === "error" ? "alert" : "status"}>
      <Icon className="w-4 h-4 shrink-0 mt-[1px]" />
      <div className="min-w-0 flex-1">
        {title && <div className="font-semibold mb-0.5">{title}</div>}
        <div className="opacity-90">{children}</div>
      </div>
      {onClose && (
        <button className="opacity-60 hover:opacity-100 self-start" onClick={onClose} aria-label="Dismiss">
          ×
        </button>
      )}
    </div>
  );
}

export function ErrorNotice({ error, onClose }: { error: ApiError | Error | null; onClose?: () => void }) {
  if (!error) return null;
  const e = error as ApiError;
  return (
    <Notice kind="error" title={e.message} onClose={onClose}>
      {e.hint}
    </Notice>
  );
}

export function Dropzone({ onFile, label, sub, compact }: { onFile: (f: File) => void; label: string; sub?: string; compact?: boolean }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (f) onFile(f);
  };
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        pick(e.dataTransfer.files);
      }}
      onClick={() => input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
      className={cx(
        "w-full h-full flex flex-col items-center justify-center text-center rounded-xl border border-dashed cursor-pointer transition",
        over ? "border-accent bg-accent/5" : "border-line-2 hover:border-[#3d4659] hover:bg-white/[.015]",
        compact ? "p-4" : "p-8",
      )}
    >
      <div className={cx("rounded-full bg-panel-2 border border-line-2 flex items-center justify-center mb-3", compact ? "w-9 h-9" : "w-12 h-12")}>
        <Upload className={compact ? "w-4 h-4 text-muted" : "w-5 h-5 text-muted"} />
      </div>
      <div className="text-ink font-medium">{label}</div>
      <div className="text-muted text-[12px] mt-1">{sub || "Drop an image here or click to browse"}</div>
      <div className="text-faint text-[11px] mt-2">JPEG · PNG · WebP · TIFF — processed only on this computer</div>
      <input ref={input} type="file" accept="image/*" className="hidden" onChange={(e) => pick(e.target.files)} />
    </div>
  );
}

/** Before/after comparison with a draggable divider; also supports toggle mode. */
export function CompareSlider({ before, after, labels = ["Before", "After"] }: { before: string; after: string; labels?: [string, string] }) {
  const [pos, setPos] = useState(0.5);
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef(false);
  const move = useCallback((clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (r) setPos(Math.min(1, Math.max(0, (clientX - r.left) / r.width)));
  }, []);
  useEffect(() => {
    const up = () => (drag.current = false);
    const mm = (e: PointerEvent) => drag.current && move(e.clientX);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointermove", mm);
    return () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointermove", mm);
    };
  }, [move]);
  return (
    <div className="w-full h-full flex items-center justify-center">
      <div
        ref={box}
        className="relative max-w-full max-h-full select-none cursor-ew-resize"
        onPointerDown={(e) => {
          drag.current = true;
          move(e.clientX);
        }}
        data-testid="compare"
      >
        <img src={after} className="block max-w-full max-h-[calc(100vh-150px)] object-contain pointer-events-none" draggable={false} alt="result" />
        <img
          src={before}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          style={{ clipPath: `inset(0 ${100 - pos * 100}% 0 0)` }}
          draggable={false}
          alt="original"
        />
        <div className="absolute top-0 bottom-0 w-px bg-white/90 shadow-[0_0_8px_rgba(0,0,0,.6)]" style={{ left: `${pos * 100}%` }}>
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-white text-black flex items-center justify-center text-[11px] font-bold shadow-lg">⇆</div>
        </div>
        <span className="absolute left-2 top-2 text-[10px] uppercase tracking-wider bg-black/60 rounded px-1.5 py-0.5">{labels[0]}</span>
        <span className="absolute right-2 top-2 text-[10px] uppercase tracking-wider bg-black/60 rounded px-1.5 py-0.5">{labels[1]}</span>
      </div>
    </div>
  );
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: "ok" | "warn" | "bad" }) {
  return (
    <div className="flex items-center justify-between text-[12px] py-0.5">
      <span className="text-muted">{label}</span>
      <span className={cx("font-mono tabular-nums", tone === "ok" && "text-ok", tone === "warn" && "text-warn", tone === "bad" && "text-bad")}>{value}</span>
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="px-1.5 py-0.5 rounded bg-panel-2 border border-line-2 text-[10px] font-mono text-muted">{children}</kbd>;
}

export function EmptyHint({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center p-8">
      <div className="w-12 h-12 rounded-xl bg-panel-2 border border-line-2 flex items-center justify-center text-muted mb-3">{icon}</div>
      <div className="font-medium">{title}</div>
      <div className="text-muted text-[12px] mt-1 max-w-sm">{children}</div>
    </div>
  );
}


/** Before/after over a live view (e.g. a 3D canvas): the "before" image covers the left part. */
export function OverlayCompare({ before, children, labels = ["Before", "After"] }: { before: string; children: ReactNode; labels?: [string, string] }) {
  const [pos, setPos] = useState(0.5);
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef(false);
  useEffect(() => {
    const up = () => (drag.current = false);
    const mm = (e: PointerEvent) => {
      if (!drag.current || !box.current) return;
      const r = box.current.getBoundingClientRect();
      setPos(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
    };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointermove", mm);
    return () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointermove", mm);
    };
  }, []);
  return (
    <div ref={box} className="absolute inset-0" data-testid="overlay-compare">
      {children}
      <div className="absolute inset-0 pointer-events-none" style={{ clipPath: `inset(0 ${100 - pos * 100}% 0 0)` }}>
        <div className="absolute inset-0 bg-[#0a0c10]" />
        <img src={before} className="absolute inset-0 w-full h-full object-contain" alt="before" draggable={false} />
      </div>
      <div className="absolute top-0 bottom-0 w-px bg-white/90 shadow-[0_0_8px_rgba(0,0,0,.6)]" style={{ left: `${pos * 100}%` }}>
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white text-black flex items-center justify-center text-[11px] font-bold shadow-lg cursor-ew-resize"
          onPointerDown={(e) => {
            e.stopPropagation();
            drag.current = true;
          }}
        >
          ⇆
        </div>
      </div>
      <span className="absolute left-2 top-2 text-[10px] uppercase tracking-wider bg-black/60 rounded px-1.5 py-0.5 pointer-events-none">{labels[0]}</span>
      <span className="absolute right-2 top-2 text-[10px] uppercase tracking-wider bg-black/60 rounded px-1.5 py-0.5 pointer-events-none">{labels[1]}</span>
    </div>
  );
}
