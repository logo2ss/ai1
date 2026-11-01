import { Box, Cpu, LayoutDashboard, Package, ScanFace, Settings2, Shapes, ShieldCheck } from "lucide-react";
import { AppProvider, Page, useApp } from "./state";
import { cx, Notice } from "./components/ui";
import { ErrorBoundary } from "./components/ErrorBoundary";
import Dashboard from "./pages/Dashboard";
import FacePage from "./pages/FacePage";
import RegionPage from "./pages/RegionPage";
import ReconPage from "./pages/ReconPage";
import ModelsPage from "./pages/ModelsPage";
import SettingsPage from "./pages/SettingsPage";

const NAV: { id: Page; label: string; icon: any; group?: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "face", label: "Face Replacement", icon: ScanFace, group: "Tools" },
  { id: "region", label: "Region Swapping", icon: Shapes, group: "Tools" },
  { id: "recon", label: "3D Reconstruction", icon: Box, group: "Tools" },
  { id: "models", label: "Models", icon: Package, group: "System" },
  { id: "settings", label: "Settings", icon: Settings2, group: "System" },
];

function Shell() {
  const { page, go, system, engineError, models } = useApp();
  const missingRequired = models.filter((m) => m.required && !m.installed);
  let lastGroup: string | undefined;
  return (
    <div className="h-full flex">
      <aside className="w-[220px] shrink-0 bg-panel border-r border-line flex flex-col">
        <div className="h-14 flex items-center gap-2.5 px-4 border-b border-line">
          <img src="./icon.svg" className="w-7 h-7" alt="" />
          <div>
            <div className="font-semibold tracking-tight leading-none">Lumen Studio</div>
            <div className="text-[10px] text-faint mt-0.5">Local photo lab</div>
          </div>
        </div>
        <nav className="flex-1 p-2 space-y-0.5 overflow-y-auto">
          {NAV.map((n) => {
            const header = n.group && n.group !== lastGroup ? n.group : null;
            lastGroup = n.group;
            return (
              <div key={n.id}>
                {header && <div className="px-2.5 pt-4 pb-1.5 text-[10px] uppercase tracking-[.12em] text-faint">{header}</div>}
                <button
                  onClick={() => go(n.id)}
                  data-testid={`nav-${n.id}`}
                  className={cx(
                    "w-full flex items-center gap-2.5 h-9 px-2.5 rounded-md text-[13px] transition",
                    page === n.id ? "bg-gradient-to-r from-accent/20 to-transparent text-ink ring-1 ring-inset ring-accent/25" : "text-muted hover:text-ink hover:bg-white/[.03]",
                  )}
                >
                  <n.icon className={cx("w-4 h-4", page === n.id && "text-accent")} />
                  {n.label}
                  {n.id === "models" && missingRequired.length > 0 && <span className="ml-auto w-2 h-2 rounded-full bg-warn" title="Required models missing" />}
                </button>
              </div>
            );
          })}
        </nav>
        <div className="p-3 border-t border-line space-y-2 text-[11px]">
          <div className="flex items-center gap-2 text-ok">
            <ShieldCheck className="w-3.5 h-3.5" /> Offline · images stay on this device
          </div>
          <div className="flex items-center gap-2 text-faint">
            <Cpu className="w-3.5 h-3.5" /> {system ? `${system.hardware.selected_device.toUpperCase()} · ${system.hardware.cpu_count} threads` : "Connecting…"}
          </div>
        </div>
      </aside>
      <main className="flex-1 min-w-0 flex flex-col">
        {engineError && (
          <div className="p-3">
            <Notice kind="error" title="Processing engine unavailable">
              {engineError} The browser version restarts the engine automatically; otherwise restart Lumen Studio. Details are in the terminal (or ~/.config/Lumen Studio/logs/engine.log).
            </Notice>
          </div>
        )}
        <div className="flex-1 min-h-0">
          {/* a failing page shows an error card instead of blanking the whole window */}
          <ErrorBoundary key={page} label={`The ${NAV.find((n) => n.id === page)?.label ?? "current"} page`}>
            {page === "dashboard" && <Dashboard />}
            {page === "face" && <FacePage />}
            {page === "region" && <RegionPage />}
            {page === "recon" && <ReconPage />}
            {page === "models" && <ModelsPage />}
            {page === "settings" && <SettingsPage />}
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary label="Lumen Studio">
      <AppProvider>
        <Shell />
      </AppProvider>
    </ErrorBoundary>
  );
}
