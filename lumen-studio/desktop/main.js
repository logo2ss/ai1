// Lumen Studio desktop shell (Electron).
//
// Starts the local Python engine on a random 127.0.0.1 port, then shows the UI
// it serves. Every request the window makes is filtered: anything that is not
// the local engine is cancelled, so images can never leave the computer.

const { app, BrowserWindow, dialog, session, shell, Menu } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");

let engine = null;
let win = null;
let engineInfo = null;

function resolveEngine() {
  const token = crypto.randomBytes(24).toString("base64url");
  if (app.isPackaged && !process.env.LUMEN_PYTHON) {
    const base = path.join(process.resourcesPath, "engine");
    const exe = process.platform === "win32" ? "lumen-engine.exe" : "lumen-engine";
    return { cmd: path.join(base, exe), args: ["--port", "0", "--static", path.join(process.resourcesPath, "ui"), "--token", token], token };
  }
  const root = path.resolve(__dirname, "..");
  const py =
    process.env.LUMEN_PYTHON ||
    (process.platform === "win32" ? path.join(root, ".venv", "Scripts", "python.exe") : path.join(root, ".venv", "bin", "python"));
  return { cmd: py, args: ["-m", "lumen_engine.server.main", "--port", "0", "--static", path.join(root, "frontend", "dist"), "--token", token], token };
}

function startEngine() {
  return new Promise((resolve, reject) => {
    const { cmd, args, token } = resolveEngine();
    if (!fs.existsSync(cmd)) {
      reject(new Error(`Processing engine not found at ${cmd}. Run the setup script first (see README).`));
      return;
    }
    const logDir = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(logDir, { recursive: true });
    const log = fs.createWriteStream(path.join(logDir, "engine.log"), { flags: "a" });
    engine = spawn(cmd, args, { env: { ...process.env, PYTHONUNBUFFERED: "1" }, windowsHide: true });
    let buf = "";
    const timer = setTimeout(() => reject(new Error("The processing engine did not start within 90 seconds.")), 90000);
    engine.stdout.on("data", (d) => {
      log.write(d);
      buf += d.toString();
      const m = buf.match(/LUMEN_READY (\{.*\})/);
      if (m && !engineInfo) {
        clearTimeout(timer);
        engineInfo = { ...JSON.parse(m[1]), token };
        resolve(engineInfo);
      }
    });
    engine.stderr.on("data", (d) => log.write(d));
    engine.on("exit", (code) => {
      log.write(`\n[engine exited with code ${code}]\n`);
      if (!engineInfo) reject(new Error(`The processing engine exited (code ${code}). See ${path.join(logDir, "engine.log")}.`));
      else if (win && !app.isQuitting) dialog.showErrorBox("Lumen Studio", "The processing engine stopped unexpectedly. Please restart the application.");
    });
  });
}

function lockDownNetwork(port) {
  const allowed = [`http://127.0.0.1:${port}/`, "devtools://", "data:", "blob:", "chrome-extension://"];
  session.defaultSession.webRequest.onBeforeRequest((details, cb) => {
    const ok = allowed.some((p) => details.url.startsWith(p));
    if (!ok) console.warn("[lumen] blocked request:", details.url);
    cb({ cancel: !ok });
  });
  session.defaultSession.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));
}

async function createWindow() {
  win = new BrowserWindow({
    width: 1480,
    height: 920,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#0a0c10",
    title: "Lumen Studio",
    show: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, preload: path.join(__dirname, "preload.js") },
  });
  win.once("ready-to-show", () => win.show());
  // Open external links (model homepages) in the user's browser, never inside the app.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url) && !url.startsWith("http://127.0.0.1")) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith(`http://127.0.0.1:${engineInfo.port}/`)) e.preventDefault();
  });
  // Exports: always ask where to save.
  win.webContents.session.on("will-download", (_e, item) => {
    const p = dialog.showSaveDialogSync(win, { defaultPath: path.join(app.getPath("pictures"), item.getFilename()) });
    if (p) item.setSavePath(p);
    else item.cancel();
  });
  await win.loadURL(`http://127.0.0.1:${engineInfo.port}/?token=${encodeURIComponent(engineInfo.token)}`);
}

if (!app.requestSingleInstanceLock()) app.quit();
app.on("second-instance", () => {
  if (win) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});

app.whenReady().then(async () => {
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      ...(process.platform === "darwin" ? [{ role: "appMenu" }] : []),
      { role: "fileMenu" },
      { role: "editMenu" },
      { label: "View", submenu: [{ role: "reload" }, { role: "toggleDevTools" }, { type: "separator" }, { role: "resetZoom" }, { role: "zoomIn" }, { role: "zoomOut" }, { type: "separator" }, { role: "togglefullscreen" }] },
      { role: "windowMenu" },
    ]),
  );
  try {
    await startEngine();
    lockDownNetwork(engineInfo.port);
    await createWindow();
    if (process.env.LUMEN_SCREENSHOT) {
      setTimeout(async () => {
        const img = await win.webContents.capturePage();
        fs.writeFileSync(process.env.LUMEN_SCREENSHOT, img.toPNG());
        app.quit();
      }, 6000);
    }
  } catch (err) {
    dialog.showErrorBox("Lumen Studio could not start", String(err.message || err));
    app.quit();
  }
});

app.on("before-quit", () => {
  app.isQuitting = true;
  if (engine && !engine.killed) engine.kill();
});
app.on("window-all-closed", () => app.quit());
