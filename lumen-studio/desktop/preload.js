// Minimal, isolated bridge. The renderer gets no Node.js access.
const { contextBridge } = require("electron");
contextBridge.exposeInMainWorld("lumenDesktop", { platform: process.platform, desktop: true });
