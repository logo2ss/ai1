import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the engine runs separately; set LUMEN_ENGINE=http://127.0.0.1:PORT
const engine = process.env.LUMEN_ENGINE || "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  server: { proxy: { "/api": engine } },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1500 },
});
