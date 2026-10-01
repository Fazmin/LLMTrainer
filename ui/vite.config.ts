import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Tauri expects a fixed dev port; the browser-only preview (fixture backend) uses the same one.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, host: "127.0.0.1" },
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
  test: { environment: "jsdom", setupFiles: ["./tests/setup.ts"], globals: true },
});
