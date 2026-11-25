import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The build output goes straight into the Python package, which ships it as
// package data and serves it at /ui. That keeps `make serve` working with no
// Node installed -- the built assets are committed.
export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  build: {
    outDir: "../src/fintext/static",
    emptyOutDir: true, // nothing but build output should live in that directory
    assetsDir: "assets",
  },
  server: {
    port: 5173,
    // `npm run dev` serves the UI with hot reload and forwards API calls to the
    // Python service, so the two can be developed independently.
    proxy: {
      "/v1": "http://127.0.0.1:8080",
      "/healthz": "http://127.0.0.1:8080",
    },
  },
});
