import { join } from "node:path";
import { defineConfig } from "vite";
import { serveData } from "./serve-data.ts";

const repo = join(import.meta.dirname, "..");

export default defineConfig({
  plugins: [
    serveData({
      "/testdata/webgpu/": join(repo, "testdata/webgpu"), // golden data and the tiny model's graphs
      "/model/dit/": join(repo, "build/webgpu/dit"), // the real DiT exports (tools/webgpu/export_dit.py)
    }),
  ],
  // as in the plan 00 spike: ORT Web finds its .wasm files relative to its own module, so it is not pre-bundled
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
