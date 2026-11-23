import { join } from "node:path";
import { defineConfig } from "vite";
import { serveData } from "./serve-data.ts";

const repo = join(import.meta.dirname, "..");

export default defineConfig({
  plugins: [
    serveData({
      "/testdata/webgpu/": join(repo, "testdata/webgpu"), // golden data and the tiny model's graphs
      "/model/dit/": join(repo, "build/webgpu/dit"), // the real DiT exports (tools/webgpu/export_dit.py)
      "/model/vocos/": join(repo, "build/webgpu/vocos"), // the real Vocos export (tools/webgpu/export_vocos.py)
      "/vocab/": join(repo, "src/f5_tts/infer/examples"), // vocab.txt, for the pipeline tests (plan 04)
    }),
  ],
  // as in the plan 00 spike: ORT Web finds its .wasm files relative to its own module, so it is not pre-bundled
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
