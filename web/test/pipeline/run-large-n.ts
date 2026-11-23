// Plan 04 step 4 (D7): does a step fit at large n, and how long does it take? Random inputs, `steps` steps
// through sampleChunk per n. Loaded into the page by test/browser-local/large-n.spec.ts.
import * as ort from "onnxruntime-web/webgpu";
import { DiT } from "../../src/dit";
import { sampleChunk, TensorLedger } from "../../src/pipeline/sample";

export interface LargeNOptions {
  modelBase: string;
  sizes: number[];
  steps: number;
  stepFile?: string;
}

export async function runLargeN(options: LargeNOptions) {
  const dit = await DiT.create({
    textUrl: `${options.modelBase}text.onnx`,
    stepUrl: `${options.modelBase}${options.stepFile ?? "step_mha.onnx"}`,
  });
  const device = await ort.env.webgpu.device;
  const gpuErrors: string[] = [];
  device.addEventListener("uncapturederror", (e) => gpuErrors.push((e as GPUUncapturedErrorEvent).error.message));
  let lost = "";
  void device.lost.then((info) => (lost = `${info.reason}: ${info.message}`));
  const ledger = new TensorLedger();
  const runs = [];
  try {
    for (const n of options.sizes) {
      const noise = Float32Array.from({ length: n * 100 }, () => Math.random() * 2 - 1);
      const cond = new Float32Array(n * 100);
      cond.set(noise.subarray(0, Math.floor(n / 3) * 100));
      const ids = Int32Array.from({ length: Math.floor(n / 4) }, (_, i) => 1 + (i % 2000));
      const grid = Float32Array.from({ length: options.steps + 1 }, (_, i) => i / 32);
      const stepMs: number[] = [];
      let last = performance.now();
      const t0 = last;
      let error = "";
      let finite = false;
      try {
        const x = await sampleChunk(
          dit,
          device,
          ledger,
          { ids, noise, stepCond: cond, duration: n, grid, cfgStrength: 2 },
          {
            onStep: () => {
              const now = performance.now();
              stepMs.push(Math.round(now - last));
              last = now;
            },
          },
        );
        finite = x.every(Number.isFinite);
      } catch (e) {
        error = String(e);
      }
      runs.push({ n, totalMs: Math.round(performance.now() - t0), stepMs, finite, error, live: ledger.live });
      if (lost) break;
    }
  } finally {
    await dit.release();
  }
  return { adapter: device.adapterInfo.vendor, runs, gpuErrors: gpuErrors.slice(0, 5), lost };
}

export type LargeNResult = Awaited<ReturnType<typeof runLargeN>>;
