// Browser-side run of plan 04 step 4: sampleChunk on every chunk of a golden case, free-running from the
// golden noise and step_cond ("golden") or from this port's own noise, step_cond and time grid ("chain").
// Records the error after every step against the golden x, the live-tensor count, and the time per chunk.
// Loaded into the page by test/browser-local/sample.spec.ts.
import * as ort from "onnxruntime-web/webgpu";
import { DiT } from "../../src/dit";
import { finishChunk, stepCond } from "../../src/pipeline/cond";
import { TorchGenerator } from "../../src/pipeline/noise";
import { timeGrid } from "../../src/pipeline/plan";
import { sampleChunk, TensorLedger } from "../../src/pipeline/sample";
import { compare, httpReader, loadCase, loadTolerances, metrics, report, xNextGolden } from "../parity";

export interface SampleRunOptions {
  caseName: string;
  modelBase: string;
  stepFile?: string;
  mode: "golden" | "chain";
  /** Read x back after every step and compare it with the golden x (costs a readback per step). */
  drift?: boolean;
}

const pad2 = (s: number) => String(s).padStart(2, "0");

export async function runSample(options: SampleRunOptions) {
  const read = httpReader("/testdata/webgpu");
  const golden = await loadCase(read, options.caseName);
  const tolerances = await loadTolerances(read);
  const s = golden.manifest.settings as { cfg_strength: number; nfe_step: number; sway_sampling_coef: number | null };
  const tLoad = performance.now();
  const dit = await DiT.create({
    textUrl: `${options.modelBase}text.onnx`,
    stepUrl: `${options.modelBase}${options.stepFile ?? "step_mha.onnx"}`,
  });
  const loadMs = Math.round(performance.now() - tLoad);
  const device = await ort.env.webgpu.device;
  const gpuErrors: string[] = [];
  device.addEventListener("uncapturederror", (e) => gpuErrors.push((e as GPUUncapturedErrorEvent).error.message));
  const ledger = new TensorLedger();
  const liveAfterChunk: number[] = [];
  const chunkMs: number[] = [];
  const drift: { chunk: number; relL2: string[] }[] = [];
  const results = [];
  const melRef = await golden.tensor("mel.ref");
  const mel = { data: melRef.data as Float32Array, frames: melRef.shape[2] as number };
  const gen = new TorchGenerator(0);
  try {
    for (const [c, chunk] of (golden.manifest.chunks as { duration: number; ref_audio_len: number }[]).entries()) {
      const n = chunk.duration;
      const chain = options.mode === "chain";
      const noise = chain ? gen.randn(n * 100) : ((await golden.tensor(`c${c}.noise.y0`)).data as Float32Array);
      const cond = chain ? stepCond(mel, n) : ((await golden.tensor(`c${c}.dit.step_cond`)).data as Float32Array);
      const grid = chain
        ? timeGrid(s.nfe_step, s.sway_sampling_coef)
        : ((await golden.tensor(`c${c}.plan.t`)).data as Float32Array);
      const ids = (await golden.tensor(`c${c}.dit.text_ids`)).data as Int32Array;
      const rel: number[] = [];
      const onStep = async (step: number, _steps: number, x: ort.Tensor) => {
        if (!options.drift) return;
        const got = (await x.getData()) as Float32Array;
        if (step + 1 < s.nfe_step) {
          const ref = (await golden.tensor(`c${c}.dit.s${pad2(step + 1)}.x`)).data;
          rel.push(metrics(got, ref).relL2 ?? NaN);
        } else {
          const { start, ref } = await xNextGolden(golden, c, step);
          rel.push(metrics(got.subarray(start * 100), ref.data).relL2 ?? NaN);
        }
      };
      const t0 = performance.now();
      const x = await sampleChunk(
        dit,
        device,
        ledger,
        { ids, noise, stepCond: cond, duration: n, grid, cfgStrength: s.cfg_strength },
        { onStep },
      );
      chunkMs.push(Math.round(performance.now() - t0));
      liveAfterChunk.push(ledger.live);
      if (options.drift) drift.push({ chunk: c, relL2: rel.map((v) => v.toExponential(2)) });
      const out = finishChunk(x, mel, chunk.ref_audio_len);
      const frames = n - chunk.ref_audio_len;
      results.push(
        ...(await compare(golden, tolerances, `c${c}.mel_out`, {
          dtype: "float32",
          shape: [1, 100, frames],
          data: out,
        })),
      );
    }
  } finally {
    await dit.release();
  }
  const r = report(results);
  return {
    caseName: options.caseName,
    mode: options.mode,
    adapter: { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture },
    lines: r.text.split("\n"),
    failed: r.failed,
    liveAfterChunk,
    liveAtEnd: ledger.live,
    chunkMs,
    loadMs,
    drift,
    gpuErrors,
  };
}

export type SampleRunResult = Awaited<ReturnType<typeof runSample>>;
