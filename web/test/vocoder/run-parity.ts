// Browser-side vocoder parity run (plan 02, step 5), the twin of tools/webgpu/check_vocos_ort.py plus the
// TypeScript inverse STFT: each chunk's golden mel_out goes through the graph on WebGPU, then istft and the
// output rescale, and is compared with vocoder.backbone (debug graph), vocoder.head_out and vocoder.wav.
// Then each chunk is decoded `timingRuns` more times to time the graph, the readback and the istft (D2).
import * as ort from "onnxruntime-web/webgpu";
import { istft } from "../../src/audio/istft";
import { rescaleOutput } from "../../src/audio/output";
import { Vocoder } from "../../src/vocoder";
import { errorSpread, withPlacement } from "../ort/diagnostics";
import { compare, httpReader, loadCase, loadTolerances, report, type Result, type Tensor } from "../parity";

export interface VocoderParityOptions {
  caseName: string;
  /** Folder URL with vocos.onnx and vocos_debug.onnx, such as "/testdata/webgpu/tiny/onnx/" or "/model/vocos/". */
  modelBase: string;
  goldenBase?: string;
  /** Run vocos_debug.onnx and check the backbone output too. */
  debug?: boolean;
  timingRuns?: number;
}

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? NaN;
const round1 = (x: number) => Math.round(x * 10) / 10;

export async function runVocoderParity(options: VocoderParityOptions) {
  const t0 = performance.now();
  const read = httpReader(options.goldenBase ?? "/testdata/webgpu");
  const golden = await loadCase(read, options.caseName);
  const tolerances = await loadTolerances(read);
  const file = options.debug ? "vocos_debug.onnx" : "vocos.onnx";
  const { value: vocoder, placement } = await withPlacement(() =>
    Vocoder.create({ url: `${options.modelBase}${file}`, logSeverityLevel: 0 }),
  );
  const loadMs = performance.now() - t0;
  const device = await ort.env.webgpu.device;
  const gpuErrors: string[] = [];
  device.addEventListener("uncapturederror", (e) => gpuErrors.push((e as GPUUncapturedErrorEvent).error.message));

  const results: Result[] = [];
  const timing: { chunk: number; frames: number; graphMs: number; readbackMs: number; istftMs: number }[] = [];
  try {
    for (let c = 0; c < golden.manifest.chunks.length; c++) {
      const mel = await golden.tensor(`c${c}.mel_out`); // (1, 100, T)
      const frames = mel.shape[2] as number;
      const input = new ort.Tensor("float32", mel.data as Float32Array, mel.shape);
      const out = await vocoder.run(input);
      try {
        for (const name of vocoder.outputNames) {
          const t = out[name] as ort.Tensor;
          const data = (await t.getData()) as Float32Array;
          const actual: Tensor = { dtype: "float32", shape: [...t.dims], data };
          results.push(...(await compare(golden, tolerances, `c${c}.vocoder.${name}`, actual)));
          if (name === "head_out") {
            const wave = rescaleOutput(istft(data, frames), golden.manifest.audio);
            results.push(
              ...(await compare(golden, tolerances, `c${c}.vocoder.wav`, {
                dtype: "float32",
                shape: [wave.length],
                data: wave,
              })),
            );
          }
        }
      } finally {
        for (const t of Object.values(out)) t.dispose();
      }
      const runs: { graphMs: number; readbackMs: number; istftMs: number }[] = [];
      for (let i = 0; i < (options.timingRuns ?? 3); i++) runs.push((await vocoder.decode(input)).timing);
      timing.push({
        chunk: c,
        frames,
        graphMs: round1(median(runs.map((r) => r.graphMs))),
        readbackMs: round1(median(runs.map((r) => r.readbackMs))),
        istftMs: round1(median(runs.map((r) => r.istftMs))),
      });
    }
  } finally {
    await vocoder.release();
  }
  const r = report(results);
  const lines = r.text.split("\n");
  return {
    caseName: options.caseName,
    adapter: { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture },
    comparisons: results.length,
    failed: r.failed,
    summary: lines.at(-1) ?? "",
    failures: lines.filter((l) => l.startsWith("FAIL")),
    errorSpread: errorSpread(results),
    placement,
    gpuErrors,
    timing,
    loadMs: Math.round(loadMs),
    totalMs: Math.round(performance.now() - t0),
  };
}

export type VocoderParityResult = Awaited<ReturnType<typeof runVocoderParity>>;
