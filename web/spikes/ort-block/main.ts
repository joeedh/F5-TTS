// Plan 00 step 3 spike page. run.mjs calls window.runSpike() through Playwright and saves the JSON.
// Frozen after the D3 decision; see docs/research/webgpu-runtime-spike.md.
import * as ort from "onnxruntime-web/webgpu";
import { runWgslMatmul } from "./matmul";
import { fetchNpy, type NpyArray } from "./npy";

interface ModelEntry {
  name: string;
  kind: string;
  n: number;
  inputs: Record<string, number[]>;
  output: number[];
  flops: number;
}

export interface SpikeOptions {
  models?: string[]; // name filter; default all
  fp16?: boolean; // also run the _fp16 copies
  warmup?: number;
  runs?: number;
  chainRuns?: number; // how many 22-block chains to time
  verbose?: boolean; // ORT verbose log, needed for placement
}

const DATA = "/data/";

function stats(xs: number[]): { median: number; min: number; max: number } {
  const s = [...xs].sort((a, b) => a - b);
  return { median: s[Math.floor(s.length / 2)] ?? NaN, min: s[0] ?? NaN, max: s[s.length - 1] ?? NaN };
}

function compare(got: Float32Array, ref: Float32Array): { relL2: number; maxAbs: number; nonFinite: number } {
  let num = 0;
  let den = 0;
  let maxAbs = 0;
  let nonFinite = 0;
  for (let i = 0; i < ref.length; i++) {
    const g = got[i] ?? NaN;
    const r = ref[i] ?? NaN;
    if (!Number.isFinite(g)) {
      nonFinite++;
      continue;
    }
    const d = g - r;
    num += d * d;
    den += r * r;
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { relL2: Math.sqrt(num / den), maxAbs, nonFinite };
}

// Captures everything ORT prints, so placement can be read back per model.
const logLines: string[] = [];
for (const level of ["log", "info", "warn", "error", "debug"] as const) {
  const orig = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    logLines.push(args.map(String).join(" "));
    orig(...args);
  };
}

// Lines from ORT's verbose log that say where nodes went.
function placementLines(lines: string[]): string[] {
  return lines.filter((l) =>
    /placed on|Node placements|node\(s\) placed|CPUExecutionProvider|fallback|not supported|Unsupported/i.test(l),
  );
}

// ORT 1.31 ignores a device set on env.webgpu.device before the first session and creates its own,
// then writes it back there. So the spike makes a tiny session first and borrows ORT's device.
// ORT destroys that device when its last session is released, so the keeper session stays open.
let keeper: ort.InferenceSession | undefined;
async function ortDevice(): Promise<{ device: GPUDevice; adapterInfo: Record<string, unknown> }> {
  const bytes = await (await fetch(`${DATA}matmul_n256.onnx`)).arrayBuffer();
  keeper ??= await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
  const device = await ort.env.webgpu.device;
  const info = device.adapterInfo;
  return {
    device,
    adapterInfo: {
      vendor: info.vendor,
      architecture: info.architecture,
      description: info.description,
      features: [...device.features].sort(),
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    },
  };
}

function gpuTensor(device: GPUDevice, arr: NpyArray): ort.Tensor {
  const buf = device.createBuffer({
    size: arr.data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, arr.data);
  return ort.Tensor.fromGpuBuffer(buf, { dataType: "float32", dims: arr.shape, dispose: () => buf.destroy() });
}

async function runModel(
  device: GPUDevice,
  entry: ModelEntry,
  file: string,
  opts: Required<SpikeOptions>,
): Promise<Record<string, unknown>> {
  const result: Record<string, unknown> = { name: entry.name, file, kind: entry.kind, n: entry.n };
  const inputs: Record<string, NpyArray> = {};
  for (const key of Object.keys(entry.inputs)) inputs[key] = await fetchNpy(`${DATA}${entry.name}.in.${key}.npy`);
  const ref = await fetchNpy(`${DATA}${entry.name}.ref.npy`);
  const bytes = await (await fetch(`${DATA}${file}`)).arrayBuffer();

  const logStart = logLines.length;
  const tCreate = performance.now();
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["webgpu"],
    preferredOutputLocation: "gpu-buffer",
    logSeverityLevel: opts.verbose ? 0 : 2,
    logVerbosityLevel: 0,
  });
  result.createMs = performance.now() - tCreate;
  result.sameDevice = (await ort.env.webgpu.device) === device;
  result.placementLog = placementLines(logLines.slice(logStart));
  result.logLineCount = logLines.length - logStart;

  try {
    const feeds: Record<string, ort.Tensor> = {};
    for (const [key, arr] of Object.entries(inputs)) feeds[key] = gpuTensor(device, arr);

    // correctness
    const first = await session.run(feeds);
    const out = first.out ?? Object.values(first)[0];
    if (!out) throw new Error("no output");
    const got = (await out.getData(true)) as Float32Array;
    result.correctness = compare(got, ref.data);

    // timing: warm-up, then single runs with inputs and outputs on the GPU
    for (let i = 0; i < opts.warmup; i++) {
      const o = await session.run(feeds);
      for (const t of Object.values(o)) t.dispose();
    }
    await device.queue.onSubmittedWorkDone();
    const times: number[] = [];
    for (let i = 0; i < opts.runs; i++) {
      const t0 = performance.now();
      const o = await session.run(feeds);
      await device.queue.onSubmittedWorkDone();
      times.push(performance.now() - t0);
      for (const t of Object.values(o)) t.dispose();
    }
    result.runMs = stats(times);
    const best = stats(times).median;
    result.gflops = entry.flops / (best / 1000) / 1e9;

    // criterion 3: the block chained 22 times, x stays on the GPU between runs
    if (entry.kind === "block") {
      const chains: number[] = [];
      for (let c = 0; c < opts.chainRuns; c++) {
        let x = feeds.x as ort.Tensor;
        const owned: ort.Tensor[] = [];
        const t0 = performance.now();
        for (let i = 0; i < 22; i++) {
          const o = await session.run({ ...feeds, x });
          x = o.out ?? (Object.values(o)[0] as ort.Tensor);
          owned.push(x);
        }
        await device.queue.onSubmittedWorkDone();
        chains.push(performance.now() - t0);
        const last = (await x.getData(false)) as Float32Array;
        result.chainFinite = last.every((v) => Number.isFinite(v));
        for (const t of owned) t.dispose();
      }
      result.chain22Ms = stats(chains);
      const stepMs = stats(chains).median;
      // a chunk of n frames after a 5.2 s reference; 32 NFE steps, the block stack only
      const newAudioS = (entry.n * 256) / 24000 - 5.2;
      result.projected = {
        stepMs,
        genS: (stepMs * 32) / 1000,
        newAudioS,
        rtf: newAudioS > 0 ? (stepMs * 32) / 1000 / newAudioS : null,
      };
    }
    out.dispose();
    for (const t of Object.values(feeds)) t.dispose();
  } finally {
    await session.release();
  }
  return result;
}

async function runSpike(options: SpikeOptions = {}): Promise<Record<string, unknown>> {
  const opts: Required<SpikeOptions> = {
    models: options.models ?? [],
    fp16: options.fp16 ?? true,
    warmup: options.warmup ?? 3,
    runs: options.runs ?? 20,
    chainRuns: options.chainRuns ?? 5,
    verbose: options.verbose ?? true,
  };
  ort.env.logLevel = opts.verbose ? "verbose" : "warning";
  const { device, adapterInfo } = await ortDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e) => errors.push(String((e as GPUUncapturedErrorEvent).error.message)));

  const manifest = (await (await fetch(`${DATA}manifest.json`)).json()) as { models: ModelEntry[] };
  const entries = manifest.models.filter((m) => opts.models.length === 0 || opts.models.includes(m.name));
  const results: Record<string, unknown>[] = [];
  for (const entry of entries) {
    const files = opts.fp16 ? [`${entry.name}.onnx`, `${entry.name}_fp16.onnx`] : [`${entry.name}.onnx`];
    for (const file of files) {
      try {
        results.push(await runModel(device, entry, file, opts));
      } catch (e) {
        results.push({ name: entry.name, file, error: String(e) });
      }
    }
    if (entry.kind === "matmul") {
      try {
        const x = await fetchNpy(`${DATA}${entry.name}.in.x.npy`);
        const w = await fetchNpy(`${DATA}matmul_w.npy`);
        const ref = await fetchNpy(`${DATA}${entry.name}.ref.npy`);
        const [m, k] = x.shape as [number, number];
        const n = w.shape[1] as number;
        const r = await runWgslMatmul(device, x.data, w.data, m, k, n, opts.warmup, opts.runs);
        const best = stats(r.timesMs).median;
        results.push({
          name: entry.name,
          file: "wgsl-tiled-matmul",
          kind: "wgsl",
          n: entry.n,
          correctness: compare(r.output, ref.data),
          runMs: stats(r.timesMs),
          gflops: entry.flops / (best / 1000) / 1e9,
        });
      } catch (e) {
        results.push({ name: entry.name, file: "wgsl-tiled-matmul", error: String(e) });
      }
    }
  }
  return {
    adapter: adapterInfo,
    userAgent: navigator.userAgent,
    ortVersion: ort.env.versions,
    results,
    gpuErrors: errors,
  };
}

declare global {
  interface Window {
    runSpike: typeof runSpike;
    spikeLog: () => string[];
  }
}
window.runSpike = runSpike;
window.spikeLog = () => logLines;
document.body.dataset.ready = "1";
