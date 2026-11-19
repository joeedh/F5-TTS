// Plan 01 step 4: times the full DiT step graph on WebGPU, as the plan 00 spike timed one block. Inputs and
// outputs stay on the GPU; a chain copies x_next back into x on the GPU between steps. run.mjs drives
// window.runStepBench() in a fresh page per configuration. Plan 07 extends this page.
import * as ort from "onnxruntime-web/webgpu";

export interface StepBenchOptions {
  /** File under `base`, such as "step_mha.onnx" or "step_mha_n1024.onnx". */
  graph: string;
  /** Model folder URL; default "/model/dit/base/". */
  base?: string;
  /** Text embedding width; 512 for the base model. */
  textDim?: number;
  /** Sequence lengths to run, in order. A static graph takes only its own n. */
  ns: number[];
  graphCapture?: boolean;
  warmup?: number;
  runs?: number;
  chains?: number;
  chainSteps?: number;
}

function stats(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  const r = (x: number | undefined) => Math.round((x ?? NaN) * 10) / 10;
  return { median: r(s[Math.floor(s.length / 2)]), min: r(s[0]), max: r(s[s.length - 1]) };
}

function gpuBuffer(device: GPUDevice, bytes: number): GPUBuffer {
  return device.createBuffer({
    size: Math.max(16, Math.ceil(bytes / 16) * 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

async function download(device: GPUDevice, buf: GPUBuffer, bytes: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: buf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, buf.size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0, bytes));
  staging.destroy();
  return out;
}

function gpuTensor(device: GPUDevice, data: Float32Array, dims: number[]): ort.Tensor {
  const buf = gpuBuffer(device, data.byteLength);
  device.queue.writeBuffer(buf, 0, data);
  return ort.Tensor.fromGpuBuffer(buf, {
    dataType: "float32",
    dims,
    download: () => download(device, buf, data.byteLength),
    dispose: () => buf.destroy(),
  });
}

function random(count: number, scale = 1): Float32Array {
  const a = new Float32Array(count);
  for (let i = 0; i < count; i++) a[i] = (Math.random() * 2 - 1) * scale;
  return a;
}

/** firstMs is the first call at this n: shader and pipeline creation for a new n, or the cold start. */
async function benchN(
  session: ort.InferenceSession,
  device: GPUDevice,
  n: number,
  opts: Required<Omit<StepBenchOptions, "graph" | "ns" | "base">>,
) {
  const feeds: Record<string, ort.Tensor> = {
    x: gpuTensor(device, random(n * 100), [1, n, 100]),
    step_cond: gpuTensor(device, random(n * 100), [1, n, 100]),
    text_cond: gpuTensor(device, random(n * opts.textDim), [1, n, opts.textDim]),
    text_uncond: gpuTensor(device, random(n * opts.textDim), [1, n, opts.textDim]),
    time: gpuTensor(device, new Float32Array([0.3]), [1]),
    cfg_strength: gpuTensor(device, new Float32Array([2]), [1]),
    dt: gpuTensor(device, new Float32Array([1 / 32]), [1]),
  };
  // with graph capture, the same input buffers go in on every run and ORT owns and reuses the outputs
  const run = async () => session.run(feeds);
  const xBuf = feeds.x!.gpuBuffer;
  const copyBack = (out: ort.InferenceSession.OnnxValueMapType) => {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer((out.x_next as ort.Tensor).gpuBuffer, 0, xBuf, 0, n * 100 * 4);
    device.queue.submit([enc.finish()]);
  };
  const release = (out: ort.InferenceSession.OnnxValueMapType) => {
    if (!opts.graphCapture) for (const t of Object.values(out)) (t as ort.Tensor).dispose();
  };

  const t0 = performance.now();
  release(await run());
  await device.queue.onSubmittedWorkDone();
  const firstMs = Math.round(performance.now() - t0);
  for (let i = 1; i < opts.warmup; i++) release(await run());
  await device.queue.onSubmittedWorkDone();

  const single: number[] = [];
  for (let i = 0; i < opts.runs; i++) {
    const s = performance.now();
    const out = await run();
    await device.queue.onSubmittedWorkDone();
    single.push(performance.now() - s);
    release(out);
  }
  const chains: number[] = [];
  let finite = true;
  for (let c = 0; c < opts.chains; c++) {
    const s = performance.now();
    for (let k = 0; k < opts.chainSteps; k++) {
      const out = await run();
      copyBack(out);
      release(out);
    }
    await device.queue.onSubmittedWorkDone();
    chains.push(performance.now() - s);
  }
  const x = (await feeds.x!.getData()) as Float32Array;
  finite = x.every((v) => Number.isFinite(v));
  for (const t of Object.values(feeds)) t.dispose();
  const chain = stats(chains);
  return {
    n,
    firstMs,
    singleMs: stats(single),
    chainMs: chain,
    perStepInChainMs: Math.round(((chain.median ?? NaN) / opts.chainSteps) * 10) / 10,
    chainFinite: finite,
  };
}

async function runStepBench(options: StepBenchOptions): Promise<Record<string, unknown>> {
  const opts = {
    graphCapture: options.graphCapture ?? false,
    warmup: options.warmup ?? 3,
    runs: options.runs ?? 20,
    chains: options.chains ?? 5,
    chainSteps: options.chainSteps ?? 32,
    textDim: options.textDim ?? 512,
  };
  const BASE = options.base ?? "/model/dit/base/";
  const errors: string[] = [];
  const graph = new Uint8Array(await (await fetch(BASE + options.graph)).arrayBuffer());
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: ["webgpu"],
    preferredOutputLocation: "gpu-buffer",
    enableGraphCapture: opts.graphCapture,
    externalData: [{ path: `${options.graph}.data`, data: `${BASE}${options.graph}.data` }],
  });
  const createMs = Math.round(performance.now() - t0);
  const device = await ort.env.webgpu.device;
  device.addEventListener("uncapturederror", (e) => errors.push((e as GPUUncapturedErrorEvent).error.message));
  const perN: Record<string, unknown>[] = [];
  try {
    for (const n of options.ns) perN.push(await benchN(session, device, n, opts));
  } finally {
    await session.release();
  }
  return {
    graph: options.graph,
    graphCapture: opts.graphCapture,
    adapter: { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture },
    userAgent: navigator.userAgent,
    ortVersion: ort.env.versions,
    createMs,
    perN,
    gpuErrors: errors,
  };
}

declare global {
  interface Window {
    runStepBench: typeof runStepBench;
  }
}
window.runStepBench = runStepBench;
document.body.dataset.ready = "1";
