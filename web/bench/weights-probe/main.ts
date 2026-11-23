// Plan 05 step 1: what ORT Web's WebGPU kernels do in fp16 and with MatMulNBits, what device limits ORT
// asks for, how Hugging Face downloads behave from a page, and how fast OPFS writes and reads. run.mjs drives
// the window.probe* functions; the graphs come from tools/webgpu/probe_kernels.py.
import * as ort from "onnxruntime-web/webgpu";

const PROBES = "/probes/";

// Wrapped before the first session, so ORT's own device requests (one per device it creates) go through it.
const requested: { features: string[]; limits: Record<string, number> }[] = [];
const adapterLimits: Record<string, number>[] = [];
const realRequestDevice = GPUAdapter.prototype.requestDevice;
GPUAdapter.prototype.requestDevice = function (descriptor?: GPUDeviceDescriptor) {
  const limits: Record<string, number> = {};
  for (const [k, v] of Object.entries(descriptor?.requiredLimits ?? {})) if (typeof v === "number") limits[k] = v;
  const entry = { features: [...(descriptor?.requiredFeatures ?? [])].map(String), limits };
  if (!requested.some((r) => JSON.stringify(r) === JSON.stringify(entry))) requested.push(entry);
  if (!adapterLimits.length) {
    const own: Record<string, number> = {};
    for (const k in this.limits) own[k] = (this.limits as unknown as Record<string, number>)[k] as number;
    adapterLimits.push(own);
  }
  return realRequestDevice.call(this, descriptor);
};

interface ProbeEntry {
  name: string;
  fp16: boolean;
  note: string;
  outShape: number[];
  inputs: { name: string; shape: number[] }[];
}

async function bytes(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.arrayBuffer();
}

/** The fp16 spacing at v: 2^(exponent - 10), and 2^-24 among the subnormals. */
function ulp16(v: number): number {
  const a = Math.abs(v);
  return a < 2 ** -14 ? 2 ** -24 : 2 ** (Math.floor(Math.log2(a)) - 10);
}

function relL2(y: ArrayLike<number>, ref: Float64Array): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < ref.length; i++) {
    num += ((y[i] as number) - (ref[i] as number)) ** 2;
    den += (ref[i] as number) ** 2;
  }
  return Math.sqrt(num / den);
}

function measure(y: Float32Array, ref: Float64Array, ideal: Float32Array, fp16: boolean) {
  let maxAbs = 0;
  let exact = 0;
  let onGrid = 0;
  let finite = true;
  for (let i = 0; i < y.length; i++) {
    const v = y[i] as number;
    if (!Number.isFinite(v)) finite = false;
    maxAbs = Math.max(maxAbs, Math.abs(v - (ref[i] as number)));
    if (fp16 && v === ideal[i]) exact++;
    if (fp16 && Number.isInteger(v / ulp16(v))) onGrid++;
  }
  const err = relL2(y, ref);
  const out: Record<string, number | boolean> = { relL2: err, maxAbs, finite };
  if (fp16) {
    // the error of rounding the exact result once to fp16: what an fp32-accumulating kernel would reach
    const floor = relL2(ideal, ref);
    Object.assign(out, {
      exactFraction: exact / y.length,
      onGridFraction: onGrid / y.length,
      floor,
      timesFloor: err / floor,
    });
  }
  return out;
}

async function probeKernels(options: { optimization?: "all" | "disabled"; only?: string[] } = {}) {
  const index = (await (await fetch(`${PROBES}probes.json`)).json()) as ProbeEntry[];
  const results = [];
  for (const p of index) {
    if (options.only && !options.only.includes(p.name)) continue;
    const r: Record<string, unknown> = { name: p.name };
    console.log(`[probe] ${p.name}`); // run.mjs files ORT's placement lines under the last marker
    try {
      const [graph, input, ref, ideal] = await Promise.all(
        ["onnx", "in.bin", "ref.bin", "ideal.bin"].map((ext) => bytes(`${PROBES}${p.name}.${ext}`)),
      );
      const session = await ort.InferenceSession.create(graph!, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: options.optimization ?? "all",
        logSeverityLevel: 0,
      });
      const all = new Float32Array(input!);
      const feeds: Record<string, ort.Tensor> = {};
      let offset = 0;
      for (const i of p.inputs) {
        const size = i.shape.reduce((a, b) => a * b, 1);
        feeds[i.name] = new ort.Tensor("float32", all.slice(offset, offset + size), i.shape);
        offset += size;
      }
      await session.run(feeds); // the first run includes shader compilation
      const t0 = performance.now();
      const out = await session.run(feeds);
      const y = (await out.y!.getData()) as Float32Array;
      r.ms = Math.round((performance.now() - t0) * 10) / 10;
      Object.assign(r, measure(y, new Float64Array(ref!), new Float32Array(ideal!), p.fp16));
      await session.release();
    } catch (e) {
      r.error = String(e).slice(0, 300);
    }
    results.push(r);
  }
  const device = await ort.env.webgpu.device;
  return {
    adapter: { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture },
    requested,
    adapterLimits: adapterLimits[0] ?? null,
    ortVersion: ort.env.versions,
    results,
  };
}

/** One request, reading only the headers and the first chunk of the body. */
async function headOf(url: string, init: RequestInit = {}) {
  const t0 = performance.now();
  try {
    const res = await fetch(url, init);
    const headers: Record<string, string> = {};
    res.headers.forEach((v, k) => (headers[k] = v));
    const reader = res.body?.getReader();
    const first = reader ? await reader.read() : null;
    await reader?.cancel();
    return {
      ok: res.ok,
      status: res.status,
      redirected: res.redirected,
      finalHost: new URL(res.url).host,
      headers,
      firstChunkBytes: first?.value?.byteLength ?? 0,
      ms: Math.round(performance.now() - t0),
    };
  } catch (e) {
    return { error: String(e), ms: Math.round(performance.now() - t0) };
  }
}

async function probeHF() {
  const base = "https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/";
  return {
    smallPlain: await headOf(`${base}config.yaml`, { cache: "no-store" }),
    largeGet: await headOf(`${base}pytorch_model.bin`, { cache: "no-store" }),
    largeRange: await headOf(`${base}pytorch_model.bin`, { cache: "no-store", headers: { Range: "bytes=0-1023" } }),
    largeHead: await headOf(`${base}pytorch_model.bin`, { method: "HEAD", cache: "no-store" }),
  };
}

const SHARD = 64 * 1024 * 1024;

async function probeOPFS(options: { shards?: number } = {}) {
  const shards = options.shards ?? 21; // 1.31 GiB, about the step graph's data
  const before = await navigator.storage.estimate();
  const persisted = await navigator.storage.persisted();
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle("probe", { create: true });
  const data = new Uint8Array(SHARD);
  for (let i = 0; i < data.length; i += 4096) data[i] = i & 0xff;
  const result: Record<string, unknown> = {
    quotaGB: (before.quota ?? 0) / 1024 ** 3,
    usageGB: (before.usage ?? 0) / 1024 ** 3,
    persisted,
  };

  // Main thread: createWritable, then a rename into place with move().
  let t0 = performance.now();
  for (let i = 0; i < shards; i++) {
    const tmp = await dir.getFileHandle(`w${i}.part`, { create: true });
    const w = await tmp.createWritable();
    await w.write(data);
    await w.close();
    const moveable = tmp as FileSystemFileHandle & { move?: (name: string) => Promise<void> };
    if (moveable.move) await moveable.move(`w${i}`);
    else result.moveMissing = true;
  }
  result.writableMs = Math.round(performance.now() - t0);

  t0 = performance.now();
  let total = 0;
  for (let i = 0; i < shards; i++) {
    const f = await (await dir.getFileHandle(result.moveMissing ? `w${i}.part` : `w${i}`)).getFile();
    total += (await f.arrayBuffer()).byteLength;
  }
  result.readMs = Math.round(performance.now() - t0);
  result.readGB = total / 1024 ** 3;

  // A worker: createSyncAccessHandle.
  const worker = new Worker(new URL("./opfs-worker.ts", import.meta.url), { type: "module" });
  result.syncHandle = await new Promise((resolve) => {
    worker.onmessage = (e) => resolve(e.data);
    worker.onerror = (e) => resolve({ error: e.message });
    worker.postMessage({ shards, size: SHARD });
  });
  worker.terminate();

  result.estimateAfterWriteGB = ((await navigator.storage.estimate()).usage ?? 0) / 1024 ** 3;
  await root.removeEntry("probe", { recursive: true });
  return result;
}

declare global {
  interface Window {
    probeKernels: typeof probeKernels;
    probeHF: typeof probeHF;
    probeOPFS: typeof probeOPFS;
  }
}
window.probeKernels = probeKernels;
window.probeHF = probeHF;
window.probeOPFS = probeOPFS;
ort.env.logLevel = "verbose";
document.body.dataset.ready = "1";
