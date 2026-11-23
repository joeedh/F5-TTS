// Plan 01 step 2: can the fp32 DiT graphs load and run in the browser? run.mjs drives window.runLoad()
// in a fresh page per scenario and samples process and GPU memory from outside the browser.
import * as ort from "onnxruntime-web/webgpu";
import { compare, httpReader, loadCase, loadTolerances, type Result, type Tensor } from "../../test/parity";

export interface LoadOptions {
  caseName?: string;
  attention?: "unfused" | "mha" | "onnx23";
  /**
   * "url": ORT fetches the external data itself. "cache": read from Cache Storage (filled on a miss).
   * "opfs-blob" and "opfs-bytes" (plan 05): stored in OPFS (filled on a miss by streaming the response), then
   * handed to ORT as the File itself, or as bytes read whole by a worker's sync access handle.
   */
  source?: Source;
  withText?: boolean;
  runs?: number;
}

type Source = "url" | "cache" | "opfs-blob" | "opfs-bytes";

const MODEL = "/model/";
const OPFS_DIR = "dit-weights";
const CACHE = "dit-weights";

const ms = (t0: number) => Math.round(performance.now() - t0);

function transfer(url: string): { transferSize: number; encodedBodySize: number } | null {
  const e = performance.getEntriesByName(new URL(url, location.href).href).at(-1) as
    PerformanceResourceTiming | undefined;
  return e ? { transferSize: e.transferSize, encodedBodySize: e.encodedBodySize } : null;
}

/** Returns the external data to hand ORT, and how long getting it took when we fetch it ourselves. */
async function opfsData(file: string, source: Source, info: Record<string, unknown>) {
  const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(OPFS_DIR, { create: true });
  let handle: FileSystemFileHandle;
  try {
    handle = await dir.getFileHandle(file);
    info[`${file}.opfsHit`] = true;
  } catch {
    info[`${file}.opfsHit`] = false;
    const t0 = performance.now();
    const res = await fetch(MODEL + file, { cache: "no-store" });
    handle = await dir.getFileHandle(file, { create: true });
    await res.body!.pipeTo(await handle.createWritable());
    info[`${file}.opfsFillMs`] = ms(t0);
  }
  const t0 = performance.now();
  if (source === "opfs-blob") {
    const blob = await handle.getFile();
    info[`${file}.opfsOpenMs`] = ms(t0);
    return [{ path: file, data: blob }];
  }
  const worker = new Worker(new URL("./opfs-worker.ts", import.meta.url), { type: "module" });
  const reply = await new Promise<{ buffer?: ArrayBuffer; error?: string }>((resolve) => {
    worker.onmessage = (e) => resolve(e.data);
    worker.postMessage({ dir: OPFS_DIR, name: file });
  });
  worker.terminate();
  if (!reply.buffer) throw new Error(`${file}: ${reply.error}`);
  info[`${file}.opfsReadMs`] = ms(t0);
  return [{ path: file, data: new Uint8Array(reply.buffer) }];
}

async function externalData(file: string, source: Source, info: Record<string, unknown>) {
  const url = MODEL + file;
  if (source === "url") return [{ path: file, data: url }];
  if (source === "opfs-blob" || source === "opfs-bytes") return opfsData(file, source, info);
  const cache = await caches.open(CACHE);
  let hit = await cache.match(url);
  info[`${file}.cacheHit`] = Boolean(hit);
  if (!hit) {
    const t0 = performance.now();
    await cache.put(url, await fetch(url));
    info[`${file}.cacheFillMs`] = ms(t0);
    hit = await cache.match(url);
  }
  if (!hit) throw new Error(`${url}: not in Cache Storage after put`);
  const t0 = performance.now();
  const bytes = new Uint8Array(await hit.arrayBuffer());
  info[`${file}.cacheReadMs`] = ms(t0);
  return [{ path: file, data: bytes }];
}

async function createSession(stem: string, source: Source, info: Record<string, unknown>) {
  const graph = await (await fetch(`${MODEL}${stem}.onnx`)).arrayBuffer();
  const data = await externalData(`${stem}.onnx.data`, source, info);
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: ["webgpu"],
    preferredOutputLocation: "gpu-buffer",
    externalData: data,
  });
  info[`${stem}.createMs`] = ms(t0);
  if (source === "url") info[`${stem}.transfer`] = transfer(MODEL + `${stem}.onnx.data`);
  return session;
}

function cpu(t: Tensor): ort.Tensor {
  return new ort.Tensor(t.dtype, t.data, t.shape);
}

async function runLoad(options: LoadOptions = {}): Promise<Record<string, unknown>> {
  const caseName = options.caseName ?? "en-short";
  const suffix = (options.attention ?? "unfused") === "unfused" ? "" : `_${options.attention}`;
  const source = options.source ?? "url";
  const runs = options.runs ?? 3;
  const info: Record<string, unknown> = { caseName, attention: options.attention ?? "unfused", source };
  const marks: Record<string, number> = {};
  const mark = (name: string) => (marks[name] = Date.now());
  const errors: string[] = [];

  const read = httpReader("/golden");
  const golden = await loadCase(read, caseName);
  const tolerances = await loadTolerances(read);
  const chunk = golden.manifest.chunks[0] as { duration: number };
  const n = chunk.duration;
  const cfg = golden.manifest.settings.cfg_strength as number;
  const grid = (await golden.tensor("c0.plan.t")).data;

  const results: Result[] = [];
  const t0 = performance.now();
  mark("start");
  let textCond: ort.Tensor;
  let textUncond: ort.Tensor;
  let textSession: ort.InferenceSession | undefined;
  if (options.withText ?? true) {
    textSession = await createSession("text", source, info);
    mark("textCreated");
    const ids = await golden.tensor("c0.dit.text_ids");
    const padded = new Int32Array(n).fill(-1);
    padded.set(ids.data.subarray(0, Math.min(n, ids.data.length)));
    const out = await textSession.run({ text: new ort.Tensor("int32", padded, [1, n]) });
    textCond = out.text_cond!;
    textUncond = out.text_uncond!;
    for (const [name, t] of [
      ["c0.dit_internal.text_embed", textCond],
      ["c0.dit_internal.text_embed_uncond", textUncond],
    ] as const) {
      const data = (await t.getData()) as Float32Array;
      results.push(...(await compare(golden, tolerances, name, { dtype: "float32", shape: [...t.dims], data })));
    }
  } else {
    textCond = cpu(await golden.tensor("c0.dit_internal.text_embed"));
    textUncond = cpu(await golden.tensor("c0.dit_internal.text_embed_uncond"));
  }

  const stepSession = await createSession(`step${suffix}`, source, info);
  mark("stepCreated");
  info.loadTotalMs = ms(t0);
  const device = await ort.env.webgpu.device;
  device.addEventListener("uncapturederror", (e) => errors.push(String((e as GPUUncapturedErrorEvent).error.message)));
  info.adapter = { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture };

  const feeds: Record<string, ort.Tensor> = {
    x: cpu(await golden.tensor("c0.dit.s00.x")),
    step_cond: cpu(await golden.tensor("c0.dit.step_cond")),
    text_cond: textCond,
    text_uncond: textUncond,
    time: cpu(await golden.tensor("c0.dit.s00.t")),
    cfg_strength: new ort.Tensor("float32", new Float32Array([cfg]), [1]),
    dt: new ort.Tensor("float32", new Float32Array([(grid[1] ?? NaN) - (grid[0] ?? NaN)]), [1]),
  };
  const runMs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const r0 = performance.now();
    const out = await stepSession.run(feeds);
    const pred = (await out.pred!.getData()) as Float32Array; // the readback is part of the timing
    runMs.push(ms(r0));
    if (i === 0) {
      const half = pred.length / 2;
      for (const [name, data] of [
        ["c0.dit.s00.out_cond", pred.subarray(0, half)],
        ["c0.dit.s00.out_uncond", pred.subarray(half)],
      ] as const) {
        results.push(...(await compare(golden, tolerances, name, { dtype: "float32", shape: [1, n, 100], data })));
      }
    }
    for (const t of Object.values(out)) t.dispose();
  }
  mark("ran");
  info.runMs = runMs;
  info.n = n;
  info.comparisons = results.map((r) => ({
    name: r.name,
    passed: r.pass,
    relL2: r.metrics?.relL2 ?? null,
    maxAbs: r.metrics?.maxAbs ?? null,
    error: r.error ?? null,
  }));
  info.passed = results.every((r) => r.pass);
  await stepSession.release();
  await textSession?.release();
  mark("released");
  info.marks = marks;
  info.gpuErrors = errors;
  info.ortVersion = ort.env.versions;
  return info;
}

declare global {
  interface Window {
    runLoad: typeof runLoad;
  }
}
window.runLoad = runLoad;
document.body.dataset.ready = "1";
