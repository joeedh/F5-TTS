// Usage: node probe.cjs [variant]   variants: default | swiftshader | vulkan
// Launches Chromium with WebGPU flags, reports adapter info, features, limits,
// and runs two compute shaders (vector add, 64x64 matmul, f16 if available).
const { chromium } = require("playwright");
const http = require("http");

const VARIANTS = {
  default: ["--enable-unsafe-webgpu", "--enable-features=WebGPU"],
  swiftshader: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ],
  vulkan: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-vulkan=native",
    "--use-angle=vulkan",
    "--ignore-gpu-blocklist",
  ],
};

async function probeInPage() {
  const out = { hasNavigatorGpu: !!navigator.gpu };
  if (!navigator.gpu) return out;
  const adapter = await navigator.gpu.requestAdapter();
  out.adapterFound = !!adapter;
  if (!adapter) return out;
  const info = adapter.info || (adapter.requestAdapterInfo && (await adapter.requestAdapterInfo())) || {};
  out.info = {
    vendor: info.vendor, architecture: info.architecture, device: info.device,
    description: info.description, subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize,
  };
  out.isFallbackAdapter = adapter.isFallbackAdapter;
  out.features = [...adapter.features].sort();
  const lim = {};
  for (const k in adapter.limits) lim[k] = adapter.limits[k];
  out.limits = lim;

  const wantFeatures = ["shader-f16", "subgroups"].filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures: wantFeatures, requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  } });
  out.deviceFeatures = [...device.features].sort();
  out.deviceMaxBufferSize = device.limits.maxBufferSize;
  out.deviceMaxStorageBufferBindingSize = device.limits.maxStorageBufferBindingSize;
  const errors = [];
  device.addEventListener("uncapturederror", (e) => errors.push(String(e.error.message)));

  async function run(code, inputs, outLen, wg) {
    const mk = (data, usage) => {
      const b = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
      new Float32Array(b.getMappedRange()).set(data); b.unmap(); return b;
    };
    const bufs = inputs.map((d) => mk(d, GPUBufferUsage.STORAGE));
    const outBuf = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const module = device.createShaderModule({ code });
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [...bufs, outBuf].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...wg); pass.end();
    enc.copyBufferToBuffer(outBuf, 0, read, 0, outLen * 4);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
    return { res, ms: performance.now() - t0 };
  }

  // 1. vector add
  const N = 1 << 16;
  const a = Float32Array.from({ length: N }, (_, i) => i), b = Float32Array.from({ length: N }, (_, i) => 2 * i);
  const add = await run(`
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> o: array<f32>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { o[g.x] = a[g.x] + b[g.x]; }`,
    [a, b], N, [N / 64, 1, 1]);
  let addOk = true; for (let i = 0; i < N; i++) if (add.res[i] !== 3 * i) { addOk = false; break; }
  out.vectorAdd = { ok: addOk, ms: add.ms };

  // 2. matmul M=K=N=256 (naive), compare to CPU
  const M = 256;
  const A = Float32Array.from({ length: M * M }, (_, i) => ((i * 7) % 13) / 13);
  const B = Float32Array.from({ length: M * M }, (_, i) => ((i * 5) % 11) / 11);
  const mm = await run(`
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;
    @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let n = ${M}u; var s = 0.0;
      for (var k = 0u; k < n; k++) { s += A[g.y * n + k] * B[k * n + g.x]; }
      C[g.y * n + g.x] = s; }`, [A, B], M * M, [M / 8, M / 8, 1]);
  let maxErr = 0;
  for (let y = 0; y < M; y += 17) for (let x = 0; x < M; x += 19) {
    let s = 0; for (let k = 0; k < M; k++) s += A[y * M + k] * B[k * M + x];
    maxErr = Math.max(maxErr, Math.abs(s - mm.res[y * M + x]));
  }
  out.matmul256 = { maxAbsErr: maxErr, ok: maxErr < 1e-2, ms: mm.ms };

  // 2b. rough throughput: naive 1024^3 matmul, 2 timed runs (not a tuned kernel, only a lower bound)
  {
    const S = 1024;
    const A2 = new Float32Array(S * S).fill(0.5), B2 = new Float32Array(S * S).fill(0.25);
    const code = `
      @group(0) @binding(0) var<storage, read> A: array<f32>;
      @group(0) @binding(1) var<storage, read> B: array<f32>;
      @group(0) @binding(2) var<storage, read_write> C: array<f32>;
      @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let n = ${S}u; var s = 0.0;
        for (var k = 0u; k < n; k++) { s += A[g.y * n + k] * B[k * n + g.x]; }
        C[g.y * n + g.x] = s; }`;
    const r1 = await run(code, [A2, B2], S * S, [S / 8, S / 8, 1]);
    const r2 = await run(code, [A2, B2], S * S, [S / 8, S / 8, 1]);
    const ms = Math.min(r1.ms, r2.ms);
    out.matmul1024 = { ok: Math.abs(r2.res[0] - S * 0.125) < 1e-3, ms, gflops: (2 * S ** 3) / (ms / 1000) / 1e9 };
  }

  // 3. shader-f16 compile check
  if (device.features.has("shader-f16")) {
    try {
      const m = device.createShaderModule({ code: `enable f16;
        @group(0) @binding(0) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(1) fn main() { let h: f16 = 1.5h; o[0] = f32(h * 2.0h); }` });
      const info2 = await m.getCompilationInfo();
      out.f16Shader = { ok: !info2.messages.some((x) => x.type === "error") };
    } catch (e) { out.f16Shader = { ok: false, error: String(e) }; }
  } else out.f16Shader = { ok: false, error: "feature not available" };

  // 4. biggest single buffer we can allocate and touch
  const tries = [128, 256, 512, 1024, 2048].map((m) => m * 1024 * 1024);
  out.bufferAlloc = [];
  for (const size of tries) {
    if (size > device.limits.maxBufferSize) { out.bufferAlloc.push({ MB: size / 1048576, ok: false, why: "over maxBufferSize" }); continue; }
    device.pushErrorScope("out-of-memory");
    const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
    const err = await device.popErrorScope();
    out.bufferAlloc.push({ MB: size / 1048576, ok: !err });
    buf.destroy();
  }
  out.uncapturedErrors = errors;
  return out;
}

(async () => {
  const variant = process.argv[2] || "default";
  const args = VARIANTS[variant];
  if (!args) { console.error("unknown variant", variant); process.exit(2); }
  // WebGPU needs a secure context; http://localhost qualifies, about:blank does not.
  const server = http.createServer((q, r) => { r.setHeader("content-type", "text/html"); r.end("<!doctype html><title>probe</title>"); });
  await new Promise((res) => server.listen(0, "127.0.0.1", res));
  const url = `http://localhost:${server.address().port}/`;
  // channel "chromium" selects the full new-headless build; the default headless shell has no WebGPU.
  const browser = await chromium.launch({ args, headless: true, channel: process.env.PROBE_CHANNEL || "chromium" });
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(m.text()));
  await page.goto(url);
  let result;
  try { result = await page.evaluate(probeInPage); } catch (e) { result = { error: String(e) }; }
  result.variant = variant;
  result.browser = browser.version();
  result.chromeGpuLogs = logs.slice(0, 5);
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
  server.close();
})();
