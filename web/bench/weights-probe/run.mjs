// Plan 05 step 1: runs the weights probes (main.ts) in Chrome and writes results/<channel>-<gpu>.json.
// Usage (from web/), after tools/webgpu/probe_kernels.py:
//   node bench/weights-probe/run.mjs --gpu nvidia|intel [--channel chrome] [--what kernels,hf,opfs]
// Kernels run twice, with ORT's graph optimizations on and off. ORT's node placement lines are kept per probe.
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";
import { serveData } from "../../serve-data.ts";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "../../..");
const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const channel = arg("channel", "chrome");
const gpu = arg("gpu", "nvidia");
const what = arg("what", "kernels,hf,opfs").split(",");

const server = await createServer({
  root: here,
  configFile: false,
  logLevel: "warn",
  plugins: [serveData({ "/probes/": join(repo, "build/webgpu/probes") })],
  server: { port: 0, host: "127.0.0.1", forwardConsole: false, fs: { allow: [join(here, "../..")] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
await server.listen();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error("vite did not start");

const flags = ["--enable-unsafe-webgpu"];
if (gpu === "nvidia") flags.push("--force_high_performance_gpu");
// A persistent profile: a plain launch gives an in-memory, 10 GB OPFS that says nothing about a real one.
const profile = join(repo, "build/webgpu/weights-probe-profile");
rmSync(profile, { recursive: true, force: true });
const context = await chromium.launchPersistentContext(profile, {
  channel,
  headless: process.env.SPIKE_HEADED !== "1",
  args: flags,
});
const browser = context.browser();
const out = { channel, gpuFlag: gpu, browserVersion: browser?.version() ?? null, date: new Date().toISOString() };

async function page() {
  const p = await context.newPage();
  const lines = { current: "", byProbe: {} };
  p.on("console", (m) => {
    const t = m.text();
    const marker = /^\[probe\] (\S+)/.exec(t);
    if (marker) lines.current = marker[1];
    else if (/placed on|Node placements|CPUExecutionProvider|MemcpyTransformer/.test(t) && lines.current)
      (lines.byProbe[lines.current] ??= []).push(t.replace(/^.*\] /, "").slice(0, 300));
  });
  await p.goto(url);
  await p.waitForSelector("body[data-ready='1']", { timeout: 60_000 });
  return { p, lines };
}

if (what.includes("kernels")) {
  for (const optimization of ["all", "disabled"]) {
    const { p, lines } = await page();
    const r = await p.evaluate((o) => window.probeKernels(o), { optimization });
    for (const x of r.results) x.placement = lines.byProbe[x.name] ?? [];
    out[`kernels_${optimization}`] = r;
    await p.close();
    console.log(`kernels (optimization ${optimization}) on ${r.adapter.vendor} ${r.adapter.architecture}`);
    for (const x of r.results) {
      const f = (v) => (typeof v === "number" ? v.toExponential(2) : v);
      const cpuOnly = !x.placement.some((l) => l.includes("All nodes placed on [WebGpuExecutionProvider]"));
      console.log(
        `  ${x.name.padEnd(22)} ${x.error ? `ERROR ${x.error}` : `relL2 ${f(x.relL2)}${x.timesFloor ? ` = ${x.timesFloor.toFixed(1)} x floor, exact ${x.exactFraction.toFixed(3)}, on fp16 grid ${x.onGridFraction.toFixed(3)}` : ""} ${x.ms} ms`}${cpuOnly ? "  (not all on WebGPU)" : ""}`,
      );
    }
  }
}
if (what.includes("hf")) {
  const { p } = await page();
  out.hf = await p.evaluate(() => window.probeHF());
  await p.close();
  console.log("hf", JSON.stringify(out.hf, null, 1));
}
if (what.includes("opfs")) {
  const { p } = await page();
  out.opfs = await p.evaluate(() => window.probeOPFS());
  await p.close();
  console.log("opfs", JSON.stringify(out.opfs));
}

const dir = join(here, "results");
mkdirSync(dir, { recursive: true });
const file = join(dir, `${channel}-${gpu}.json`);
const earlier = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : {};
writeFileSync(file, JSON.stringify({ ...earlier, ...out }, null, 1) + "\n"); // a partial run (--what) keeps the rest
await context.close();
rmSync(profile, { recursive: true, force: true });
await server.close();
