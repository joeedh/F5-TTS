// Plan 05 step 3: loads each weight variant in Chrome (main.ts), one fresh browser each, and records load time,
// one step's time and peak memory. Writes results/chrome-<gpu>.json. Usage (from web/), after
// tools/webgpu/convert_precision.py:
//   node bench/variant-load/run.mjs --gpu nvidia|intel [--variants fp32,w16,w8,fp16,q8,q4] [--case en-short]
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";
import { serveData } from "../../serve-data.ts";
import { startSampler } from "../memory-sampler.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "../../..");
const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const gpu = arg("gpu", "nvidia");
const variants = arg("variants", "fp32,w16,w8,fp16,q8,q4").split(",");
const caseName = arg("case", "en-short");

const server = await createServer({
  root: here,
  configFile: false,
  logLevel: "warn",
  plugins: [
    serveData({ "/variants/": join(repo, "build/webgpu/variants"), "/golden/": join(repo, "testdata/webgpu") }),
  ],
  server: { port: 0, host: "127.0.0.1", forwardConsole: false, fs: { allow: [join(here, "../..")] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
await server.listen();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error("vite did not start");

const flags = ["--enable-unsafe-webgpu"];
if (gpu === "nvidia") flags.push("--force_high_performance_gpu");

// A fresh browser per variant, so no variant's memory counts towards another's peaks.
async function measure(variant) {
  const profile = join(repo, `build/webgpu/variant-load-profile/${gpu}-${variant}`);
  rmSync(profile, { recursive: true, force: true });
  mkdirSync(profile, { recursive: true });
  const sampler = startSampler(profile);
  const context = await chromium.launchPersistentContext(profile, { channel: "chrome", headless: true, args: flags });
  const result = { variant, pageErrors: [] };
  try {
    result.browserVersion = context.browser()?.version() ?? null;
    const page = await context.newPage();
    page.on("pageerror", (e) => result.pageErrors.push(String(e)));
    await page.goto(url);
    await page.waitForSelector("body[data-ready='1']", { timeout: 60_000 });
    result.peaksBefore = await sampler.peaksSoFar();
    Object.assign(result, await page.evaluate((v) => window.loadVariant(v), variant));
    result.peaksLoad = await sampler.peaksSoFar();
    const from = Date.now();
    Object.assign(result, await page.evaluate((c) => window.runStep(c), caseName));
    result.gpuDedicatedRunGB = sampler.peak(from, Date.now())?.gpuDedicatedGB ?? null;
    // the largest live sample of any one process: a check on the peak counters, which must not be below it
    result.maxProcessSampledGB = sampler.peak(0, Date.now())?.maxProcessPrivateGB ?? null;
    result.peaksRun = await sampler.peaksSoFar();
  } catch (e) {
    result.error = String(e);
  }
  await context.close();
  sampler.stop();
  rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 });
  return result;
}

const pair = (p) => (p ? `renderer ${p.rendererGB} + GPU process ${p.gpuProcessGB} GB` : "not measured");
const runs = [];
for (const variant of variants) {
  const r = await measure(variant);
  runs.push(r);
  const f = (x) => (typeof x === "number" ? x.toExponential(1) : x);
  const errors = (r.comparisons ?? []).map((x) => `${x.name.replace(/^c0\.(dit\.s00\.)?/, "")} ${f(x.relL2)}`);
  console.log(
    r.error
      ? `${variant.padEnd(5)} ERROR ${r.error}`
      : `${variant.padEnd(5)} load ${r.loadMs} ms; peaks loading: ${pair(r.peaksLoad)}; running: ${pair(r.peaksRun)};` +
          ` step ${r.stepMs.join("/")} ms; vs fp32 golden ${errors.join(", ")}`,
  );
}

const dir = join(here, "results");
mkdirSync(dir, { recursive: true });
const file = join(dir, `chrome-${gpu}.json`);
// a run of some variants replaces only those
const earlier = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")).runs : [];
const kept = earlier.filter((e) => !runs.some((r) => r.variant === e.variant));
writeFileSync(file, JSON.stringify({ gpuFlag: gpu, runs: [...kept, ...runs] }, null, 1) + "\n");
await server.close();
