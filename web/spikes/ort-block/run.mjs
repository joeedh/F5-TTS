// Drives the spike page with Playwright and writes results/<channel>-<gpu>.json.
// Usage (from web/): node spikes/ort-block/run.mjs --channel chrome --gpu nvidia [--models a,b] [--no-fp16]
// --gpu nvidia launches with --force_high_performance_gpu; --gpu intel uses no flag (Windows default).
// See docs/research/webgpu-laptop.md for why.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const channel = arg("channel", "chrome");
const gpu = arg("gpu", "nvidia");
const models = arg("models", "")
  .split(",")
  .filter((s) => s);
const fp16 = !argv.includes("--no-fp16");
const tag = arg("tag", `${channel}-${gpu}`);

const server = await createServer({
  root: here,
  configFile: false,
  logLevel: "warn",
  server: { port: 0, host: "127.0.0.1", forwardConsole: false },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
await server.listen();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error("vite did not start");

const flags = ["--enable-unsafe-webgpu"];
if (gpu === "nvidia") flags.push("--force_high_performance_gpu");
const browser = await chromium.launch({ channel, headless: process.env.SPIKE_HEADED !== "1", args: flags });
const page = await browser.newPage();
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(String(e)));
await page.goto(url);
await page.waitForSelector("body[data-ready='1']", { timeout: 60_000 });

const t0 = Date.now();
const result = await page.evaluate((opts) => window.runSpike(opts), { models, fp16 });
result.channel = channel;
result.gpuFlag = gpu;
result.browserVersion = browser.version();
result.wallS = (Date.now() - t0) / 1000;
result.pageErrors = pageErrors;

const outDir = join(here, "results");
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, `${tag}.json`), JSON.stringify(result, null, 2));
const log = await page.evaluate(() => window.spikeLog());
writeFileSync(join(outDir, `${tag}.log`), log.join("\n"));

for (const r of result.results) {
  const c = r.correctness;
  const line = r.error
    ? `ERROR ${r.error}`
    : `relL2 ${c?.relL2?.toExponential(2)} maxAbs ${c?.maxAbs?.toExponential(2)} nonFinite ${c?.nonFinite}` +
      ` median ${r.runMs?.median?.toFixed(2)} ms ${r.gflops?.toFixed(0)} GFLOPS` +
      (r.projected ? ` chain22 ${r.projected.stepMs.toFixed(1)} ms RTF ${r.projected.rtf?.toFixed(2)}` : "") +
      (r.placementLog?.length ? ` placement-lines ${r.placementLog.length}` : "");
  console.log(`${r.file.padEnd(24)} ${line}`);
}
console.log(
  `adapter ${result.adapter.vendor} ${result.adapter.architecture}; ${result.wallS}s; gpuErrors ${result.gpuErrors.length}`,
);
await browser.close();
await server.close();
