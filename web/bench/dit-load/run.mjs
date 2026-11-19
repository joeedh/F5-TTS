// Plan 01 step 2: loads the real fp32 DiT graphs in Chrome and records time and memory per scenario.
// Usage (from web/): node bench/dit-load/run.mjs --gpu nvidia|intel [--channel chrome] [--attention unfused]
//   [--case en-short] [--scenarios url,url,cache,cache] [--no-text] [--tag name]
// Each scenario runs in a new page of one persistent profile, so the second "url" run can use the HTTP
// cache and the second "cache" run reads Cache Storage. Memory is sampled from outside (sample-memory.ps1).
import { spawn } from "node:child_process";
import { createReadStream, mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, normalize } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "../../..");
const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const channel = arg("channel", "chrome");
const gpu = arg("gpu", "nvidia");
const attention = arg("attention", "unfused");
const caseName = arg("case", "en-short");
const scenarios = arg("scenarios", "url,url,cache,cache").split(",");
const withText = !argv.includes("--no-text");
const tag = arg("tag", `${channel}-${gpu}-${attention}`);
const roots = { "/model/": join(repo, "build/webgpu/dit/base"), "/golden/": join(repo, "testdata/webgpu") };

// Serves the model and golden folders with a length and a cache lifetime, so the HTTP cache may keep them
const serveData = {
  name: "serve-data",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const url = decodeURIComponent((req.url ?? "").split("?")[0]);
      const prefix = Object.keys(roots).find((p) => url.startsWith(p));
      if (!prefix) return next();
      const file = normalize(join(roots[prefix], url.slice(prefix.length)));
      if (!file.startsWith(roots[prefix])) return next();
      let size;
      try {
        size = statSync(file).size;
      } catch {
        res.statusCode = 404;
        return res.end();
      }
      res.setHeader("Content-Length", size);
      res.setHeader("Content-Type", "application/octet-stream");
      res.setHeader("Cache-Control", "max-age=86400");
      createReadStream(file).pipe(res);
    });
  },
};

const server = await createServer({
  root: here,
  configFile: false,
  logLevel: "warn",
  plugins: [serveData],
  server: { port: 0, host: "127.0.0.1", forwardConsole: false, fs: { allow: [join(here, "../..")] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
await server.listen();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error("vite did not start");

const profile = join(repo, "build/webgpu/dit-load-profile", tag);
rmSync(profile, { recursive: true, force: true });
mkdirSync(profile, { recursive: true });

const samples = [];
const sampler = spawn("powershell.exe", ["-NoProfile", "-File", join(here, "sample-memory.ps1"), "-Profile", profile]);
createInterface({ input: sampler.stdout }).on("line", (line) => {
  try {
    samples.push(JSON.parse(line));
  } catch {
    // not a sample
  }
});

const flags = ["--enable-unsafe-webgpu"];
if (gpu === "nvidia") flags.push("--force_high_performance_gpu");
const context = await chromium.launchPersistentContext(profile, {
  channel,
  headless: process.env.SPIKE_HEADED !== "1",
  args: flags,
});

const GB = 1024 ** 3;
const peak = (from, to) => {
  const inside = samples.filter((s) => s.t >= from && s.t <= to);
  const before = samples.filter((s) => s.t < from).at(-1);
  const max = (k) => Math.max(...inside.map((s) => s[k]));
  const gb = (x) => Math.round((x / GB) * 100) / 100;
  if (!inside.length) return null;
  return {
    samples: inside.length,
    baseline: before ? { privateGB: gb(before.private), gpuDedicatedGB: gb(before.gpuDedicated) } : null,
    privateGB: gb(max("private")),
    maxProcessPrivateGB: gb(max("maxPrivate")),
    rendererGB: gb(max("renderer")),
    gpuProcessGB: gb(max("gpuProcess")),
    gpuDedicatedGB: gb(max("gpuDedicated")),
    gpuSharedGB: gb(max("gpuShared")),
  };
};

const runs = [];
for (const [i, source] of scenarios.entries()) {
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  await page.goto(url);
  await page.waitForSelector("body[data-ready='1']", { timeout: 60_000 });
  await new Promise((r) => setTimeout(r, 1500)); // a baseline sample before the load
  const from = Date.now();
  let result;
  try {
    result = await page.evaluate((opts) => window.runLoad(opts), { caseName, attention, source, withText });
  } catch (e) {
    result = { source, error: String(e) };
  }
  await new Promise((r) => setTimeout(r, 1000));
  const m = result.marks ?? {};
  result.memory = peak(from, Date.now());
  result.memoryLoad = m.stepCreated ? peak(m.start, m.stepCreated) : null;
  result.memoryRun = m.ran ? peak(m.stepCreated, m.ran) : null;
  result.pageErrors = pageErrors;
  result.scenario = `${i}:${source}`;
  runs.push(result);
  await page.close();
  const c = result.comparisons ?? [];
  console.log(
    `${result.scenario.padEnd(8)} ` +
      (result.error
        ? `ERROR ${result.error}`
        : `load ${result.loadTotalMs} ms (step create ${result[`step${attention === "unfused" ? "" : `_${attention}`}.createMs`]} ms)` +
          ` run ${result.runMs.join("/")} ms  pass ${c.filter((x) => x.passed).length}/${c.length}` +
          [
            ["load", result.memoryLoad],
            ["run", result.memoryRun],
          ]
            .map(
              ([k, p]) =>
                `\n   ${k}: private ${p?.privateGB} GB (renderer ${p?.rendererGB}, gpu process ${p?.gpuProcessGB})` +
                ` gpu ${p?.gpuDedicatedGB} + shared ${p?.gpuSharedGB} GB`,
            )
            .join("")),
  );
}

const out = { channel, gpuFlag: gpu, browserVersion: context.browser()?.version() ?? null, attention, caseName, runs };
const outDir = join(here, "results");
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, `${tag}.json`), JSON.stringify(out, null, 2));
await context.close();
sampler.kill();
await server.close();
