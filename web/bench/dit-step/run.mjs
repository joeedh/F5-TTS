// Plan 01 step 4: times the mha step graph, dynamic n against static n, with and without graph capture.
// Usage (from web/): node bench/dit-step/run.mjs --gpu nvidia|intel [--channel chrome|msedge] [--only name]
// Needs the exports in build/webgpu/dit/base/: step_mha.onnx, step_mha_n{1024,2048}.onnx and
// step_n{1024,2048}.onnx (tools/webgpu/export_dit.py --model base [--attention mha] --static-n 1024). Each
// configuration runs in a new page. Writes results/<channel>-<gpu>.json.
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
const only = arg("only", "");

// dynamic: warm at 1024, then the first call at 2048 and at 1913 (a golden chunk) is the new-n latency.
// Graph capture fails in ORT Web 1.31.1 on every base graph (createBindGroup error; docs/debugging.md).
// The -capture configs stay so a newer ORT can be retried; they use the unfused static graphs.
const configs = [
  { name: "mha-dynamic", graph: "step_mha.onnx", ns: [1024, 2048, 1913] },
  { name: "mha-static-1024", graph: "step_mha_n1024.onnx", ns: [1024] },
  { name: "mha-static-2048", graph: "step_mha_n2048.onnx", ns: [2048] },
  { name: "unfused-static-1024", graph: "step_n1024.onnx", ns: [1024] },
  { name: "unfused-static-1024-capture", graph: "step_n1024.onnx", ns: [1024], graphCapture: true },
  { name: "unfused-static-2048", graph: "step_n2048.onnx", ns: [2048] },
  { name: "unfused-static-2048-capture", graph: "step_n2048.onnx", ns: [2048], graphCapture: true },
].filter((c) => !only || c.name === only);
// --graph file --n 64 [--base /model/dit/x/] [--text-dim 64] [--capture]: one ad hoc configuration instead
if (arg("graph", "")) {
  configs.length = 0;
  configs.push({
    name: "custom",
    graph: arg("graph"),
    ns: arg("n", "1024").split(",").map(Number),
    base: arg("base", "/model/dit/base/"),
    textDim: Number(arg("text-dim", "512")),
    graphCapture: argv.includes("--capture"),
  });
}

const server = await createServer({
  root: join(here, "../.."),
  logLevel: "warn",
  server: { port: 0, host: "127.0.0.1" },
});
await server.listen();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error("vite did not start");
const args = ["--enable-unsafe-webgpu"];
if (gpu === "nvidia") args.push("--force_high_performance_gpu");
const browser = await chromium.launch({ channel, args });

const results = [];
for (const config of configs) {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  await page.goto(`${url}bench/dit-step/`);
  await page.waitForSelector("body[data-ready='1']", { timeout: 60_000 });
  let r;
  try {
    r = await page.evaluate((c) => window.runStepBench(c), config);
  } catch (e) {
    r = { error: String(e) };
  }
  r.name = config.name;
  r.pageErrors = pageErrors;
  results.push(r);
  await page.close();
  if (r.error) {
    console.log(`${config.name.padEnd(28)} ERROR ${r.error}`);
    continue;
  }
  for (const p of r.perN) {
    console.log(
      `${config.name.padEnd(28)} n ${String(p.n).padStart(4)}  first ${p.firstMs} ms  single ${p.singleMs.median} ms` +
        `  chain/step ${p.perStepInChainMs} ms  finite ${p.chainFinite}` +
        (r.gpuErrors.length ? `  gpuErrors ${r.gpuErrors.length}` : ""),
    );
  }
}

const out = { channel, gpuFlag: gpu, browserVersion: browser.version(), results };
mkdirSync(join(here, "results"), { recursive: true });
const tag = arg("graph", "") ? "-custom" : only ? `-${only}` : "";
writeFileSync(join(here, "results", `${channel}-${gpu}${tag}.json`), JSON.stringify(out, null, 2));
await browser.close();
await server.close();
