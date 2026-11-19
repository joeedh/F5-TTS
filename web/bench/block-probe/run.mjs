// Runs the one-block probe (tools/webgpu/probe_block.py) on a laptop GPU and prints the error of every
// intermediate against the float64 reference. Usage (from web/): node bench/block-probe/run.mjs --gpu nvidia|intel
import { chromium } from "playwright";
import { createServer } from "vite";

const argv = process.argv.slice(2);
const gpu = argv[argv.indexOf("--gpu") + 1] ?? "nvidia";
const server = await createServer({ logLevel: "warn", server: { port: 0, host: "127.0.0.1" } });
await server.listen();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error("vite did not start");
const args = ["--enable-unsafe-webgpu"];
if (gpu === "nvidia") args.push("--force_high_performance_gpu");
const browser = await chromium.launch({ channel: "chrome", args });
const page = await browser.newPage();
await page.goto(url);
const r = await page.evaluate(async () => {
  const m = await import(/* @vite-ignore */ "/bench/block-probe/main.ts");
  return m.runProbe();
});
const e = (x) => (x === null || x === undefined ? "-" : x.toExponential(2));
console.log(`adapter ${r.adapter}`);
for (const row of r.rows) {
  console.log(
    `${row.name.padEnd(6)} relL2 ${e(row.relL2)}  maxAbs ${e(row.maxAbs)}  (torch CPU fp32 ${e(row.cpuFp32RelL2)})`,
  );
}
await browser.close();
await server.close();
