// Plan 04 step 7 (D5): generates the quality subset with the port in Chrome, through the Vite dev server
// (which serves the models and data/seedtts_testset). Usage, from web/:
//   node bench/quality/run.mjs --lang en [--gpu nvidia] [--limit 2]
// Writes data/seedtts_testset/gen/webgpu/<lang>/<utt>.wav (float32, 24 kHz) and timing.json. Utterances
// already generated are skipped, so an interrupted run can be resumed.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "../..");
const data = join(web, "../data/seedtts_testset");
const argv = process.argv.slice(2);
const arg = (name, dflt) => (argv.includes(`--${name}`) ? argv[argv.indexOf(`--${name}`) + 1] : dflt);
const lang = arg("lang");
if (lang !== "en" && lang !== "zh") throw new Error("--lang en|zh");
const limit = Number(arg("limit", "100000"));
const rows = JSON.parse(readFileSync(join(data, "quality_subset.json"), "utf8"))[lang].slice(0, limit);
const outDir = join(data, "gen/webgpu", lang);
mkdirSync(outDir, { recursive: true });

/** A 32-bit float mono WAV, as soundfile writes with subtype FLOAT. */
function wavFile(pcm, rate) {
  const h = Buffer.alloc(44);
  h.write("RIFF", 0);
  h.writeUInt32LE(36 + pcm.length, 4);
  h.write("WAVEfmt ", 8);
  h.writeUInt32LE(16, 16);
  h.writeUInt16LE(3, 20);
  h.writeUInt16LE(1, 22);
  h.writeUInt32LE(rate, 24);
  h.writeUInt32LE(rate * 4, 28);
  h.writeUInt16LE(4, 32);
  h.writeUInt16LE(32, 34);
  h.write("data", 36);
  h.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([h, pcm]);
}

const server = await createServer({
  root: web,
  configFile: join(web, "vite.config.ts"),
  server: { port: 5175, strictPort: true },
});
await server.listen();
const args = ["--enable-unsafe-webgpu"];
if (arg("gpu") === "nvidia") args.push("--force_high_performance_gpu");
const browser = await chromium.launch({ channel: "chrome", args });
const timingPath = join(outDir, "timing.json");
const timing = existsSync(timingPath) ? JSON.parse(readFileSync(timingPath, "utf8")) : {};
try {
  const page = await browser.newPage();
  await page.goto("http://localhost:5175/bench/quality/index.html");
  await page.waitForFunction(() => "qualityBench" in globalThis);
  const { adapter } = await page.evaluate(() => globalThis.qualityBench.init());
  console.log(`adapter ${adapter}`);
  for (const [k, row] of rows.entries()) {
    const file = join(outDir, `${row.utt}.wav`);
    if (existsSync(file)) continue;
    const r = await page.evaluate((row) => globalThis.qualityBench.generate(row), row);
    if (r.live !== 0) throw new Error(`${row.utt}: ${r.live} GPU tensors left`);
    writeFileSync(file, wavFile(Buffer.from(r.wave, "base64"), 24000));
    timing[row.utt] = r.ms / 1000;
    writeFileSync(timingPath, JSON.stringify(timing, null, 1) + "\n");
    console.log(`${k + 1}/${rows.length} ${row.utt}: ${r.seconds.toFixed(2)} s in ${(r.ms / 1000).toFixed(2)} s`);
  }
} finally {
  await browser.close();
  await server.close();
}
