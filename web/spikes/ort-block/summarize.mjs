// Prints markdown tables from results/*.json (written by run.mjs).
// Usage (from web/): node spikes/ort-block/summarize.mjs [tag ...]   default: the four browser/GPU runs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const tags = process.argv.slice(2);
if (tags.length === 0) tags.push("chrome-nvidia", "chrome-intel", "msedge-nvidia", "msedge-intel");

const fmt = (x, d = 2) => (x === undefined || x === null || Number.isNaN(x) ? "-" : x.toFixed(d));
const exp = (x) => (x === undefined ? "-" : x.toExponential(1));

for (const tag of tags) {
  const r = JSON.parse(readFileSync(join(here, "results", `${tag}.json`), "utf8"));
  console.log(`\n### ${tag}: ${r.adapter.vendor} ${r.adapter.architecture}, ${r.browserVersion}\n`);
  console.log("| Model | rel L2 | max abs | median ms | GFLOPS | 22-block chain ms | placement |");
  console.log("|---|---|---|---|---|---|---|");
  const byName = Object.fromEntries(r.results.map((x) => [x.file, x]));
  for (const x of r.results) {
    if (x.error) {
      console.log(`| ${x.file} | error: ${x.error.slice(0, 80)} | | | | | |`);
      continue;
    }
    const placed = x.placementLog
      ? x.placementLog.every((l) => !/placed on \[CPU/i.test(l)) &&
        x.placementLog.some((l) => /All nodes placed on \[WebGpuExecutionProvider\]/.test(l))
        ? "all WebGPU"
        : "SEE LOG"
      : "n/a";
    console.log(
      `| ${x.file} | ${exp(x.correctness.relL2)} | ${exp(x.correctness.maxAbs)} | ${fmt(x.runMs.median, 1)} | ` +
        `${fmt(x.gflops, 0)} | ${x.chain22Ms ? fmt(x.chain22Ms.median, 0) : ""} | ${placed} |`,
    );
  }
  console.log("\nProjected real-time factor (32 steps; blocks only, and blocks plus conv_pos_embed):\n");
  console.log("| n | new audio s | fp32 RTF | fp32 RTF + convpos | fp16 RTF | fp16 RTF + convpos |");
  console.log("|---|---|---|---|---|---|");
  for (const n of [1024, 2048]) {
    const row = [n];
    const b32 = byName[`block_n${n}.onnx`]?.projected;
    const b16 = byName[`block_n${n}_fp16.onnx`]?.projected;
    const c32 = byName[`convpos_n${n}.onnx`]?.runMs?.median ?? NaN;
    const c16 = byName[`convpos_n${n}_fp16.onnx`]?.runMs?.median ?? NaN;
    row.push(fmt(b32?.newAudioS, 1));
    for (const [b, c] of [
      [b32, c32],
      [b16, c16],
    ]) {
      row.push(fmt(b?.rtf));
      row.push(b ? fmt(((b.stepMs + c) * 32) / 1000 / b.newAudioS) : "-");
    }
    console.log(`| ${row.join(" | ")} |`);
  }
}
