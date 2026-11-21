// Plan 02 step 5: the real Vocos graph on the laptop GPUs, then the TypeScript istft, for every chunk of every
// golden case, with per-chunk timing of the graph, the head_out readback and the istft (D2).
// Local only: needs build/webgpu/vocos/ (tools/webgpu/export_vocos.py --model base) and the golden cases.
// Run from web/:
//   npx playwright test -c playwright.local.config.ts vocoder              # Intel GPU
//   PW_GPU=nvidia npx playwright test -c playwright.local.config.ts vocoder
// VOCODER_CASES=en-short,zh-short limits the cases.
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { VocoderParityResult } from "../vocoder/run-parity";

const repo = join(import.meta.dirname, "../../..");
const goldenDir = join(repo, "testdata/webgpu");
const cases = (
  process.env.VOCODER_CASES?.split(",") ??
  readdirSync(goldenDir).filter((d) => d !== "tiny" && existsSync(join(goldenDir, d, "manifest.json")))
).filter((c) => c);

test.skip(!existsSync(join(repo, "build/webgpu/vocos/vocos_debug.onnx")), "no build/webgpu/vocos/vocos_debug.onnx");

for (const caseName of cases) {
  test(`${caseName}: vocoder matches the golden data on WebGPU`, async ({ page }, testInfo) => {
    await page.goto("/");
    const r: VocoderParityResult = await page.evaluate(
      async (opts) => {
        const url = "/test/vocoder/run-parity.ts";
        const m = await import(/* @vite-ignore */ url);
        return m.runVocoderParity(opts);
      },
      { caseName, modelBase: "/model/vocos/", debug: true, timingRuns: 5 },
    );
    const { failures, ...rest } = r;
    testInfo.annotations.push({ type: "result", description: JSON.stringify({ ...rest, failures: failures.length }) });
    console.log(`${caseName} (${r.adapter.vendor}): ${r.summary}; load ${r.loadMs} ms, total ${r.totalMs} ms`);
    console.log(`  placement ${JSON.stringify(r.placement.sessions)} cpu ${JSON.stringify(r.placement.cpuNodes)}`);
    for (const t of r.timing) console.log(`  timing ${JSON.stringify(t)}`);
    for (const f of failures.slice(0, 20)) console.log(f);
    for (const [check, s] of Object.entries(r.errorSpread)) {
      const worst = s.worst.map((w) => `${w.name} ${w.relL2.toFixed(3)}`).join(", ");
      console.log(`  ${check}: relL2/tol median ${s.relL2Median} max ${s.relL2Max}; worst ${worst}`);
    }
    expect(failures, r.summary).toEqual([]);
    expect(r.gpuErrors).toEqual([]);
    expect(r.placement.offenders).toEqual([]);
  });
}
