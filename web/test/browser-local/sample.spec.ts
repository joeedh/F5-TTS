// Plan 04 step 4: the ODE loop free-running on WebGPU, every chunk of every golden case, against mel_out.
// Local only: needs build/webgpu/dit/base/ and the golden cases. Run from web/:
//   npx playwright test -c playwright.local.config.ts sample              # Intel GPU
//   PW_GPU=nvidia npx playwright test -c playwright.local.config.ts sample
// SAMPLE_MODE=chain uses this port's own noise, step_cond and time grid; SAMPLE_DRIFT=0 skips the per-step
// readbacks (for timing); DIT_CASES=en-short,zh-short limits the cases.
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { SampleRunResult } from "../pipeline/run-sample";

const repo = join(import.meta.dirname, "../../..");
const modelDir = join(repo, "build/webgpu/dit/base");
const goldenDir = join(repo, "testdata/webgpu");
const mode = (process.env.SAMPLE_MODE ?? "golden") as "golden" | "chain";
const drift = process.env.SAMPLE_DRIFT !== "0";
const cases = (
  process.env.DIT_CASES?.split(",") ??
  readdirSync(goldenDir).filter((d) => d !== "tiny" && existsSync(join(goldenDir, d, "manifest.json")))
).filter((c) => c);

test.skip(!existsSync(join(modelDir, "step_mha.onnx")), "no step_mha.onnx in build/webgpu/dit/base");

for (const caseName of cases) {
  test(`${caseName}: sampleChunk (${mode}) matches mel_out on WebGPU`, async ({ page }, testInfo) => {
    await page.goto("/");
    const r: SampleRunResult = await page.evaluate(
      async (opts) => {
        const url = "/test/pipeline/run-sample.ts";
        const m = await import(/* @vite-ignore */ url);
        return m.runSample(opts);
      },
      { caseName, modelBase: "/model/dit/base/", mode, drift },
    );
    const { lines, ...rest } = r;
    testInfo.annotations.push({ type: "result", description: JSON.stringify(rest) });
    console.log(`${caseName} (${mode}, ${r.adapter.vendor}): chunks ${r.chunkMs.join(", ")} ms; load ${r.loadMs} ms`);
    for (const l of lines) console.log(`  ${l}`);
    for (const d of r.drift) console.log(`  c${d.chunk} relL2 per step: ${d.relL2.join(" ")}`);
    console.log(`  live tensors after each chunk ${r.liveAfterChunk.join(", ")}, at end ${r.liveAtEnd}`);
    expect(r.gpuErrors).toEqual([]);
    expect(r.liveAfterChunk.every((v) => v === 0)).toBe(true);
    expect(r.liveAtEnd).toBe(0);
    expect(r.failed, lines.at(-1)).toBe(0);
  });
}
