// Plan 04 step 4 (D7): single steps at large n on the laptop GPUs. Local only. Run from web/:
//   PW_GPU=nvidia npx playwright test -c playwright.local.config.ts large-n
/// LARGE_N=4096,8192 picks the sizes; LARGE_N_STEPS=3 the steps per size (the first includes the text graph);
// LARGE_N_STEP_FILE=step.onnx runs the unfused-attention graph instead of step_mha.onnx.
import { existsSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { LargeNResult } from "../pipeline/run-large-n";

const modelDir = join(import.meta.dirname, "../../../build/webgpu/dit/base");
const sizes = (process.env.LARGE_N ?? "2048,4096,8192").split(",").map(Number);
const steps = Number(process.env.LARGE_N_STEPS ?? "3");
const stepFile = process.env.LARGE_N_STEP_FILE ?? "step_mha.onnx";

test.skip(!existsSync(join(modelDir, "step_mha.onnx")), "no step_mha.onnx in build/webgpu/dit/base");

test(`steps at n = ${sizes.join(", ")}`, async ({ page }, testInfo) => {
  await page.goto("/");
  const r: LargeNResult = await page.evaluate(
    async (opts) => {
      const url = "/test/pipeline/run-large-n.ts";
      const m = await import(/* @vite-ignore */ url);
      return m.runLargeN(opts);
    },
    { modelBase: "/model/dit/base/", sizes, steps, stepFile },
  );
  testInfo.annotations.push({ type: "result", description: JSON.stringify(r) });
  for (const run of r.runs)
    console.log(
      `${r.adapter} n=${run.n}: steps ${run.stepMs.join(", ")} ms; finite ${run.finite}; live ${run.live} ${run.error}`,
    );
  if (r.lost) console.log(`device lost: ${r.lost}`);
  for (const e of r.gpuErrors) console.log(`gpu error: ${e}`);
  expect(r.runs.every((run) => run.finite && !run.error && run.live === 0)).toBe(true);
});
