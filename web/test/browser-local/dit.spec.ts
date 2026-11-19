// Plan 01 step 3: the real DiT exports on the laptop GPUs, every step of every chunk of every golden case.
// Local only: needs build/webgpu/dit/base/ (tools/webgpu/export_dit.py) and the golden cases
// (tools/webgpu/make_golden.py); skipped where they are absent. Run from web/:
//   npx playwright test -c playwright.local.config.ts                 # Intel GPU, unfused
//   PW_GPU=nvidia DIT_ATTENTION=mha npx playwright test -c playwright.local.config.ts
// DIT_CASES=en-short,zh-short limits the cases; DIT_DEBUG=1 also checks (and prints) the internals;
// DIT_MAX_STEPS=17 runs only the first steps of each chunk.
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { DitParityResult } from "../dit/run-parity";

const repo = join(import.meta.dirname, "../../..");
const attention = (process.env.DIT_ATTENTION ?? "unfused") as "unfused" | "mha" | "onnx23";
const suffix = attention === "unfused" ? "" : `_${attention}`;
const modelDir = join(repo, "build/webgpu/dit/base");
const goldenDir = join(repo, "testdata/webgpu");
const debug = process.env.DIT_DEBUG === "1";
const maxSteps = process.env.DIT_MAX_STEPS ? Number(process.env.DIT_MAX_STEPS) : undefined;
const stepFile = `${debug ? "step_debug" : "step"}${suffix}.onnx`;
const cases = (
  process.env.DIT_CASES?.split(",") ??
  readdirSync(goldenDir).filter((d) => d !== "tiny" && existsSync(join(goldenDir, d, "manifest.json")))
).filter((c) => c);

test.skip(!existsSync(join(modelDir, stepFile)), `no ${stepFile} in build/webgpu/dit/base`);

for (const caseName of cases) {
  test(`${caseName}: DiT (${attention}) matches the golden data on WebGPU`, async ({ page }, testInfo) => {
    await page.goto("/");
    const r: DitParityResult = await page.evaluate(
      async (opts) => {
        const url = "/test/dit/run-parity.ts";
        const m = await import(/* @vite-ignore */ url);
        return m.runDitParity(opts);
      },
      { caseName, modelBase: "/model/dit/base/", attention, debug, maxSteps },
    );
    const { failures, internals, ...rest } = r;
    for (const line of internals) console.log(line);
    testInfo.annotations.push({ type: "result", description: JSON.stringify({ ...rest, failures: failures.length }) });
    console.log(`${caseName} (${attention}, ${r.adapter.vendor}): ${r.summary}; ${r.totalMs} ms`);
    for (const f of failures.slice(0, 20)) console.log(f);
    for (const [check, s] of Object.entries(r.errorSpread)) {
      const worst = s.worst.map((w) => `${w.name} ${w.relL2.toFixed(2)}`).join(", ");
      console.log(`  ${check}: relL2/tol median ${s.relL2Median} p90 ${s.relL2P90} max ${s.relL2Max}; worst ${worst}`);
    }
    expect(failures, r.summary).toEqual([]);
    expect(r.gpuErrors).toEqual([]);
    expect(r.placement.offenders).toEqual([]);
  });
}
