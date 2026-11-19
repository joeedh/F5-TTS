// Plan 01 step 3: the tiny model's exported DiT graphs on WebGPU against the tiny golden case. Runs in CI on
// SwiftShader. The real model is checked by test/browser-local/dit.spec.ts (local only).
import { expect, test } from "@playwright/test";
import type { DitParityResult } from "../dit/run-parity";

test("tiny DiT graphs match the golden data on WebGPU", async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.goto("/");
  const r: DitParityResult = await page.evaluate(async () => {
    const url = "/test/dit/run-parity.ts";
    const m = await import(/* @vite-ignore */ url);
    return m.runDitParity({ caseName: "tiny", modelBase: "/testdata/webgpu/tiny/onnx/", debug: true });
  });
  testInfo.annotations.push({ type: "result", description: JSON.stringify({ ...r, failures: r.failures.length }) });
  expect(r.failures, r.summary).toEqual([]);
  expect(r.comparisons).toBe(46); // the same count as tools/webgpu/check_dit_ort.py --case tiny --debug
  expect(r.gpuErrors).toEqual([]);
  expect(r.placement.sessions).toHaveLength(2);
  expect(r.placement.offenders).toEqual([]);
});
