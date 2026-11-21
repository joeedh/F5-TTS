// Plan 02 step 5: the tiny model's Vocos graph on WebGPU, then the TypeScript istft, against the tiny golden
// case. Runs in CI on SwiftShader. The real model is checked by test/browser-local/vocoder.spec.ts.
import { expect, test } from "@playwright/test";
import type { VocoderParityResult } from "../vocoder/run-parity";

test("tiny vocoder graph and istft match the golden data on WebGPU", async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  await page.goto("/");
  const r: VocoderParityResult = await page.evaluate(async () => {
    const url = "/test/vocoder/run-parity.ts";
    const m = await import(/* @vite-ignore */ url);
    return m.runVocoderParity({
      caseName: "tiny",
      modelBase: "/testdata/webgpu/tiny/onnx/",
      debug: true,
      timingRuns: 1,
    });
  });
  testInfo.annotations.push({ type: "result", description: JSON.stringify({ ...r, failures: r.failures.length }) });
  expect(r.failures, r.summary).toEqual([]);
  // backbone, head_out (two checks) and wav for the one chunk: check_vocos_ort.py's 3 plus the wav
  expect(r.comparisons).toBe(4);
  expect(r.gpuErrors).toEqual([]);
  expect(r.placement.sessions).toHaveLength(1);
  expect(r.placement.offenders).toEqual([]);
  expect(r.placement.cpuNodes.flat().filter((n) => n.startsWith("Gelu"))).toEqual([]); // the exact GELU (R6)
});
