// Plan 04 step 6: the TTS API end to end on the tiny model, from its input.wav to its final.wav, and the API's
// behaviour (seed, cfg below 1e-5, refusals, abort). Runs in CI on SwiftShader; the real model is checked by
// test/browser-local/final.spec.ts.
import { expect, test } from "@playwright/test";
import type { FinalRunResult } from "../pipeline/run-final";

test("TTS on the tiny model reproduces final.wav, and the API behaves", async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  await page.goto("/");
  const r: FinalRunResult = await page.evaluate(async () => {
    const url = "/test/pipeline/run-final.ts";
    const m = await import(/* @vite-ignore */ url);
    return m.runFinal({
      caseName: "tiny",
      ditBase: "/testdata/webgpu/tiny/onnx/",
      stepFile: "step.onnx",
      vocoderUrl: "/testdata/webgpu/tiny/onnx/vocos.onnx",
      apiChecks: true,
    });
  });
  const { wave: _wave, ...rest } = r;
  testInfo.annotations.push({ type: "result", description: JSON.stringify(rest) });
  for (const l of r.lines) console.log(`  ${l}`);
  for (const [name, ok] of Object.entries(r.api)) console.log(`  ${ok ? "ok  " : "FAIL"} ${name}`);
  expect(r.failed, r.lines.at(-1)).toBe(0);
  expect(Object.entries(r.api).filter(([, ok]) => !ok)).toEqual([]);
  expect(Object.keys(r.api)).toHaveLength(10);
  expect(r.live).toBe(0);
  expect(r.gpuErrors).toEqual([]);
});
