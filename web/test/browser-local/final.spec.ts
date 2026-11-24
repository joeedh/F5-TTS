// Plan 04 step 5: end to end on WebGPU, from each golden case's input.wav and texts to final.wav. Local only:
// needs build/webgpu/dit/base/, build/webgpu/vocos/ and the golden cases. Run from web/:
//   PW_GPU=nvidia npx playwright test -c playwright.local.config.ts final
// DIT_CASES=en-multi,zh-short limits the cases. WEIGHT_VARIANT=fp32 loads a plan 05 weight variant instead;
// the others differ from the fp32 golden data by their own weight error. Each result is written to build/webgpu/listen/ for listening.
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { FinalRunResult } from "../pipeline/run-final";

const repo = join(import.meta.dirname, "../../..");
const goldenDir = join(repo, "testdata/webgpu");
// not test-results/, which Playwright empties at the start of every run
const outDir = join(repo, "build/webgpu/listen");
const variant = process.env.WEIGHT_VARIANT;
const cases = (
  process.env.DIT_CASES?.split(",") ??
  readdirSync(goldenDir).filter((d) => d !== "tiny" && existsSync(join(goldenDir, d, "manifest.json")))
).filter((c) => c);

test.skip(
  !existsSync(join(repo, "build/webgpu/dit/base/step_mha.onnx")) ||
    !existsSync(join(repo, "build/webgpu/vocos/vocos.onnx")),
  "needs build/webgpu/dit/base/step_mha.onnx and build/webgpu/vocos/vocos.onnx",
);

/** A 32-bit float mono WAV. */
function wavFile(samples: number[], rate: number): Buffer {
  const data = Buffer.from(new Float32Array(samples).buffer);
  const h = Buffer.alloc(44);
  h.write("RIFF", 0);
  h.writeUInt32LE(36 + data.length, 4);
  h.write("WAVEfmt ", 8);
  h.writeUInt32LE(16, 16);
  h.writeUInt16LE(3, 20); // IEEE float
  h.writeUInt16LE(1, 22);
  h.writeUInt32LE(rate, 24);
  h.writeUInt32LE(rate * 4, 28);
  h.writeUInt16LE(4, 32);
  h.writeUInt16LE(32, 34);
  h.write("data", 36);
  h.writeUInt32LE(data.length, 40);
  return Buffer.concat([h, data]);
}

for (const caseName of cases) {
  test(`${caseName}: input.wav to final.wav on WebGPU`, async ({ page }, testInfo) => {
    await page.goto("/");
    const r: FinalRunResult = await page.evaluate(
      async (opts) => {
        const url = "/test/pipeline/run-final.ts";
        const m = await import(/* @vite-ignore */ url);
        return m.runFinal(opts);
      },
      {
        caseName,
        ditBase: "/model/dit/base/",
        vocoderUrl: "/model/vocos/vocos.onnx",
        variantUrl: variant && `/model/variants/${variant}/`,
      },
    );
    const { wave, lines, ...rest } = r;
    mkdirSync(outDir, { recursive: true });
    const file = join(outDir, `${caseName}-${r.adapter}${variant ? `-${variant}` : ""}.wav`);
    writeFileSync(file, wavFile(wave, 24000));
    testInfo.annotations.push({ type: "result", description: JSON.stringify(rest) });
    console.log(
      `${caseName} (${r.adapter}): ${r.audioSeconds.toFixed(2)} s of audio in ${r.totalMs} ms; chunks done at ${r.chunkDoneMs.join(", ")} ms; ${file}`,
    );
    for (const l of lines) console.log(`  ${l}`);
    expect(r.gpuErrors).toEqual([]);
    expect(r.live).toBe(0);
    expect(r.failed, lines.at(-1)).toBe(0);
  });
}
