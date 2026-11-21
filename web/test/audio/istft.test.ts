// Plan 02 step 4: the inverse STFT and output rescale, from the golden head_out to the golden vocoder.wav
// (check vocoder.istft), and against torch on a synthetic spectrum with nonzero DC and Nyquist phases.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { irfftFrame, istft } from "../../src/audio/istft";
import { rescaleOutput } from "../../src/audio/output";
import {
  fsReader,
  judge,
  loadCase,
  loadTolerances,
  metrics,
  parseNpy,
  toArrayBuffer,
  type GoldenCase,
} from "../parity";

const ROOT = join(import.meta.dirname, "../../../testdata/webgpu");
const REAL = ["en-short", "zh-short", "en-multi", "en-multi-fixdur", "en-prep", "en-epss16"];
const built = REAL.filter((c) => existsSync(join(ROOT, c, "c0.vocoder.head_out.npy")));
const fixture = (file: string) => parseNpy(toArrayBuffer(readFileSync(join(ROOT, "audio-fixtures", "istft", file))));

async function checkCase(golden: GoldenCase) {
  const tolerances = await loadTolerances(await fsReader(ROOT));
  for (let c = 0; c < golden.manifest.chunks.length; c++) {
    const head = await golden.tensor(`c${c}.vocoder.head_out`); // (1, frames, 1026)
    const want = await golden.tensor(`c${c}.vocoder.wav`);
    const wave = rescaleOutput(istft(head.data as Float32Array, head.shape[1] as number), golden.manifest.audio);
    expect(wave.length).toBe(want.data.length);
    const r = judge(golden, tolerances, "vocoder", `c${c}.vocoder.wav`, "vocoder.istft", metrics(wave, want.data));
    expect(r.pass, `chunk ${c}: ${JSON.stringify(r.metrics)}`).toBe(true);
  }
}

describe("istft against torch (synthetic spectrum)", () => {
  const head = fixture("head_out.npy"); // (6, 1026)
  const frames = head.shape[0] as number;

  it("irfft ignores the imaginary parts of bins 0 and 512, as torch.fft.irfft does", () => {
    const want = fixture("irfft.npy");
    const out = new Float64Array(1024);
    for (let t = 0; t < frames; t++) {
      const re = new Float64Array(513);
      const im = new Float64Array(513);
      for (let k = 0; k < 513; k++) {
        const mag = Math.min(Math.exp(head.data[t * 1026 + k] as number), 100);
        const phase = head.data[t * 1026 + 513 + k] as number;
        re[k] = mag * Math.cos(phase);
        im[k] = mag * Math.sin(phase);
      }
      expect(Math.abs(im[0] as number)).toBeGreaterThan(1e-3); // the fixture really has a DC phase
      irfftFrame(re, im, out);
      const m = metrics(out, want.data.subarray(t * 1024, (t + 1) * 1024));
      expect(m.relL2).toBeLessThan(1e-6);
    }
  });

  it("matches torch.istft(center=True), with the clip at 100", () => {
    const want = fixture("wav.npy");
    const wave = istft(head.data as Float32Array, frames);
    expect(wave.length).toBe((frames - 1) * 256);
    expect(metrics(wave, want.data).relL2).toBeLessThan(1e-6);
  });
});

describe("istft and rescale against vocoder.wav", () => {
  it("passes on the tiny case", async () => checkCase(await loadCase(await fsReader(ROOT), "tiny")));
  it.skipIf(built.length === 0).each(built)(
    "passes on %s",
    async (c) => checkCase(await loadCase(await fsReader(ROOT), c)),
    60_000,
  );
});
