// Plan 02 step 1: the TypeScript log-mel against torchaudio's filterbank and the golden mel.ref.
// The tiny case is committed; the real cases run where make_golden.py has built them.
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { FFT } from "../../src/audio/fft";
import { logMel, melFilterbank, N_FREQS, N_MELS } from "../../src/audio/mel";
import { compare, fsReader, loadCase, loadTolerances, parseNpy, toArrayBuffer, type Tensor } from "../parity";

const ROOT = join(import.meta.dirname, "../../../testdata/webgpu");
const REAL = ["en-short", "zh-short", "en-multi", "en-multi-fixdur", "en-prep", "en-epss16"];
const built = REAL.filter((c) => existsSync(join(ROOT, c, "mel.ref.npy")));

async function melCase(caseName: string) {
  const read = await fsReader(ROOT);
  const golden = await loadCase(read, caseName);
  const tolerances = await loadTolerances(read);
  const audio = await golden.tensor("audio.ref"); // (1, samples)
  const mel = logMel(audio.data as Float32Array);
  const actual: Tensor = { dtype: "float32", shape: [1, N_MELS, mel.frames], data: mel.data };
  return compare(golden, tolerances, "mel.ref", actual);
}

describe("FFT", () => {
  it("matches a direct DFT", () => {
    const n = 64;
    const x = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) + 0.3 * Math.cos(i * 2.1) - 0.1);
    const re = x.slice();
    const im = new Float64Array(n);
    new FFT(n).forward(re, im);
    for (let k = 0; k < n; k++) {
      let sr = 0;
      let si = 0;
      for (let j = 0; j < n; j++) {
        sr += (x[j] as number) * Math.cos((-2 * Math.PI * j * k) / n);
        si += (x[j] as number) * Math.sin((-2 * Math.PI * j * k) / n);
      }
      expect(re[k]).toBeCloseTo(sr, 10);
      expect(im[k]).toBeCloseTo(si, 10);
    }
  });

  it("inverts", () => {
    const n = 1024;
    const x = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.01) * (i % 7));
    const re = x.slice();
    const im = new Float64Array(n);
    const fft = new FFT(n);
    fft.forward(re, im);
    fft.inverseUnscaled(re, im);
    for (let i = 0; i < n; i++) expect((re[i] as number) / n).toBeCloseTo(x[i] as number, 10);
  });
});

describe("melFilterbank", () => {
  it("matches torchaudio's matrix", async () => {
    const path = join(ROOT, "audio-fixtures", "mel_fbank.npy");
    const ref = parseNpy(toArrayBuffer(await readFile(path)));
    expect(ref.shape).toEqual([N_FREQS, N_MELS]);
    const fb = melFilterbank();
    let worst = 0;
    for (let i = 0; i < fb.length; i++) worst = Math.max(worst, Math.abs((fb[i] as number) - (ref.data[i] as number)));
    // 3.8e-6: torch's vectorised float32 pow rounds 5 of the 102 centre frequencies differently from JS.
    // The same formula in plain float64 would be 9.0e-6 off.
    expect(worst).toBeLessThan(5e-6);
    // same zero pattern: a triangle edge on the wrong side of a bin would show here
    for (let i = 0; i < fb.length; i++) expect((fb[i] as number) > 0).toBe((ref.data[i] as number) > 0);
  });
});

describe("logMel", () => {
  it("matches the tiny case's mel.ref", async () => {
    const results = await melCase("tiny");
    expect(results.map((r) => r.check)).toEqual(["mel.log_masked", "mel.linear"]);
    for (const r of results) expect(r.pass, `${r.check} ${JSON.stringify(r.metrics)}`).toBe(true);
  });

  it("rejects audio shorter than the reflect pad", () => {
    expect(() => logMel(new Float32Array(512))).toThrow(/more than 512/);
  });

  it.skipIf(built.length === 0).each(built)(
    "matches %s's mel.ref",
    async (caseName) => {
      for (const r of await melCase(caseName)) expect(r.pass, `${r.check} ${JSON.stringify(r.metrics)}`).toBe(true);
    },
    30_000,
  );
});
