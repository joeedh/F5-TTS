// Plan 04 step 3: step_cond and the restore-and-crop against the golden cases, exactly.
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { LogMel } from "../../src/audio/mel";
import { finishChunk, stepCond } from "../../src/pipeline/cond";
import { fsReader, loadCase, type GoldenCase, type Reader } from "../parity";

const TESTDATA = join(import.meta.dirname, "../../../testdata/webgpu");
const golden = readdirSync(TESTDATA).filter((d) => existsSync(join(TESTDATA, d, "manifest.json")));

let read: Reader;
beforeAll(async () => {
  read = await fsReader(TESTDATA);
});

async function refMel(g: GoldenCase): Promise<LogMel> {
  const t = await g.tensor("mel.ref"); // (1, 100, n_ref)
  return { data: t.data as Float32Array, frames: t.shape[2] as number };
}
const bytes = (a: Float32Array | Float64Array | Int32Array | Uint8Array) =>
  Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const chunksOf = (g: GoldenCase) => g.manifest.chunks as { duration: number; ref_audio_len: number }[];

describe("stepCond", () => {
  it.each(golden)("%s: equals every chunk's dit.step_cond", async (name) => {
    const g = await loadCase(read, name);
    const mel = await refMel(g);
    for (const [c, chunk] of chunksOf(g).entries()) {
      const ref = await g.tensor(`c${c}.dit.step_cond`);
      expect(ref.shape).toEqual([1, chunk.duration, 100]);
      expect(bytes(stepCond(mel, chunk.duration)).equals(bytes(ref.data))).toBe(true);
    }
  });
});

describe("finishChunk", () => {
  it.each(golden)("%s: restores the reference and crops to mel_out", async (name) => {
    const g = await loadCase(read, name);
    const mel = await refMel(g);
    for (const [c, chunk] of chunksOf(g).entries()) {
      const out = await g.tensor(`c${c}.mel_out`); // (1, 100, duration - ref_audio_len)
      const frames = out.shape[2] as number;
      expect(frames).toBe(chunk.duration - chunk.ref_audio_len);
      // the final x: NaN where the reference is restored, the golden output elsewhere
      const x = new Float32Array(chunk.duration * 100).fill(Number.NaN);
      for (let f = mel.frames; f < chunk.duration; f++)
        for (let k = 0; k < 100; k++) x[f * 100 + k] = out.data[k * frames + f - chunk.ref_audio_len] as number;
      const got = finishChunk(x, mel, chunk.ref_audio_len);
      expect(bytes(got).equals(bytes(out.data))).toBe(true);
    }
  });

  it("refuses shapes that do not fit", () => {
    const mel: LogMel = { data: new Float32Array(100 * 5), frames: 5 };
    expect(() => finishChunk(new Float32Array(100 * 5), mel, 4)).toThrow(/not \(1, duration, 100\)/);
    expect(() => finishChunk(new Float32Array(100 * 9), mel, 5)).toThrow(/under n_ref/);
    expect(() => stepCond(mel, 5)).toThrow(/must exceed/);
  });
});
