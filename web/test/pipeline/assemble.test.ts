// Plan 04 step 5: the cross-fade against the golden final.wav, and against upstream's code on cases the
// golden data never reaches (tools/webgpu/crossfade_fixtures.py).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { crossFade, crossFadeF64 } from "../../src/pipeline/assemble";
import { parseNpy, toArrayBuffer } from "../parity";

const TESTDATA = join(import.meta.dirname, "../../../testdata/webgpu");
const golden = readdirSync(TESTDATA).filter((d) => existsSync(join(TESTDATA, d, "manifest.json")));
const npy = (path: string) => parseNpy(toArrayBuffer(readFileSync(path)));
const bytes = (a: Float32Array | Float64Array) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);

describe("golden cases", () => {
  it.each(golden)("%s: the chunk waves cross-fade to final.wav exactly", (name) => {
    const m = JSON.parse(readFileSync(join(TESTDATA, name, "manifest.json"), "utf8")) as {
      chunks: unknown[];
      settings: { cross_fade_duration: number };
    };
    const waves = m.chunks.map((_, c) => npy(join(TESTDATA, name, `c${c}.vocoder.wav.npy`)).data as Float32Array);
    const ref = npy(join(TESTDATA, name, "final.wav.npy")).data as Float32Array;
    const got = crossFade(waves, m.settings.cross_fade_duration);
    expect(got.length).toBe(ref.length);
    expect(bytes(got).equals(bytes(ref))).toBe(true);
  });
});

interface FadeCase {
  fade: number;
  waves: string[];
  dtype: "float32" | "float64";
  result: string;
}
const fixtures = JSON.parse(readFileSync(join(TESTDATA, "pipeline/crossfade_cases.json"), "utf8")) as Record<
  string,
  FadeCase
>;
const f32 = (b64: string) => new Float32Array(toArrayBuffer(Buffer.from(b64, "base64")));

describe("upstream's code on cases the golden data never reaches", () => {
  it.each(Object.entries(fixtures))("%s", (_name, c) => {
    const raw = Buffer.from(c.result, "base64");
    const ref = c.dtype === "float64" ? new Float64Array(toArrayBuffer(raw)) : Float64Array.from(f32(c.result));
    const got = crossFadeF64(c.waves.map(f32), c.fade);
    expect(got.length).toBe(ref.length);
    expect(bytes(got).equals(bytes(ref))).toBe(true);
  });
});
