// Plan 04 step 2: torch's CPU randn against the golden noise, chunk after chunk from seed 0, in every case.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { TorchGenerator } from "../../src/pipeline/noise";
import { parseNpy, toArrayBuffer } from "../parity";

const TESTDATA = join(import.meta.dirname, "../../../testdata/webgpu");
const golden = readdirSync(TESTDATA).filter((d) => existsSync(join(TESTDATA, d, "manifest.json")));
const npy = (path: string) => parseNpy(toArrayBuffer(readFileSync(path)));

describe("torch.randn on the CPU", () => {
  it.each(golden)("%s: every chunk's y0 from seed 0", (name) => {
    const m = JSON.parse(readFileSync(join(TESTDATA, name, "manifest.json"), "utf8")) as { chunks: unknown[] };
    const gen = new TorchGenerator(0);
    for (let c = 0; c < m.chunks.length; c++) {
      const ref = npy(join(TESTDATA, name, `c${c}.noise.y0.npy`));
      const got = gen.randn(ref.data.length);
      let maxAbs = 0;
      let differ = 0;
      for (let i = 0; i < got.length; i++) {
        const d = Math.abs((got[i] as number) - (ref.data[i] as number));
        if (d > 0) differ++;
        if (d > maxAbs) maxAbs = d;
      }
      expect({ chunk: c, maxAbs, differ }).toEqual({ chunk: c, maxAbs: 0, differ: 0 });
    }
  });
});

interface RandnCases {
  cpu_capability: string;
  cases: { seed: number; sizes: number[]; draws: number[][] }[];
}
const fixture = JSON.parse(readFileSync(join(TESTDATA, "pipeline/randn_cases.json"), "utf8")) as RandnCases;

describe("seeds and sizes (tools/webgpu/noise_fixtures.py)", () => {
  it("was made on an AVX2 machine, the path this port follows", () => expect(fixture.cpu_capability).toBe("AVX2"));

  it.each(fixture.cases.map((c) => [c.seed, c] as const))("seed %i", (_seed, c) => {
    const gen = new TorchGenerator(c.seed);
    c.sizes.forEach((n, k) => expect(Array.from(gen.randn(n)), `size ${n}`).toEqual(c.draws[k]));
  });

  it("takes only uint32 seeds, which is all torch's CPU generator uses", () => {
    expect(() => new TorchGenerator(2 ** 32)).toThrow(/uint32/);
    expect(() => new TorchGenerator(-1)).toThrow(/uint32/);
    expect(() => new TorchGenerator(1.5)).toThrow(/uint32/);
  });
});
