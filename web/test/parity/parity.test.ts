// Self-tests of the parity runner (plan 00, step 6), on the committed tiny case.
// The Python twin (tools/webgpu/parity_selftest.py) runs the same scenarios.
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { fsReader, loadCase, loadTolerances, type GoldenCase, type Tolerances } from "./case";
import { checksFor, compare, report, type Result } from "./compare";
import { parseNpy, toArrayBuffer, type Tensor } from "./npy";

const ROOT = join(import.meta.dirname, "../../../testdata/webgpu");

let golden: GoldenCase;
let tolerances: Tolerances;

beforeAll(async () => {
  const read = await fsReader(ROOT);
  golden = await loadCase(read, "tiny");
  tolerances = await loadTolerances(read);
});

function copy(t: Tensor): Tensor {
  return { ...t, shape: [...t.shape], data: t.data.slice() };
}

async function compareAll(override: Record<string, Tensor> = {}): Promise<Result[]> {
  const results: Result[] = [];
  for (const e of golden.manifest.tensors) {
    if (checksFor(e.name).length === 0) continue;
    results.push(...(await compare(golden, tolerances, e.name, override[e.name] ?? (await golden.tensor(e.name)))));
  }
  return results;
}

describe("checksFor", () => {
  // the same table is in parity_selftest.py; both runners must map names the same way
  it.each([
    ["c0.dit.s03.out_cond", ["dit.out"]],
    ["c2.dit.s31.out_uncond", ["dit.out"]],
    ["c0.dit_internal.s16.block21", ["dit_internal.block"]],
    ["c0.dit_internal.text_embed_uncond", ["dit_internal.text_embed"]],
    ["c0.dit_internal.s00.input_embed", ["dit_internal.input_embed"]],
    ["c0.dit_internal.s16.norm_out", ["dit_internal.norm_out"]],
    ["mel.ref", ["mel.log_masked", "mel.linear"]],
    ["c1.vocoder.head_out", ["vocoder.head_out", "vocoder.stft_complex"]],
    ["c0.dit.text_ids", ["exact"]],
    ["c0.noise.y0", []],
    ["c0.dit.s00.x", []],
    ["c0.dit.step_cond", []],
    ["final.wav", []],
  ])("%s", (name, checks) => {
    expect(checksFor(name)).toEqual(checks);
  });

  it("knows every check it returns", () => {
    const names = golden.manifest.tensors.map((t) => t.name);
    for (const check of names.flatMap(checksFor)) {
      if (check !== "exact") expect(tolerances.checks).toHaveProperty([check]);
    }
  });
});

describe("parity runner", () => {
  it("passes golden data against itself", async () => {
    const results = await compareAll();
    const r = report(results);
    expect(r.failed, r.text).toBe(0);
    expect(r.firstFailingStage).toBeNull();
    expect(results.length).toBeGreaterThan(30);
  });

  it("fails a perturbed DiT output and names the dit stage", async () => {
    const t = copy(await golden.tensor("c0.dit.s03.out_cond"));
    for (let i = 0; i < t.data.length; i++) t.data[i] = (t.data[i] as number) * 1.01;
    const r = report(await compareAll({ "c0.dit.s03.out_cond": t }));
    expect(r.failed).toBe(1);
    expect(r.firstFailingStage).toBe("dit");
    expect(r.text.split("\n").at(-1)).toContain("c0.dit.s03.out_cond");
  });

  it("names the earliest failing stage when several fail", async () => {
    const dit = copy(await golden.tensor("c0.dit.s03.out_cond"));
    dit.data[0] = (dit.data[0] as number) + 10;
    const wav = copy(await golden.tensor("c0.vocoder.wav"));
    wav.data[0] = (wav.data[0] as number) + 1;
    const mel = copy(await golden.tensor("mel.ref"));
    for (let i = 0; i < mel.data.length; i++) mel.data[i] = (mel.data[i] as number) + 0.5;
    const r = report(await compareAll({ "c0.dit.s03.out_cond": dit, "c0.vocoder.wav": wav, "mel.ref": mel }));
    expect(r.firstFailingStage).toBe("mel");
  });

  it("fails a mel in the wrong layout with a shape error, not a numeric one", async () => {
    const mel = await golden.tensor("mel.ref"); // (1, 100, n)
    const transposed: Tensor = { ...mel, shape: [1, mel.shape[2] as number, 100] };
    const results = await compare(golden, tolerances, "mel.ref", transposed);
    expect(results.every((r) => !r.pass && r.error?.startsWith("shape"))).toBe(true);
    expect(results[0]?.metrics).toBeUndefined();
  });

  it("fails NaN, and wrong token ids", async () => {
    const wav = copy(await golden.tensor("c0.vocoder.wav"));
    wav.data[100] = NaN;
    expect((await compare(golden, tolerances, "c0.vocoder.wav", wav))[0]?.pass).toBe(false);
    const ids = copy(await golden.tensor("c0.dit.text_ids"));
    ids.data[0] = (ids.data[0] as number) + 1;
    expect((await compare(golden, tolerances, "c0.dit.text_ids", ids))[0]?.pass).toBe(false);
  });

  it("uses the tiny case's own tolerance where tolerances.json has one", async () => {
    // Linux and Windows CPU differ by ~1e-6 here, above the shared 1e-6 (plan 00, step 7)
    const t = copy(await golden.tensor("c0.dit_internal.s04.time_embed"));
    for (let i = 0; i < t.data.length; i++) t.data[i] = (t.data[i] as number) * (1 + 3e-6);
    const [r] = await compare(golden, tolerances, "c0.dit_internal.s04.time_embed", t);
    expect(r?.tolerance?.relL2).toBe(tolerances.checks["dit_internal.time_embed"]?.tiny?.rel_l2);
    expect(r?.tolerance?.relL2).toBeGreaterThan(tolerances.checks["dit_internal.time_embed"]?.rel_l2 as number);
    expect(r?.pass).toBe(true);
  });

  it("refuses to compare an input", async () => {
    await expect(compare(golden, tolerances, "c0.noise.y0", await golden.tensor("c0.noise.y0"))).rejects.toThrow(
      /input/,
    );
  });
});

// Smoke test (plan 00, step 5 data): the CUDA fp32 rerun, saved by measure_floor.py --dump, run
// through the runner as if it were a port. Passes by construction; skipped where the dumps are absent.
const REAL = ["en-short", "zh-short", "en-multi", "en-multi-fixdur", "en-prep", "en-epss16"];
const dumped = REAL.filter((c) => existsSync(join(ROOT, c, "cuda")));

describe.skipIf(dumped.length === 0)("smoke: CUDA rerun as a port", () => {
  it.each(dumped)("%s passes", async (caseName) => {
    const read = await fsReader(ROOT);
    const g = await loadCase(read, caseName);
    const { readFile } = await import("node:fs/promises");
    const results: Result[] = [];
    for (const file of readdirSync(join(ROOT, caseName, "cuda"))) {
      const name = file.replace(/\.npy$/, "");
      if (checksFor(name).length === 0) continue;
      const t = parseNpy(toArrayBuffer(await readFile(join(ROOT, caseName, "cuda", file))));
      results.push(...(await compare(g, tolerances, name, t)));
    }
    const r = report(results);
    expect(r.failed, r.text).toBe(0);
    expect(results.length).toBeGreaterThan(50);
  });
});
