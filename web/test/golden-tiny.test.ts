// Checks that the committed tiny golden case is complete: every tensor the manifest lists exists,
// with an .npy header that agrees with the manifest. The real loader comes with the parity runner.
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const CASE_DIR = join(import.meta.dirname, "../../testdata/webgpu/tiny");
const STAGES = new Set([
  "input",
  "prep",
  "audio",
  "mel",
  "plan",
  "noise",
  "dit",
  "dit_internal",
  "mel_out",
  "vocoder",
  "final",
]);

interface TensorEntry {
  name: string;
  file: string;
  stage: string;
  shape: number[];
  axes: string;
  dtype: string;
}

const manifest = JSON.parse(readFileSync(join(CASE_DIR, "manifest.json"), "utf8")) as {
  case: string;
  dtype: string;
  tensors: TensorEntry[];
};

function npyHeader(path: string): { descr: string; shape: number[] } {
  const buf = readFileSync(path);
  const len = buf.readUInt16LE(8);
  const header = buf.subarray(10, 10 + len).toString("latin1");
  const descr = /'descr': '([^']+)'/.exec(header)?.[1] ?? "";
  const shape = (/'shape': \(([^)]*)\)/.exec(header)?.[1] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);
  return { descr, shape };
}

describe("tiny golden case", () => {
  it("is fp32 and covers every stage", () => {
    expect(manifest.case).toBe("tiny");
    expect(manifest.dtype).toBe("float32");
    expect(new Set(manifest.tensors.map((t) => t.stage))).toEqual(STAGES);
  });

  it.each(manifest.tensors.map((t) => [t.name, t] as const))("%s matches its .npy header", (_name, t) => {
    const { descr, shape } = npyHeader(join(CASE_DIR, t.file));
    expect(descr).toBe(t.dtype === "int32" ? "<i4" : "<f4");
    expect(shape).toEqual(t.shape);
    expect(t.axes.split(" ")).toHaveLength(t.shape.length);
    const bytes = t.shape.reduce((a, b) => a * b, 1) * 4;
    expect(statSync(join(CASE_DIR, t.file)).size).toBeGreaterThanOrEqual(bytes);
  });
});
