// Plan 05 step 3: widening stored shards into the fp32 layout ORT reads, against tools/webgpu/weights_fixtures.py,
// which ran convert_precision.py's own save() (testdata/webgpu/weights/).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  checkFeatures,
  f16ToF32Bits,
  needsWidening,
  type Shard,
  type VariantInfo,
  widenShard,
} from "../../src/weights/variant";

const dir = join(import.meta.dirname, "../../../testdata/webgpu/weights");
const read = (path: string) => new Uint8Array(readFileSync(join(dir, path)));

describe("f16ToF32Bits", () => {
  it("equals numpy's conversion for all 65536 fp16 values, NaN payloads included", () => {
    const want = new Uint32Array(read("f16_to_f32.bin").buffer.slice(0));
    const got = f16ToF32Bits();
    let differ = 0;
    for (let h = 0; h < 65536; h++) if (got[h] !== want[h]) differ++;
    expect(differ).toBe(0);
  });
});

describe("widenShard", () => {
  const table = JSON.parse(readFileSync(join(dir, "tiny/tiny.json"), "utf8")) as { shards: Shard[] };

  it("covers every stored dtype", () => {
    const dtypes = new Set(table.shards.flatMap((s) => s.tensors.map((t) => t.dtype)));
    expect([...dtypes].sort()).toEqual(["f16", "f32", "i8", "raw"]);
  });

  for (const shard of table.shards) {
    it(`gives ${shard.file}'s layout byte for byte`, () => {
      const got = widenShard(read(`tiny/${shard.file}`), shard);
      const want = read(`tiny/layout/${shard.file}`);
      expect(got.byteLength).toBe(want.byteLength);
      expect(Buffer.from(got).equals(Buffer.from(want))).toBe(true);
    });
  }

  it("works on a stored buffer at an odd offset", () => {
    const shard = table.shards[0] as Shard;
    const stored = read(`tiny/${shard.file}`);
    const padded = new Uint8Array(stored.byteLength + 1);
    padded.set(stored, 1);
    const got = widenShard(padded.subarray(1), shard);
    expect(Buffer.from(got).equals(Buffer.from(read(`tiny/layout/${shard.file}`)))).toBe(true);
  });

  it("refuses a shard of the wrong size", () => {
    const shard = table.shards[0] as Shard;
    expect(() => widenShard(read(`tiny/${shard.file}`).subarray(1), shard)).toThrow(/expected/);
  });

  it("knows which shards need widening", () => {
    // tiny.data.2 has only f32 and raw tensors, but its stored offsets are packed closer than the layout's
    expect(table.shards.map(needsWidening)).toEqual([true, true, true, true]);
    const t = { name: "x", dtype: "raw" as const, count: 8, shape: [8], layoutLength: 64, storedLength: 64 };
    const same: Shard = {
      file: "x",
      layoutSize: 4160,
      storedSize: 4160,
      tensors: [{ ...t, layoutOffset: 4096, storedOffset: 4096 }],
    };
    expect(needsWidening(same)).toBe(false);
    expect(needsWidening({ ...same, tensors: [{ ...t, layoutOffset: 4096, storedOffset: 64 }] })).toBe(true);
    expect(needsWidening({ ...same, storedSize: 128 })).toBe(true);
  });
});

describe("checkFeatures", () => {
  const info = (features?: string[]): VariantInfo => ({ variant: "x", features, graphs: {} }) as unknown as VariantInfo;
  const adapter = (features: string[]) => ({ features: new Set(features) }) as unknown as GPUAdapter;

  it("refuses an fp16 variant on an adapter without shader-f16", async () => {
    await expect(checkFeatures(info(["shader-f16"]), adapter([]))).rejects.toThrow(/x needs shader-f16/);
  });

  it("accepts it where the adapter has the feature, and any variant that needs nothing", async () => {
    await expect(checkFeatures(info(["shader-f16"]), adapter(["shader-f16"]))).resolves.toBeUndefined();
    await expect(checkFeatures(info([]), adapter([]))).resolves.toBeUndefined();
    await expect(checkFeatures(info(), null)).resolves.toBeUndefined();
  });
});
