// Plan 04 step 1: the chunk plan and time grid against the golden cases (through the parity runner's checks)
// and against upstream's own code on cases the golden data never reaches (tools/webgpu/plan_fixtures.py).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { planChunks, timeGrid } from "../../src/pipeline/plan";
import { Pinyin } from "../../src/text/pinyin";
import { Segmenter } from "../../src/text/segment";
import { addTrailingSpace } from "../../src/text/strings";
import { convertCharToPinyin } from "../../src/text/tokens";
import { compare, fsReader, loadCase, loadTolerances, type Reader, type Tolerances } from "../parity";

const TESTDATA = join(import.meta.dirname, "../../../testdata/webgpu");
const SRC = join(import.meta.dirname, "../../src/text");
const golden = readdirSync(TESTDATA).filter((d) => existsSync(join(TESTDATA, d, "manifest.json")));

interface Settings {
  nfe_step: number;
  sway_sampling_coef: number | null;
  speed: number;
  fix_duration: number | null;
  cross_fade_duration: number;
}

let read: Reader;
let tolerances: Tolerances;
let seg: Segmenter;
const pinyin = Pinyin.parse(readFileSync(join(SRC, "pinyin_dict.txt"), "utf8"));
beforeAll(async () => {
  read = await fsReader(TESTDATA);
  tolerances = await loadTolerances(read);
  seg = await Segmenter.load(readFileSync(join(SRC, "jieba.wasm")));
});

describe("golden cases", () => {
  it.each(golden)("%s: durations, reference lengths, speeds and time grid", async (name) => {
    const g = await loadCase(read, name);
    const m = g.manifest;
    const s = m.settings as unknown as Settings;
    const text = m.text as { chunks: string[]; ref_text_spaced: string };
    const chunks = m.chunks as { duration: number; ref_audio_len: number; local_speed: number; tokens: string[] }[];
    const plans = planChunks({
      refSamples: g.entry("audio.ref").shape.at(-1) as number,
      refTextSpaced: text.ref_text_spaced,
      chunks: text.chunks,
      tokenCounts: chunks.map((c) => c.tokens.length),
      speed: s.speed,
      fixDuration: s.fix_duration,
      crossFadeDuration: s.cross_fade_duration,
    });
    expect(plans.map((p) => [p.duration, p.refAudioLen, p.localSpeed])).toEqual(
      chunks.map((c) => [c.duration, c.ref_audio_len, c.local_speed]),
    );
    const t = timeGrid(s.nfe_step, s.sway_sampling_coef);
    for (let c = 0; c < chunks.length; c++) {
      const results = await compare(g, tolerances, `c${c}.plan.t`, { dtype: "float32", shape: [t.length], data: t });
      expect(results.map((r) => [r.check, r.pass, r.metrics?.maxAbs])).toEqual([["plan.t", true, expect.any(Number)]]);
    }
  });
});

interface FixtureCase {
  ref_samples: number;
  ref_text: string;
  chunks: string[];
  settings: Partial<Settings>;
  results: { duration_asked: number; tokens: number; duration: number; t: number[] }[];
}
const fixtures = JSON.parse(readFileSync(join(TESTDATA, "pipeline/plan_cases.json"), "utf8")) as Record<
  string,
  FixtureCase
>;

describe("upstream's code on cases the golden data never reaches", () => {
  it.each(Object.entries(fixtures))("%s", (_name, c) => {
    const refTextSpaced = addTrailingSpace(c.ref_text);
    const tokenCounts = c.chunks.map((ch) => convertCharToPinyin(refTextSpaced + ch, (t) => seg.cut(t), pinyin).length);
    expect(tokenCounts).toEqual(c.results.map((r) => r.tokens));
    const plans = planChunks({
      refSamples: c.ref_samples,
      refTextSpaced,
      chunks: c.chunks,
      tokenCounts,
      speed: c.settings.speed,
      fixDuration: c.settings.fix_duration,
      crossFadeDuration: c.settings.cross_fade_duration,
    });
    expect(plans.map((p) => p.duration)).toEqual(c.results.map((r) => r.duration));
    const sway = "sway_sampling_coef" in c.settings ? (c.settings.sway_sampling_coef ?? null) : -1;
    const t = timeGrid(c.settings.nfe_step ?? 32, sway);
    for (const r of c.results) {
      const maxAbs = Math.max(...r.t.map((v, i) => Math.abs(v - (t[i] as number))));
      expect(t.length).toBe(r.t.length);
      expect(maxAbs, "plan.t tolerance").toBeLessThanOrEqual(1e-6);
    }
  });

  it("exercises the minimum-duration rule", () => {
    const r = fixtures["min-duration"]?.results[0];
    expect(r && r.duration_asked < r.duration && r.duration === r.tokens + 1).toBe(true);
  });
});

describe("long chunks (D7)", () => {
  it("refuses a chunk past 8192 frames, by computed duration", () => {
    const base = { refSamples: 5 * 24000, refTextSpaced: "Some call me nature. ", tokenCounts: [100] };
    expect(() => planChunks({ ...base, chunks: ["Hello there."], fixDuration: 90 })).toThrow(
      /chunk 1 needs 8437 frames/,
    );
    expect(() => planChunks({ ...base, chunks: ["word ".repeat(400)] })).toThrow(/more than 8192/);
    expect(planChunks({ ...base, chunks: ["Hello there."], fixDuration: 87 })[0]?.duration).toBe(8156);
  });
});
