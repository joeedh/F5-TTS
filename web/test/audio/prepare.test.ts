// Plan 02 step 2: reference preparation against the golden prep.audio (exact) and audio.ref, for the
// committed tiny case, the real cases where make_golden.py has built them, and the synthetic fixtures.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { downmixRmsResample, trimAndClip } from "../../src/audio/prepare";
import { roundHalfEven, type Pcm } from "../../src/audio/pydub";
import { parseWav } from "../../src/audio/wav";
import {
  compare,
  fsReader,
  judge,
  loadCase,
  loadTolerances,
  metrics,
  parseNpy,
  toArrayBuffer,
  type GoldenCase,
  type Tensor,
} from "../parity";

const ROOT = join(import.meta.dirname, "../../../testdata/webgpu");
const REAL = ["en-short", "zh-short", "en-multi", "en-multi-fixdur", "en-prep", "en-epss16"];
const built = REAL.filter((c) => existsSync(join(ROOT, c, "prep.audio.npy")));

/** (channels, frames) float in [-1, 1), as soundfile reads 16-bit PCM, to interleaved int16. */
export function toPcm(t: Tensor, sampleRate: number): Pcm {
  const [channels = 0, frames = 0] = t.shape;
  const samples = new Int16Array(channels * frames);
  for (let c = 0; c < channels; c++)
    for (let i = 0; i < frames; i++) samples[i * channels + c] = Math.round((t.data[c * frames + i] as number) * 32768);
  return { samples, channels, sampleRate };
}

/** Interleaved int16 back to the (channels, frames) float layout of the golden tensors. */
export function fromPcm(pcm: Pcm): Tensor {
  const { samples, channels } = pcm;
  const frames = samples.length / channels;
  const data = new Float32Array(samples.length);
  for (let c = 0; c < channels; c++)
    for (let i = 0; i < frames; i++) data[c * frames + i] = (samples[i * channels + c] as number) / 32768;
  return { dtype: "float32", shape: [channels, frames], data };
}

async function load(caseName: string): Promise<GoldenCase> {
  return loadCase(await fsReader(ROOT), caseName);
}

async function checkPrep(golden: GoldenCase) {
  const input = toPcm(await golden.tensor("input.wav"), (golden.manifest.input as { sample_rate: number }).sample_rate);
  const out = trimAndClip(input);
  expect(out.sampleRate).toBe((golden.manifest.prep as { sample_rate: number }).sample_rate);
  const tolerances = await loadTolerances(await fsReader(ROOT));
  const [r] = await compare(golden, tolerances, "prep.audio", fromPcm(out));
  expect(r?.check).toBe("exact");
  expect(r?.pass, r?.error).toBe(true);
}

async function checkReference(golden: GoldenCase) {
  const prepared = toPcm(
    await golden.tensor("prep.audio"),
    (golden.manifest.prep as { sample_rate: number }).sample_rate,
  );
  const ref = downmixRmsResample(prepared);
  expect(ref.raised).toBe(golden.manifest.audio.raised);
  expect(Math.abs(ref.rms - golden.manifest.audio.rms)).toBeLessThan(1e-7);
  const tolerances = await loadTolerances(await fsReader(ROOT));
  const actual: Tensor = { dtype: "float32", shape: [1, ref.audio.length], data: ref.audio };
  const [r] = await compare(golden, tolerances, "audio.ref", actual);
  expect(r?.check).toBe("audio.resample");
  expect(r?.pass, `${r?.error ?? ""} ${JSON.stringify(r?.metrics)}`).toBe(true);
}

describe("trimAndClip", () => {
  it("gives the tiny case's prep.audio exactly", async () => checkPrep(await load("tiny")));
  it.skipIf(built.length === 0).each(built)(
    "gives %s's prep.audio exactly",
    async (c) => checkPrep(await load(c)),
    60_000,
  );
});

describe("downmixRmsResample", () => {
  it("gives the tiny case's audio.ref", async () => checkReference(await load("tiny")));
  it.skipIf(built.length === 0).each(built)("gives %s's audio.ref", async (c) => checkReference(await load(c)), 60_000);
});

const FIXTURES = join(ROOT, "audio-fixtures", "prep");
interface FixtureEntry {
  note: string;
  prep: { sample_rate: number; channels: number; frames: number };
  audio: { rms: number; raised: boolean; ref: boolean };
}
const fixtures = JSON.parse(readFileSync(join(FIXTURES, "manifest.json"), "utf8")) as Record<string, FixtureEntry>;
const wav = (file: string) => parseWav(toArrayBuffer(readFileSync(join(FIXTURES, file))));

describe("synthetic fixtures (export_audio_fixtures.py)", () => {
  it.each(Object.keys(fixtures))("%s: trimAndClip gives upstream's output exactly", (name) => {
    const want = wav(`${name}.out.wav`);
    const got = trimAndClip(wav(`${name}.in.wav`));
    expect([got.sampleRate, got.channels, got.samples.length]).toEqual([
      want.sampleRate,
      want.channels,
      want.samples.length,
    ]);
    const first = got.samples.findIndex((v, i) => v !== want.samples[i]);
    expect(first, `first differing sample ${first}`).toBe(-1);
  });

  it.each(Object.keys(fixtures).filter((n) => fixtures[n]?.audio.ref))(
    "%s: downmixRmsResample gives audio.ref",
    async (name) => {
      const entry = fixtures[name] as FixtureEntry;
      const ref = downmixRmsResample(wav(`${name}.out.wav`));
      expect(ref.raised).toBe(entry.audio.raised);
      expect(Math.abs(ref.rms - entry.audio.rms) / entry.audio.rms).toBeLessThan(1e-6);
      const want = parseNpy(toArrayBuffer(readFileSync(join(FIXTURES, `${name}.ref.npy`))));
      expect(want.shape).toEqual([1, ref.audio.length]);
      // the golden-case tolerance through the parity runner's judge; audio.resample has no tiny override
      const tiny = await load("tiny");
      const r = judge(
        tiny,
        await loadTolerances(await fsReader(ROOT)),
        "audio",
        name,
        "audio.resample",
        metrics(ref.audio, want.data),
      );
      expect(r.pass, JSON.stringify(r.metrics)).toBe(true);
    },
  );
});

describe("roundHalfEven", () => {
  it("rounds as Python's round()", () => {
    expect([0.5, 1.5, 2.5, 750.5, 751.5, 2.4, 2.6].map(roundHalfEven)).toEqual([0, 2, 2, 750, 752, 2, 3]);
  });
});
