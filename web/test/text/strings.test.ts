// Plan 03 step 2: the string rules against Python, on every golden case's text stage and every line of the
// committed edge-case dump (tools/webgpu/text_reference.py).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { addTrailingSpace, chunkText, fixRefText, maxChars, pyStrip, utf8Length } from "../../src/text/strings";
import type { Manifest } from "../parity";

const ROOT = join(import.meta.dirname, "../../../testdata/webgpu");

export interface DumpLine {
  text: string;
  segments: string[];
  pinyin_calls: [string, string[]][];
  tokens: string[];
  ids: number[];
  fixed_ref: string;
  spaced_ref: string;
  utf8_len: number;
  chunks: Record<string, string[]>;
}
export interface Dump {
  versions: Record<string, string>;
  chunk_max_chars: number[];
  lines: DumpLine[];
}
export const edgeDump = JSON.parse(readFileSync(join(ROOT, "text/edge_cases.json"), "utf8")) as Dump;

const golden = readdirSync(ROOT).filter((d) => existsSync(join(ROOT, d, "manifest.json")));
const manifest = (name: string) => JSON.parse(readFileSync(join(ROOT, name, "manifest.json"), "utf8")) as Manifest;

describe("golden cases' text stage", () => {
  it.each(golden)("%s", (name) => {
    const m = manifest(name);
    const input = m.input as { ref_text: string; gen_text: string };
    const prep = m.prep as { ref_text: string; sample_rate: number };
    const text = m.text as { max_chars: number; chunks: string[]; ref_text_spaced: string };
    const speed = (m.settings as { speed: number }).speed;
    const samples = m.tensors.find((t) => t.name === "prep.audio")?.shape.at(-1) as number;

    const fixed = fixRefText(input.ref_text);
    expect(fixed).toBe(prep.ref_text);
    const mc = maxChars(fixed, samples, prep.sample_rate, speed);
    expect(mc).toBe(text.max_chars);
    const chunks = chunkText(input.gen_text, mc);
    expect(chunks).toEqual(text.chunks);
    const spaced = addTrailingSpace(fixed);
    expect(spaced).toBe(text.ref_text_spaced);
    expect(m.chunks.map((c) => c.joined_text)).toEqual(chunks.map((c) => spaced + c));
  });
});

describe("edge cases against Python", () => {
  const lines = edgeDump.lines;
  it("has the dump", () => expect(lines.length).toBeGreaterThan(200));
  it.each(lines.map((l, i) => [i, JSON.stringify(l.text).slice(0, 60), l] as const))("line %i %s", (_i, _label, l) => {
    expect(utf8Length(l.text)).toBe(l.utf8_len);
    expect(fixRefText(l.text)).toBe(l.fixed_ref);
    expect(addTrailingSpace(l.fixed_ref)).toBe(l.spaced_ref);
    for (const m of edgeDump.chunk_max_chars)
      expect(chunkText(l.text, m), `max_chars ${m}`).toEqual(l.chunks[String(m)]);
  });
});

describe("pyStrip", () => {
  const cp = (...codes: number[]) => String.fromCodePoint(...codes);
  it("strips Python's whitespace and keeps the rest", () => {
    expect(pyStrip(cp(0x1c, 0x85, 0x20) + "a" + cp(0x20, 0x3000))).toBe("a");
    const bom = cp(0xfeff); // not whitespace to Python
    expect(pyStrip(bom + "a" + bom)).toBe(bom + "a" + bom);
    expect(pyStrip(cp(0x200b) + "a")).toBe(cp(0x200b) + "a");
  });
});

// the Seed-TTS corpus (tools/webgpu/seedtts_corpus.py), local only
const corpusDir = join(ROOT, "text/corpus");
const corpora = ["seedtts_en", "seedtts_zh"].filter((c) => existsSync(join(corpusDir, `${c}.json`)));
export const loadCorpus = (name: string) => JSON.parse(readFileSync(join(corpusDir, `${name}.json`), "utf8")) as Dump;

describe.skipIf(corpora.length === 0)("corpus against Python", () => {
  it.each(corpora)("%s", (name) => {
    const dump = loadCorpus(name);
    const bad: string[] = [];
    for (const l of dump.lines) {
      if (utf8Length(l.text) !== l.utf8_len) bad.push(`utf8_len: ${l.text}`);
      if (fixRefText(l.text) !== l.fixed_ref) bad.push(`fixed_ref: ${l.text}`);
      if (addTrailingSpace(l.fixed_ref) !== l.spaced_ref) bad.push(`spaced_ref: ${l.text}`);
      for (const m of dump.chunk_max_chars)
        if (JSON.stringify(chunkText(l.text, m)) !== JSON.stringify(l.chunks[String(m)]))
          bad.push(`chunks ${m}: ${l.text}`);
    }
    expect(bad.slice(0, 10), `${bad.length} of ${dump.lines.length} lines differ`).toEqual([]);
  });
});
