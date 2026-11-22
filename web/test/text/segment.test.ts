// Plan 03 step 3: our jieba-rs 0.11.0 wasm against rjieba's segments, on the edge cases and the corpus.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { Segmenter } from "../../src/text/segment";
import { cutNoCjk, hasCjk } from "../../src/text/segment_ascii";
import { customTranslate } from "../../src/text/strings";
import { corpora, type DumpLine, edgeDump, loadCorpus } from "./dumps";

let seg: Segmenter;
beforeAll(async () => {
  seg = await Segmenter.load(readFileSync(join(import.meta.dirname, "../../src/text/jieba.wasm")));
});

describe("jieba wasm against rjieba", () => {
  it.each(edgeDump.lines.map((l, i) => [i, JSON.stringify(l.text).slice(0, 60), l] as const))(
    "line %i %s",
    (_i, _label, l) => expect(seg.cut(customTranslate(l.text))).toEqual(l.segments),
  );

  it.skipIf(corpora.length === 0).each(corpora)("corpus %s", (name) => {
    const dump = loadCorpus(name);
    const bad = dump.lines.filter(
      (l) => JSON.stringify(seg.cut(customTranslate(l.text))) !== JSON.stringify(l.segments),
    );
    expect(
      bad.slice(0, 5).map((l) => l.text),
      `${bad.length} of ${dump.lines.length} lines differ`,
    ).toEqual([]);
  });

  it("handles empty text, memory growth and lone surrogates", () => {
    expect(seg.cut("")).toEqual([]);
    const long = "我来到北京清华大学。".repeat(20000);
    expect(seg.cut(long).join("")).toBe(long);
    expect(() => seg.cut("a\ud800")).toThrow(/surrogate/);
  });
});

describe("pure-TS path for text with no CJK (D2)", () => {
  const noCjk = (lines: DumpLine[]) => lines.filter((l) => !hasCjk(customTranslate(l.text)));

  it.each(noCjk(edgeDump.lines).map((l) => [JSON.stringify(l.text).slice(0, 60), l] as const))(
    "edge case %s",
    (_label, l) => expect(cutNoCjk(customTranslate(l.text))).toEqual(l.segments),
  );

  it.skipIf(corpora.length === 0).each(corpora)("corpus %s", (name) => {
    const lines = noCjk(loadCorpus(name).lines);
    const bad = lines.filter((l) => JSON.stringify(cutNoCjk(customTranslate(l.text))) !== JSON.stringify(l.segments));
    expect(
      bad.slice(0, 5).map((l) => l.text),
      `${bad.length} of ${lines.length} lines differ`,
    ).toEqual([]);
  });

  it("matches the wasm on random strings of the characters that matter", () => {
    // block characters, the dictionary words' pieces, separators, CR and LF, and non-CJK non-ASCII
    const pieces = [
      "a",
      "Z",
      "0",
      "9",
      "+",
      "#",
      "&",
      ".",
      "_",
      "%",
      "-",
      "C",
      "c",
      "AT",
      "T",
      "++",
      " ",
      "\r",
      "\n",
      "'",
      ",",
      "!",
      "é",
      "〇",
      "ㄅ",
      "\u{1f600}",
      " ",
      "\t",
    ];
    let seed = 1;
    const rand = (n: number) => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed % n);
    for (let t = 0; t < 20000; t++) {
      let s = "";
      for (let k = rand(12) + 1; k > 0; k--) s += pieces[rand(pieces.length)];
      expect(cutNoCjk(s), JSON.stringify(s)).toEqual(seg.cut(s));
    }
  });

  it("refuses text with CJK", () => expect(() => cutNoCjk("a一")).toThrow(/CJK/));
});
