// Plan 03 step 4: the pypinyin port against pypinyin, on every lazy_pinyin call convert_char_to_pinyin made
// in the dumps, and on the local pinyin cases (tools/webgpu/pinyin_reference.py).
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Pinyin } from "../../src/text/pinyin";
import { corpora, edgeDump, loadCorpus, TESTDATA } from "./dumps";

const pinyin = Pinyin.parse(readFileSync(join(import.meta.dirname, "../../src/text/pinyin_dict.txt"), "utf8"));
const casesFile = join(TESTDATA, "text/corpus/pinyin_cases.json");

function mismatches(calls: [string, string[]][]): string[] {
  return calls
    .filter(([text, want]) => JSON.stringify(pinyin.lazyPinyin(text)) !== JSON.stringify(want))
    .map(([text, want]) => `${text}: ${JSON.stringify(pinyin.lazyPinyin(text))}, pypinyin ${JSON.stringify(want)}`);
}

describe("lazy_pinyin against pypinyin", () => {
  it.each(
    edgeDump.lines.filter((l) => l.pinyin_calls.length).map((l) => [JSON.stringify(l.text).slice(0, 60), l] as const),
  )("edge case %s", (_label, l) => expect(mismatches(l.pinyin_calls)).toEqual([]));

  it.skipIf(corpora.length === 0).each(corpora)("corpus %s", (name) => {
    const calls = loadCorpus(name).lines.flatMap((l) => l.pinyin_calls);
    const bad = mismatches(calls);
    expect(bad.slice(0, 10), `${bad.length} of ${calls.length} calls differ`).toEqual([]);
  });

  it.skipIf(!existsSync(casesFile))("every dictionary entry and random mixes", () => {
    const { cases } = JSON.parse(readFileSync(casesFile, "utf8")) as { cases: [string, string[]][] };
    const bad = mismatches(cases);
    expect(bad.slice(0, 10), `${bad.length} of ${cases.length} cases differ`).toEqual([]);
  });

  it("gives [] for empty text and keeps a non-Han run whole", () => {
    expect(pinyin.lazyPinyin("")).toEqual([]);
    expect(pinyin.lazyPinyin("，。")).toEqual(["，。"]);
  });

  it("accepts a CRLF dictionary", () => {
    const crlf = Pinyin.parse("# c\r\n中\tzhōng\r\n");
    expect(crlf.lazyPinyin("中")).toEqual(["zhong1"]);
  });
});
