// Plan 03 step 5: tokens and ids against Python, on the golden cases (their c*.dit.text_ids, through the
// parity runner's exact check), the edge cases and the corpus.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { Pinyin } from "../../src/text/pinyin";
import { Segmenter } from "../../src/text/segment";
import { convertCharToPinyin, parseVocab, TextFrontend, textToIds } from "../../src/text/tokens";
import { compare, fsReader, loadCase, loadTolerances, type Reader } from "../parity";
import { corpora, type DumpLine, edgeDump, loadCorpus, TESTDATA } from "./dumps";

const SRC = join(import.meta.dirname, "../../src/text");
const vocabText = readFileSync(join(import.meta.dirname, "../../../src/f5_tts/infer/examples/vocab.txt"), "utf8");
const vocab = parseVocab(vocabText);
const pinyin = Pinyin.parse(readFileSync(join(SRC, "pinyin_dict.txt"), "utf8"));
let seg: Segmenter;
let read: Reader;
beforeAll(async () => {
  seg = await Segmenter.load(readFileSync(join(SRC, "jieba.wasm")));
  read = await fsReader(TESTDATA);
});
const tokensOf = (text: string) => convertCharToPinyin(text, (t) => seg.cut(t), pinyin);

describe("vocab", () => {
  it("has upstream's 2545 entries, space at 0, and 𠮶 last", () => {
    expect(vocab.size).toBe(2545);
    expect(vocab.get(" ")).toBe(0);
    expect([...vocab.entries()].at(-1)).toEqual(["\u{20bb6}", 2544]);
  });
  it("reads CRLF, LF and CR files alike, as Python's text mode does", () => {
    expect(parseVocab(vocabText.replace(/\r?\n/g, "\r\n"))).toEqual(vocab);
    expect(parseVocab(vocabText.replace(/\r?\n/g, "\n"))).toEqual(vocab);
    expect(parseVocab("a\rb\n")).toEqual(
      new Map([
        ["a", 0],
        ["b", 1],
      ]),
    );
    expect(parseVocab("a\nbc")).toEqual(
      new Map([
        ["a", 0],
        ["b", 1],
      ]),
    ); // line[:-1] on an unterminated line
  });
});

const golden = readdirSync(TESTDATA).filter((d) => existsSync(join(TESTDATA, d, "manifest.json")));

describe("golden cases", () => {
  it.each(golden)("%s: tokens and text_ids", async (name) => {
    const [g, tolerances] = [await loadCase(read, name), await loadTolerances(read)];
    const chunks = g.manifest.chunks as { joined_text: string; tokens: string[] }[];
    for (const [c, chunk] of chunks.entries()) {
      const tokens = tokensOf(chunk.joined_text);
      expect(tokens).toEqual(chunk.tokens);
      const ids = textToIds(tokens, vocab);
      const results = await compare(g, tolerances, `c${c}.dit.text_ids`, {
        dtype: "int32",
        shape: [1, ids.length],
        data: ids,
      });
      expect(results.map((r) => [r.check, r.pass, r.error])).toEqual([["exact", true, undefined]]);
    }
  });
});

function differences(lines: DumpLine[]): string[] {
  return lines.flatMap((l) => {
    const tokens = tokensOf(l.text);
    const ids = Array.from(textToIds(tokens, vocab));
    if (JSON.stringify(tokens) !== JSON.stringify(l.tokens)) return [`tokens: ${JSON.stringify(l.text)}`];
    if (JSON.stringify(ids) !== JSON.stringify(l.ids)) return [`ids: ${JSON.stringify(l.text)}`];
    return [];
  });
}

describe("tokens and ids against Python", () => {
  it.each(edgeDump.lines.map((l) => [JSON.stringify(l.text).slice(0, 60), l] as const))("edge case %s", (_label, l) => {
    const tokens = tokensOf(l.text);
    expect(tokens).toEqual(l.tokens);
    expect(Array.from(textToIds(tokens, vocab))).toEqual(l.ids);
  });

  it.skipIf(corpora.length === 0).each(corpora)("corpus %s", (name) => {
    const lines = loadCorpus(name).lines;
    const bad = differences(lines);
    expect(bad.slice(0, 10), `${bad.length} of ${lines.length} lines differ`).toEqual([]);
  });
});

describe("TextFrontend", () => {
  it("loads jieba only for CJK and pinyin only for Han, once each", async () => {
    let jiebaLoads = 0;
    let pinyinLoads = 0;
    const fe = new TextFrontend({
      segmenter: async () => (jiebaLoads++, seg),
      pinyin: async () => (pinyinLoads++, pinyin),
    });
    expect(await fe.tokens("Hello, world.")).toEqual(tokensOf("Hello, world."));
    expect([jiebaLoads, pinyinLoads]).toEqual([0, 0]);
    expect(await fe.tokens("〇 and ㄅ")).toEqual(tokensOf("〇 and ㄅ")); // 〇 is Han to pypinyin only
    expect([jiebaLoads, pinyinLoads]).toEqual([0, 1]);
    expect(await fe.tokens("你好")).toEqual(tokensOf("你好"));
    expect(await fe.tokens("世界")).toEqual(tokensOf("世界"));
    expect([jiebaLoads, pinyinLoads]).toEqual([1, 1]);
  });

  it("matches the full path on every edge case and corpus line", async () => {
    const fe = new TextFrontend({ segmenter: async () => seg, pinyin: async () => pinyin });
    const lines = [...edgeDump.lines, ...corpora.flatMap((c) => loadCorpus(c).lines)];
    const bad: string[] = [];
    for (const l of lines) if (JSON.stringify(await fe.tokens(l.text)) !== JSON.stringify(l.tokens)) bad.push(l.text);
    expect(bad.slice(0, 10), `${bad.length} of ${lines.length} lines differ`).toEqual([]);
  });

  it("refuses a lone surrogate on both paths, as Python raises UnicodeEncodeError", async () => {
    const fe = new TextFrontend({ segmenter: async () => seg, pinyin: async () => pinyin });
    const lone = String.fromCharCode(0xd800);
    await expect(fe.tokens(`a${lone}b`)).rejects.toThrow(/surrogate/);
    await expect(fe.tokens(`你${lone}`)).rejects.toThrow(/surrogate/);
  });

  it("refuses Han text without a dictionary, as no fallback would match", () => {
    expect(() => convertCharToPinyin("你", (t) => seg.cut(t))).toThrow(/pinyin dictionary/);
  });
});
