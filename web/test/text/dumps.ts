// The Python reference dumps (tools/webgpu/text_reference.py): the committed edge cases, and the local-only
// Seed-TTS corpus (tools/webgpu/seedtts_corpus.py), which is absent in CI.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const TESTDATA = join(import.meta.dirname, "../../../testdata/webgpu");

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
export const edgeDump = JSON.parse(readFileSync(join(TESTDATA, "text/edge_cases.json"), "utf8")) as Dump;

const corpusDir = join(TESTDATA, "text/corpus");
export const corpora = ["seedtts_en", "seedtts_zh"].filter((c) => existsSync(join(corpusDir, `${c}.json`)));
export const loadCorpus = (name: string) => JSON.parse(readFileSync(join(corpusDir, `${name}.json`), "utf8")) as Dump;
