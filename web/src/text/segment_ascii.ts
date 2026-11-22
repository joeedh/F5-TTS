// rjieba.cut for text with no CJK ideographs, without the jieba wasm (plan 03 step 3, D2). For such text,
// jieba-rs 0.11.0 reduces to this: its route always takes the five ASCII dictionary words, and the HMM's
// skip splitter cuts the rest of each block. Proven against rjieba and the wasm in test/text/segment.test.ts.

/** jieba-rs's is_cjk: the main CJK Unified Ideographs block and its extensions. */
const CJK = /[一-鿿㐀-䶿豈-﫿\u{20000}-\u{2a6df}\u{2a700}-\u{2ebef}\u{2f800}-\u{2fa1f}]/u;
/** is_han_default's ASCII part: the characters that stay inside a block. */
const BLOCK = /[A-Za-z0-9+#&._%-]+/g;
/** The dictionary words made only of block characters. None can overlap another, so a left-to-right scan
 * finds what the max-probability route takes: ln(3) - L beats -L per character, as L = ln(total) > 0. */
const ASCII_WORDS = /AT&T|C\+\+|C#|c\+\+|c#/g;
const ALNUM = /[A-Za-z0-9]/;

/** True if jieba would see a CJK ideograph, so the wasm is needed. */
export function hasCjk(text: string): boolean {
  return CJK.test(text);
}

/** skip_match_len: an alphanumeric run, then connector-plus-run groups (. _ -), then an optional %. */
function skipMatchLength(s: string, start: number): number {
  let end = start;
  for (;;) {
    while (end < s.length && ALNUM.test(s[end] as string)) end++;
    if (end + 1 < s.length && ".-_".includes(s[end] as string) && ALNUM.test(s[end + 1] as string)) end++;
    else break;
  }
  if (s[end] === "%") end++;
  return end - start;
}

/** HmmSkipSplitter: alternating gaps (each one token) and skip_match_len matches. */
function skipSplit(run: string, out: string[]): void {
  let pos = 0;
  while (pos < run.length) {
    let cursor = pos;
    while (cursor < run.length && !ALNUM.test(run[cursor] as string)) cursor++;
    if (cursor === run.length) {
      out.push(run.slice(pos));
      return;
    }
    const len = skipMatchLength(run, cursor);
    if (cursor > pos) out.push(run.slice(pos, cursor));
    out.push(run.slice(cursor, cursor + len));
    pos = cursor + len;
  }
}

/** Characters outside blocks are one token each, except that \r\n is one token. */
function splitUnmatched(gap: string, out: string[]): void {
  const chars = Array.from(gap);
  for (let i = 0; i < chars.length; i++) {
    if (chars[i] === "\r" && chars[i + 1] === "\n") {
      out.push("\r\n");
      i++;
    } else out.push(chars[i] as string);
  }
}

/** rjieba.cut(text) for text where hasCjk(text) is false. */
export function cutNoCjk(text: string): string[] {
  if (hasCjk(text)) throw new Error("cutNoCjk: text has CJK ideographs; use the jieba wasm");
  const out: string[] = [];
  let pos = 0;
  for (const block of text.matchAll(BLOCK)) {
    if (block.index > pos) splitUnmatched(text.slice(pos, block.index), out);
    let x = 0;
    for (const word of block[0].matchAll(ASCII_WORDS)) {
      if (word.index > x) skipSplit(block[0].slice(x, word.index), out);
      out.push(word[0]);
      x = word.index + word[0].length;
    }
    if (x < block[0].length) skipSplit(block[0].slice(x), out);
    pos = block.index + block[0].length;
  }
  if (pos < text.length) splitUnmatched(text.slice(pos), out);
  return out;
}
