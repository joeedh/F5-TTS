// Upstream's text string rules (plan 03 step 2), from infer/utils_infer.py: the reference-text fix-up, the
// trailing-space rule, max_chars and chunk_text. Lengths are UTF-8 bytes, as Python's len(s.encode()).
// Whitespace is Python's set (str.isspace, re's \s and strip() agree), never JS's \s or trim(), which
// differ at U+001C to U+001F, U+0085 and U+FEFF.

/** The 29 code points Python treats as whitespace (Python 3.13 and 3.15). */
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const LEADING = new RegExp(`^[${PY_SPACE}]+`, "u");
const TRAILING = new RegExp(`[${PY_SPACE}]+$`, "u");
/** chunk_text's split: whitespace after ASCII punctuation, or nothing after CJK punctuation. */
const SENTENCE_SPLIT = new RegExp(`(?<=[;:,.!?])[${PY_SPACE}]+|(?<=[；：，。！？])`, "u");

/** Python's str.strip() with no arguments. */
export function pyStrip(s: string): string {
  return s.replace(LEADING, "").replace(TRAILING, "");
}

/** Bytes in UTF-8, by code point; a lone surrogate counts 3 bytes, as it would encode to U+FFFD. */
export function utf8Length(s: string): number {
  let n = 0;
  for (const ch of s) {
    const c = ch.codePointAt(0) as number;
    n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  }
  return n;
}

/** True if the last code point is 1 byte in UTF-8 (ASCII); false for an empty string. */
function endsInAscii(s: string): boolean {
  const last = Array.from(s).at(-1);
  return last !== undefined && (last.codePointAt(0) as number) < 0x80;
}

/** The end of preprocess_ref_audio_text: end the reference text with ". " or "。". */
export function fixRefText(refText: string): string {
  if (!refText.endsWith(". ") && !refText.endsWith("。")) return refText + (refText.endsWith(".") ? " " : ". ");
  return refText;
}

/** infer_batch_process: a reference text whose last character is ASCII gets a trailing space. */
export function addTrailingSpace(refText: string): string {
  return endsInAscii(refText) ? `${refText} ` : refText;
}

/**
 * infer_process's chunk size, in UTF-8 bytes: int(len(ref_text.encode("utf-8")) / (n / sr) * (22 - n / sr)
 * * speed), in the same order. `fixedRefText` is after fixRefText and before addTrailingSpace; `refSamples`
 * and `sampleRate` are the prepared reference's, before resampling.
 */
export function maxChars(fixedRefText: string, refSamples: number, sampleRate: number, speed = 1): number {
  return Math.trunc((utf8Length(fixedRefText) / (refSamples / sampleRate)) * (22 - refSamples / sampleRate) * speed);
}

/** chunk_text: joins sentences while a chunk stays within `maxChars` UTF-8 bytes. */
export function chunkText(text: string, maxChars = 135): string[] {
  const chunks: string[] = [];
  let current = "";
  for (const sentence of text.split(SENTENCE_SPLIT)) {
    if (!sentence) continue;
    const piece = endsInAscii(sentence) ? `${sentence} ` : sentence;
    if (utf8Length(current) + utf8Length(sentence) <= maxChars) current += piece;
    else {
      if (current) chunks.push(pyStrip(current));
      current = piece;
    }
  }
  if (current) chunks.push(pyStrip(current));
  return chunks;
}
