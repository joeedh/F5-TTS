// Tokens and ids (plan 03 step 5): convert_char_to_pinyin and the vocab lookup from model/utils.py.
// TextFrontend loads the jieba wasm only for text with CJK ideographs, and the pinyin dictionary only for
// text with Han characters (pypinyin's RE_HANS, which also has 〇); other text needs neither.

import { hasHans, Pinyin } from "./pinyin";
import { Segmenter } from "./segment";
import { cutNoCjk, hasCjk } from "./segment_ascii";
import { customTranslate } from "./strings";

/** convert_char_to_pinyin's is_chinese: U+3100 to U+9FFF, wider and narrower than RE_HANS in places. */
const isChinese = (c: string) => c >= "\u3100" && c <= "\u9fff";

/** lazy_pinyin(text, style=TONE3, tone_sandhi=True). Text with no Han is one item, with no dictionary. */
function lazyPinyin(pinyin: Pinyin | undefined, text: string): string[] {
  if (pinyin) return pinyin.lazyPinyin(text);
  if (hasHans(text)) throw new Error("convertCharToPinyin: text has Han characters but no pinyin dictionary");
  return text ? [text] : [];
}

/**
 * convert_char_to_pinyin([text])[0] (polyphone=True). `cut` is rjieba.cut; `pinyin` may be left out for
 * text with no Han character.
 */
export function convertCharToPinyin(text: string, cut: (text: string) => string[], pinyin?: Pinyin): string[] {
  const tokens: string[] = [];
  for (const seg of cut(customTranslate(text))) {
    const chars = Array.from(seg);
    const bytes = new TextEncoder().encode(seg).length;
    if (bytes === chars.length) {
      // pure ASCII. Python's `in` on a string is a substring test, as includes() is
      const last = tokens.at(-1);
      if (last !== undefined && bytes > 1 && !" :'\"".includes(last)) tokens.push(" ");
      tokens.push(...chars);
    } else if (bytes === 3 * chars.length) {
      // all 3-byte characters: indexed per character, as upstream; Python raises IndexError if items run short
      const items = lazyPinyin(pinyin, seg);
      if (items.length < chars.length) throw new Error(`lazy_pinyin(${JSON.stringify(seg)}) gave too few items`);
      chars.forEach((c, i) => {
        if (isChinese(c)) tokens.push(" ");
        tokens.push(items[i] as string);
      });
    } else {
      for (const c of chars) {
        if ((c.codePointAt(0) as number) < 256) tokens.push(c);
        else if (isChinese(c)) tokens.push(" ", ...lazyPinyin(pinyin, c));
        else tokens.push(c);
      }
    }
  }
  return tokens;
}

/**
 * get_tokenizer(path, "custom"): Python reads the file in text mode, which turns \r\n and \r into \n, and
 * keys each line on line[:-1]. A later duplicate overwrites an earlier one, as in a dict.
 */
export function parseVocab(text: string): Map<string, number> {
  const vocab = new Map<string, number>();
  const lines = text.replace(/\r\n?/g, "\n").split(/(?<=\n)/);
  lines.forEach((line, i) => {
    if (line) vocab.set(Array.from(line).slice(0, -1).join(""), i);
  });
  return vocab;
}

/** list_str_to_idx for one text: vocab_char_map.get(c, 0), as int32. */
export function textToIds(tokens: string[], vocab: Map<string, number>): Int32Array {
  return Int32Array.from(tokens, (t) => vocab.get(t) ?? 0);
}

export interface TextFrontendLoaders {
  /** Default: Segmenter.load(), the wasm next to segment.ts. */
  segmenter?: () => Promise<Segmenter>;
  /** Default: Pinyin.load(), the dictionary next to pinyin.ts. */
  pinyin?: () => Promise<Pinyin>;
}

/** convertCharToPinyin, loading the jieba wasm and the pinyin dictionary the first time text needs them. */
export class TextFrontend {
  private segmenter?: Promise<Segmenter>;
  private pinyin?: Promise<Pinyin>;

  constructor(private readonly loaders: TextFrontendLoaders = {}) {}

  async tokens(text: string): Promise<string[]> {
    const translated = customTranslate(text);
    // both start at once, so the first Chinese sentence waits for the slower load, not the sum
    const [segmenter, pinyin] = await Promise.all([
      hasCjk(translated) ? this.loadSegmenter() : undefined,
      hasHans(translated) ? this.loadPinyin() : undefined,
    ]);
    return convertCharToPinyin(text, segmenter ? (t) => segmenter.cut(t) : cutNoCjk, pinyin);
  }

  private loadSegmenter(): Promise<Segmenter> {
    this.segmenter ??= (this.loaders.segmenter ?? (() => Segmenter.load()))();
    return this.segmenter;
  }

  private loadPinyin(): Promise<Pinyin> {
    this.pinyin ??= (this.loaders.pinyin ?? (() => Pinyin.load()))();
    return this.pinyin;
  }
}
