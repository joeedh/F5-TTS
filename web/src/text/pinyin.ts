// pypinyin 0.55.0's lazy_pinyin(text, style=TONE3, tone_sandhi=True), as convert_char_to_pinyin calls it
// (plan 03 step 4, D3). Each function mirrors the pypinyin function it names, quirks included; the
// algorithm is written out in docs/research/webgpu-text-frontend.md. Data: pinyin_dict.txt.

/** The built dictionary, for a browser fetch; Vite rewrites this to the hashed asset. */
export const PINYIN_DICT_URL = new URL("./pinyin_dict.txt", import.meta.url);

/** constants.RE_HANS's character class (the UCS-4 build), copied range for range. */
const HAN =
  /[〇-礼㐀-䶿一-鿿豈-﫿\u{20000}-\u{2a6df}\u{2a703}-\u{2b73f}\u{2b740}-\u{2b81d}\u{2b825}-\u{2bf6e}\u{2c029}-\u{2ce93}\u{2d016}\u{2d11b}-\u{2ebd9}\u{2f80a}-\u{2fa1f}\u{30000}-\u{3134a}\u{300f7}-\u{31288}\u{30edd}\u{30ede}\u{31350}-\u{32389}]/u;
const isHan = (c: string) => HAN.test(c);

/** Python's \d on str patterns: any Unicode decimal digit. */
const DIGIT = /\p{Nd}/u;
const DIGITS = /\p{Nd}/gu;

// phonetic_symbol.phonetic_symbol, in its order; four keys are a letter plus a combining mark
const PHONETIC_SYMBOL: [string, string][] = [
  ["ā", "a1"], ["á", "a2"], ["ǎ", "a3"], ["à", "a4"], ["ē", "e1"], ["é", "e2"], ["ě", "e3"], ["è", "e4"],
  ["ō", "o1"], ["ó", "o2"], ["ǒ", "o3"], ["ò", "o4"], ["ī", "i1"], ["í", "i2"], ["ǐ", "i3"], ["ì", "i4"],
  ["ū", "u1"], ["ú", "u2"], ["ǔ", "u3"], ["ù", "u4"], ["ü", "v"], ["ǖ", "v1"], ["ǘ", "v2"], ["ǚ", "v3"],
  ["ǜ", "v4"], ["ń", "n2"], ["ň", "n3"], ["ǹ", "n4"], ["m̄", "m1"], ["ḿ", "m2"], ["m̀", "m4"],
  ["ê̄", "ê1"], ["ế", "ê2"], ["ê̌", "ê3"], ["ề", "ê4"],
]; // prettier-ignore
const SINGLE_SYMBOL = new Map(PHONETIC_SYMBOL.filter(([k]) => k.length === 1));
const MULTI_SYMBOL = PHONETIC_SYMBOL.filter(([k]) => k.length > 1);
const RE_PHONETIC_SYMBOL = new RegExp(`[${[...SINGLE_SYMBOL.keys()].join("")}]`, "gu");
const SYMBOL_REVERSE = new Map(PHONETIC_SYMBOL.map(([k, v]) => [v, k]));
/** style._constants.RE_TONE3; Python's $ also matches before a final \n, which no pinyin has. */
const RE_TONE3 = /^([a-zêü]+)([1-5])([a-zêü]*)$/u;
/** tone2_to_tone's regex: RE_TONE2 without its $. */
const RE_TONE2_ANYWHERE = /([aeoiuvnmêü])([1-5])/gu;

/** style._utils.replace_symbol_to_number */
function replaceSymbolToNumber(pinyin: string): string {
  let value = pinyin.replace(RE_PHONETIC_SYMBOL, (s) => SINGLE_SYMBOL.get(s) as string);
  for (const [symbol, to] of MULTI_SYMBOL) value = value.replaceAll(symbol, to);
  return value;
}

/** style.tone.ToneConverter.to_tone3, which Style.TONE3 is registered to. */
export function toTone3Style(pinyin: string): string {
  return replaceSymbolToNumber(pinyin).replace(RE_TONE3, "$1$3$2");
}

/** style._tone_rule.right_mark_index */
function rightMarkIndex(p: string): number | undefined {
  if (p.includes("iou")) return p.indexOf("u");
  if (p.includes("uei")) return p.indexOf("i");
  if (p.includes("uen")) return p.indexOf("u");
  for (const group of [
    ["a", "o", "e"],
    ["iu", "ui"],
    ["i", "u", "v", "ü"],
    ["n", "m", "ê"],
  ])
    for (const c of group) if (p.includes(c)) return p.indexOf(c) + c.length - 1;
  return undefined;
}

/** _tone_convert.tone3_to_tone2, with v_to_u=False. */
function tone3ToTone2(tone3: string): string {
  const noNumber = tone3.replace(DIGITS, "").replaceAll("ü", "v");
  const markIndex = rightMarkIndex(noNumber) ?? noNumber.length - 1;
  const number = DIGIT.exec(tone3)?.[0];
  if (number === undefined) return tone3;
  return `${noNumber.slice(0, markIndex + 1)}${number}${noNumber.slice(markIndex + 1)}`.replaceAll("ü", "v");
}

/** _tone_convert.tone_to_tone2, with v_to_u=False and neutral_tone_with_five=False. */
function toneToTone2(tone: string): string {
  return tone3ToTone2(toTone3Style(tone));
}

/** _tone_convert.tone2_to_tone */
function tone2ToTone(tone2: string): string {
  const s = tone2.replaceAll("ü", "v").replaceAll("5", "").replaceAll("0", "");
  return s.replace(RE_TONE2_ANYWHERE, (m) => SYMBOL_REVERSE.get(m) || m).replaceAll("v", "ü");
}

/** contrib.tone_sandhi.ToneSandhiMixin._third_tone */
function thirdTone(items: string[]): string[] {
  const t2 = items.map(toneToTone2);
  if (!t2.join("").includes("3")) return items;
  let changed = false;
  let thirdNum = 0;
  for (const p of t2) thirdNum = p.includes("3") ? thirdNum + 1 : 0;
  if (thirdNum === 2) {
    const i = t2.findIndex((v) => v.includes("3"));
    t2[i] = (t2[i] as string).replaceAll("3", "2");
    changed = true;
  } else if (thirdNum > 2) {
    // as upstream: the first thirdNum - 1 third tones anywhere in the word, not only in the final run
    let n = 1;
    for (let i = 0; i < t2.length; i++) {
      const v = t2[i] as string;
      if (!v.includes("3")) continue;
      if (n === thirdNum) break;
      t2[i] = v.replaceAll("3", "2");
      changed = true;
      n++;
    }
  }
  return changed ? t2.map(tone2ToTone) : items;
}

/** _bu and _yi, which differ only in the character and the tone of a final one. */
function buYi(han: string[], items: string[], ch: string, finalTone: string): string[] {
  if (!han.includes(ch)) return items;
  const t2 = items.map(toneToTone2);
  let changed = false;
  for (let i = 0; i < han.length; i++) {
    if (han[i] !== ch) continue;
    const current = t2[i] as string;
    if (i < han.length - 1) {
      // as upstream: before a fourth tone, "4" becomes "2", so yi1 stays yi1
      t2[i] = (t2[i + 1] as string).includes("4") ? current.replaceAll("4", "2") : current.replace(DIGITS, "4");
    } else t2[i] = current.replace(DIGITS, finalTone);
    changed = true;
  }
  return changed ? t2.map(tone2ToTone) : items;
}

export class Pinyin {
  private readonly prefixes = new Set<string>();

  private constructor(
    private readonly chars: Map<string, string>,
    private readonly phrases: Map<string, string[]>,
  ) {
    for (const word of phrases.keys()) {
      const cps = Array.from(word);
      for (let i = 1; i <= cps.length; i++) this.prefixes.add(cps.slice(0, i).join(""));
    }
  }

  /** Parses pinyin_dict.txt (tools/webgpu/export_pinyin_dict.py). */
  static parse(text: string): Pinyin {
    const chars = new Map<string, string>();
    const phrases = new Map<string, string[]>();
    for (const line of text.split(/\r?\n/)) {
      if (!line || line.startsWith("#")) continue;
      const tab = line.indexOf("\t");
      const key = line.slice(0, tab);
      const value = line.slice(tab + 1);
      if (Array.from(key).length === 1) chars.set(key, value);
      else phrases.set(key, value.split(" "));
    }
    return new Pinyin(chars, phrases);
  }

  static async load(url: URL | string = PINYIN_DICT_URL): Promise<Pinyin> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`pinyin dictionary: HTTP ${response.status} for ${url}`);
    return Pinyin.parse(await response.text());
  }

  /** lazy_pinyin(text, style=Style.TONE3, tone_sandhi=True), errors="default", strict=True. */
  lazyPinyin(text: string): string[] {
    const out: string[] = [];
    for (const run of simpleSeg(text)) {
      if (!isHan(Array.from(run)[0] ?? "")) {
        if (run) out.push(run); // handle_nopinyin with errors="default": the run as it is, one item
        continue;
      }
      for (const word of this.mmseg(run)) out.push(...this.convert(word));
    }
    return out;
  }

  /** seg.mmseg.Seg.cut with no_non_phrases=True: forward maximum matching over phrase prefixes. */
  private mmseg(text: string): string[] {
    const words: string[] = [];
    let remain = Array.from(text);
    while (remain.length > 0) {
      let lastValid = 0;
      let all = true;
      for (let index = 0; index < remain.length; index++) {
        const word = remain.slice(0, index + 1).join("");
        if (this.prefixes.has(word)) {
          if (this.phrases.has(word)) lastValid = index + 1;
          continue;
        }
        all = false;
        if (lastValid) {
          words.push(remain.slice(0, lastValid).join(""));
          remain = remain.slice(lastValid);
        } else {
          words.push(remain[0] as string);
          remain = remain.slice(1);
        }
        break;
      }
      if (!all) continue;
      // the whole remainder is a prefix
      if (lastValid) {
        words.push(remain.slice(0, lastValid).join(""));
        remain = remain.slice(lastValid);
      } else {
        words.push(...remain);
        break;
      }
    }
    return words;
  }

  /** DefaultConverter.convert for a Han word: lookup, tone sandhi (post_pinyin), then TONE3. */
  private convert(word: string): string[] {
    const han = Array.from(word);
    let items = this.phrases.get(word) ?? han.map((c) => this.chars.get(c) ?? c);
    items = thirdTone(items);
    items = buYi(han, items, "不", "4");
    items = buYi(han, items, "一", "1");
    return items.map((p) => toTone3Style(p)); // _remove_dup_and_empty: no item is empty
  }
}

/** seg.simpleseg._seg: maximal runs of Han and of non-Han characters. "" gives [""], as upstream. */
function simpleSeg(text: string): string[] {
  const runs: string[] = [];
  let s = "";
  let flag: boolean | undefined;
  for (const c of text) {
    const han = isHan(c);
    if (flag === undefined || flag === han) s += c;
    else {
      runs.push(s);
      s = c;
    }
    flag = han;
  }
  runs.push(s);
  return runs;
}
