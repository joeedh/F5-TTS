# The text front end in the browser (plan 03)

What upstream's text front end does, written out exactly enough to port, and how each part was checked. The plan is `docs/plans/webgpu-03-text-frontend.md`; the code is `web/src/text/`.

## Segmentation: jieba-rs 0.11.0 (step 3)

- rjieba 0.2.2 embeds jieba-rs 0.11.0 (its SBOM, in `rjieba-*.dist-info/sboms`). `rjieba.cut(text)` is `Jieba::new().cut(text, hmm=true)`.
- `jieba-wasm` 2.4.0 embeds a fork of jieba-rs 0.7.0. It splits `x-ray`, `v1.2.3` and `Wi-Fi`, which 0.11.0 keeps whole, and that changes F5-TTS tokens. So the port builds 0.11.0 itself: `tools/webgpu/jieba_wasm/`.
- For text with no CJK ideograph (jieba-rs's `is_cjk`), jieba-rs reduces to a short rule, ported in `segment_ascii.ts`:
  - Text splits into blocks of `[A-Za-z0-9+#&._%-]` and the characters between them.
  - Between blocks, each character is a token, except `\r\n`, which is one.
  - Within a block, the five pure-ASCII dictionary words (`AT&T`, `C#`, `C++`, `c#`, `c++`) are always tokens. A word scores `ln 3 - L` against `-L` per character, with `L = ln(total frequency)`, about 18, and no word can overlap another.
  - The rest of a block goes through the HMM's skip splitter: `skip_match_len` runs (alphanumerics joined across one `.`, `_` or `-` at a time, then an optional `%`), and each gap between runs is one token.
- Results: zero mismatches against rjieba, for the wasm and for the TS path (see the plan's step 3 result).

## Pinyin: pypinyin 0.55.0's `lazy_pinyin` (step 4)

`convert_char_to_pinyin` calls `lazy_pinyin(text, style=Style.TONE3, tone_sandhi=True)`, with the defaults `errors='default'` and `strict=True`. That goes through `core.lazy_pinyin` to `Pinyin(UltimateConverter(tone_sandhi=True)).lazy_pinyin`. With a string input, the path is the following.

### Segmenting (`seg/simpleseg.py`, `seg/mmseg.py`)

- `simple_seg`: maximal runs of characters that do and do not match `RE_HANS`.
  - `RE_HANS` is wider than jieba's CJK and F5-TTS's `is_chinese`. It includes `〇` (U+3007), U+E815 to U+E864 (private use), CJK extension A, the main block, U+F900 to U+FAFF, and a list of astral ranges. The port copies the class range for range.
  - An empty string gives `['']`.
- A non-Han run is converted whole (below). A Han run is cut by `mmseg.seg`, a `Seg` with `no_non_phrases=True`, trained on every `PHRASES_DICT` key:
  - Walk prefixes of the remaining text while they are in the prefix set, remembering the longest one that is itself a phrase.
  - At the first prefix that is not in the set: yield the longest phrase if there was one, otherwise one character. Then start again after it.
  - If the whole remainder is a prefix: yield the longest phrase and continue after it, or, if there was none, yield every remaining character singly and stop.
  - So every word it yields is a `PHRASES_DICT` key or a single character.

### Converting a word (`converter.py`)

- A non-Han run: `handle_nopinyin` with `errors='default'` returns the run itself, as one item. The empty string gives no item.
- A Han word (`DefaultConverter.convert`):
  1. `_phrase_pinyin`: the `PHRASES_DICT` entry if the word is one, otherwise per character `PINYIN_DICT[ord(c)].split(',')`, or `[c]` if the character has no entry.
  2. `post_pinyin`, which is the tone sandhi below, on the tone-marked readings.
  3. `convert_styles`: each item's first reading through `Style.TONE3`.
  4. `_remove_dup_and_empty`: an empty item becomes `''`. No reading in the data is empty.
- With `heteronym=False`, only the first reading of each item is ever read, so `export_pinyin_dict.py` keeps only first readings.
  - The data: 41,923 characters and 47,111 phrases.
  - Every phrase is all Han, longer than one character, and has one reading per character.
  - No reading is empty, and none contains a comma or a space.

### Tone styles (`style/tone.py`, `style/_tone_convert.py`, `style/_utils.py`)

- `Style.TONE3` is registered to `ToneConverter.to_tone3`, not to the module function `to_tone3`. It does:
  - `replace_symbol_to_number`: each single-character tone-marked vowel becomes letter plus digit (`ā` to `a1`, `ü` to `v`, `ǜ` to `v4`). Then the four two-character keys (`m̄`, `m̀`, `ê̄`, `ê̌`) are replaced, in that order.
  - `RE_TONE3 = ^([a-zêü]+)([1-5])([a-zêü]*)$` with `\1\3\2`, which moves the digit to the end. A neutral tone has no digit, so `de` stays `de`.
- The sandhi works in TONE2 (digit after the marked vowel):
  - `tone_to_tone2` is `tone3_to_tone2(ToneConverter.to_tone3(p))`.
  - `tone3_to_tone2` removes the digits and turns `ü` into `v`. It places the first digit after `right_mark_index` (or after the last character). With no digit, it returns its input unchanged.
  - `tone2_to_tone` turns `ü` into `v` and drops `5` and `0`. It replaces each `[aeoiuvnmêü][1-5]` from the reverse symbol table, then turns `v` back into `ü`.
- Python's `\d` matches any Unicode decimal digit. The port uses `\p{Nd}`.

### Tone sandhi (`contrib/tone_sandhi.py`)

Applied to one word at a time, in this order. Each step converts the word's readings to TONE2. If it changes anything, it converts all of them back with `tone2_to_tone`.

- `_third_tone`:
  - Counts the run of third tones that ends the word (`third_num`).
  - If it is 2, the first third tone in the word becomes second.
  - If it is more than 2, the first `third_num - 1` third tones in the word become second. They are counted from the start of the word, not from the start of the final run. No phrase in the data has a third tone before the final run, so this quirk is not reachable today.
- `_bu`, for each `不`:
  - Followed by anything: `4` becomes `2` if the next reading has a `4`, otherwise every digit becomes `4`.
  - Last in the word: every digit becomes `4`.
- `_yi`, the same for `一`, except that a final `一` gets `1`.
  - `一` is `yi1`. Before a fourth tone, `"yi1".replace('4', '2')` changes nothing, so `一次` stays `yi1 ci4`. Before any other tone it becomes `yi4`.
- Sandhi never crosses words. mmseg splits `版本` into `版` and `本`, so it stays `ban3 ben3`.

### How the port was checked

- `web/test/text/pinyin.test.ts` compares `Pinyin.lazyPinyin` with pypinyin on:
  - every `lazy_pinyin` call the edge cases and the Seed-TTS corpus made;
  - and `pinyin_cases.json` (`tools/webgpu/pinyin_reference.py`, local only): every dictionary character, every phrase, and 30,000 random mixes of phrases, characters, `一`, `不`, third tones and non-Han pieces.
- Every word `mmseg` yields is a phrase or a single character, and all of those are in `pinyin_cases.json`. So the conversion and the sandhi are checked on every input they can receive. Segmenting is checked by the random mixes and the corpus.
- `pinyin-pro` 3.29.5, for comparison: on all-Han calls, it differs from pypinyin on 4.36% of characters in the Chinese corpus. Examples: `了` alone is `liao3`, `长` is `chang2`, `还给` is `hai2 gei3`, and `可以` stays `ke3 yi3`.

## Size and loading (step 6)

Measured with `web/bench/text/run.mjs`, 2026-11-22, on the laptop.
- The bench builds `bench/text/` with Vite's production build and serves it as a CDN would: brotli 11 when the browser accepts it, and `application/wasm`.
- Each of 3 runs uses a fresh browser context, so its HTTP cache starts cold.
- Results are in `web/bench/text/results/{chrome,msedge}.json`.

### Transfer sizes (the web build's assets)

| Asset | Raw | gzip -9 | brotli 11 (what Chrome received) |
|---|---|---|---|
| `jieba-*.wasm` | 5.65 MB | 2.14 MB | 1.70 MB |
| `pinyin_dict-*.txt` | 1.60 MB | 670 KB | 541 KB |
| the bench page's JS (all of `web/src/text/`) | 10 KB | 4.6 KB | 4.0 KB |

- Vite emits both assets with hashed names from `new URL("./…", import.meta.url)`, so they can be cached forever.
- Chrome's resource timing shows `transferSize` at the brotli size, and `decodedBodySize` at the raw size.
- Over a real link the 2.25 MB is the main cost: about 0.9 s at 20 Mbit/s. That is small next to about 1.3 GB of weights, but it lands on the first Chinese sentence unless it is started earlier.

### Time (Chrome 158 and Edge 158, medians of 3 cold runs)

| | Chrome | Edge |
|---|---|---|
| First Chinese sentence, both assets cold (local server, so almost no network time) | 262 ms | 225 ms |
| Jieba wasm load: fetch, compile, instantiate, dictionary build | 260 ms | 208 ms |
| of which compile | 4 ms | 3 ms |
| of which instantiate and build the dictionary | 94 ms | 78 ms |
| Pinyin dictionary load: fetch and parse | 124 ms | 124 ms |
| of which parse | 69 ms | 52 ms |
| A typical sentence after loading (84 to 125 bytes, en, zh or mixed) | 0.1 to 0.2 ms | 0.1 ms |

- The two loads start together (`Promise.all` in `TextFrontend.tokens`). Locally that does not shorten the first sentence, because both finish on the main thread. On a real network the fetches overlap.
- The main thread sees two long tasks during the first Chinese sentence: about 80 to 110 ms for the dictionary build in the wasm, and 80 to 120 ms around the dictionary parse.
- After loading, the text front end costs nothing next to one DiT step (400 ms or more).

### Loading behaviour (D2)

- Text with no CJK ideograph and no `RE_HANS` character fetches neither asset. In every run the English sentence left both loaders uncalled, and resource timing listed no Chinese asset.
- The first Chinese sentence loaded each asset once, and later sentences, including mixed ones, reused them.
- Plan 06's loading strategy follows from these numbers:
  - The code is 4 KB, so it stays in the main bundle. Only the two data assets load on demand.
  - Plan 06 should start them as soon as the text box holds a CJK or Han character, for example by calling `TextFrontend.tokens` on input, rather than when generation starts. That hides the network time.
  - Running the text front end in a worker would remove the two long tasks, about 200 ms in all, from the main thread. That is worth it only if the UI must stay responsive while it loads.
